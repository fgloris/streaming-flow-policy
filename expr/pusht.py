# Standard imports
import collections
from dataclasses import dataclass
import gdown
import os
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

# always call this first
from streaming_flow_policy.all import set_random_seed
set_random_seed(0)

from utils import PushTDataset
from model import *

device = torch.device('cuda')

# Download demonstration data from Google Drive
#dataset_path = "pusht_cchi_v7_replay.zarr.zip"
#if not os.path.isfile(dataset_path):
#    id = "1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq&confirm=t"
#    gdown.download(id=id, output=dataset_path, quiet=False)

# |o|o|                             observations: 2
# | |a|a|a|a|a|a|a|a|               actions executed: 8
# |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

obs_horizon = 2
action_horizon = 8
pred_horizon = 16

# Create dataset from file
dataset = PushTDataset(
    dataset_path="pusht_cchi_v7_replay.zarr",
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    action_horizon=action_horizon,
)
# Save training data statistics (min, max) for each dim
stats = dataset.stats

# Create dataloader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    num_workers=1,
    shuffle=True,
    pin_memory=True,  # accelerate cpu-gpu transfer
    persistent_workers=True, # don't kill worker process after each epoch
)

# Visualize data in batch
batch = next(iter(dataloader))
print("batch['obs'].shape:", batch['obs'].shape)
print("batch['action'].shape", batch['action'].shape)

obs_horizon = 2
obs_dim = 5
action_dim = 2

# ----------------- baseline: diffusion policy -------------------------
# Create network object
dp_noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon,
    updownsample_type = 'Conv',
    sin_embedding_scale = 1,  # original setting
)

num_diffusion_iters = 100
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule='squaredcos_cap_v2',
    # clip output to [-1,1] to improve stability
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    prediction_type='epsilon'
)
# ------------------ streaming flow policy -----------------------------
sfp_velocity_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon,
    # because SFP diffuses over a single action,
    updownsample_type = 'Linear',
    # because the original model assumes timesteps of the order of [0, 100]
    # but SFP uses a time range of [0, 1]
    sin_embedding_scale = 100,
)


import os
checkpoint_dir = "ckpt"
os.makedirs(checkpoint_dir, exist_ok=True)

train_diffusion = True
if train_diffusion:
    num_epochs = 100

    dp_noise_pred_net.to(device)
    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema_dp = EMAModel(
        parameters=dp_noise_pred_net.parameters(),
        power=0.75)

    # Standard ADAM optimizer
    # Note that EMA parametesr are not optimized
    optimizer = torch.optim.AdamW(
        params=dp_noise_pred_net.parameters(),
        lr=1e-4, weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataset) * num_epochs
    )

    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
            # batch loop
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    # Note that the data is normalized in the dataset.
                    # Device transfer
                    nobs = nbatch['obs'].to(device)  # (B, To, O)
                    naction = nbatch['action'].to(device)  # (B, Tp, A)
                    B = nobs.shape[0]

                    # Observation as FiLM conditioning
                    obs_cond = nobs.flatten(start_dim=1)  # (B, To*O)

                    # Sample noise to add to actions
                    noise = torch.randn(naction.shape, device=device)  # (B, Tp, A)

                    # sample a diffusion iteration for each data point
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (B,), device=device
                    ).long()  # (B,)

                    # Forward diffusion process: Add noise to the clean images
                    # according to the noise magnitude at each diffusion iteration.
                    noisy_actions = noise_scheduler.add_noise(
                        naction, noise, timesteps)  # (B, Tp, A)

                    # Predict the noise residual.
                    noise_pred = dp_noise_pred_net(
                        noisy_actions, timesteps, global_cond=obs_cond)

                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()

                    # update Exponential Moving Average of the model weights
                    ema_dp.step(dp_noise_pred_net.parameters())

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
            tglobal.set_postfix(loss=np.mean(epoch_loss))

    # Weights of the EMA model
    # is used for inference
    ema_noise_pred_net_dp = dp_noise_pred_net
    ema_dp.copy_to(ema_noise_pred_net_dp.parameters())

    # save state_dict
    model_save_path = os.path.join(checkpoint_dir, "dp_noise_pred_net_ema.pth")
    torch.save(ema_noise_pred_net_dp.state_dict(), model_save_path)

    print(f"Diffusion policy 训练完成！ EMA 模型权重已成功保存至: {model_save_path}")

train_sfp = True
if train_sfp:
    σ0 = 0.4
    k = 10
    num_epochs = 100
    
    sfp_velocity_net.to(device)
    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(
        parameters=sfp_velocity_net.parameters(),
        power=0.75)

    # Standard ADAM optimizer
    # Note that EMA parametesr are not optimized
    optimizer = torch.optim.AdamW(
        params=sfp_velocity_net.parameters(),
        lr=1e-4, weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs
    )

    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
            # batch loop
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    # Device transfer
                    # Note that data is already normalized in the dataset.
                    nobs = nbatch['obs'].to(device)  # (B, To, O)
                    naction = nbatch['action'].to(device)  # (B, Tp, A)

                    # SFP integrates actions starting from the current timestep.
                    # But sequences extracted from the PushTDataset include actions
                    # corresponding to the previous timesteps as well (Tp includes
                    # To - 1 previous actions). The next line removes those.
                    ξ = naction[:, obs_horizon-1:, :]  # (B, Tp - To + 1, A)

                    # Sample t uniformly from [0, 1].
                    t = torch.rand(ξ.shape[0]).float().to(device)  # (B,)

                    ξt, dξdt = LinearlyInterpolateTrajectory(ξ, t)  # (B, A) and (B, A)
                    a, v = SampleCFMInputsAndTargets(ξt, dξdt, t, k, σ0)  # (B, A) and (B, A)
                    a, v = a.unsqueeze(1), v.unsqueeze(1)  # (B, 1, A) and (B, 1, A)
                    
                    # Conditional flow matching (CFM) loss: Mean-squared error
                    # between predicted velocity and target velocity
                    v̂t = sfp_velocity_net(
                        sample=a,
                        timestep=t,
                        global_cond=nobs.flatten(start_dim=1),
                    )  # (B, 1, A)
                    loss = nn.functional.mse_loss(v, v̂t)  # (,) L2 loss

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()

                    # update Exponential Moving Average of the model weights
                    ema.step(sfp_velocity_net.parameters())

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
            tglobal.set_postfix(loss=np.mean(epoch_loss))

    # Weights of the EMA model
    # is used for inference
    ema_spf_velocity_net = sfp_velocity_net
    ema.copy_to(ema_spf_velocity_net.parameters())

    # save state_dict
    model_save_path = os.path.join(checkpoint_dir, "ema_spf_velocity_net_ema.pth")
    torch.save(ema_spf_velocity_net.state_dict(), model_save_path)

    print(f"Streaming Flow policy 训练完成！ EMA 模型权重已成功保存至: {model_save_path}")

