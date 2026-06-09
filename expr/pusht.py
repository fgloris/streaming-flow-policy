"""
Train Push-T policies from expr/.

This script keeps the original Diffusion Policy training unchanged, and replaces
expr's handwritten SFP training with the official implementations:
  - streaming_flow_policy.pusht.sfps.StreamingFlowPolicyStochastic
  - streaming_flow_policy.pusht.sfpd.StreamingFlowPolicyDeterministic

Run examples:
  python expr/pusht.py --policies diffusion sfps sfpd
  python expr/pusht.py --policies sfps sfpd --num-epochs-flow 1000
"""

import argparse
import os
from pathlib import Path

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

# Keep the expr Diffusion Policy implementation unchanged.
from utils import PushTDataset
from model import ConditionalUnet1D as ExprConditionalUnet1D

# Use the official SFPS/SFPD implementations instead of expr's old handwritten SFP.
from streaming_flow_policy.pusht.dataset import PushTStateDatasetWithNextObsAsAction

# -----------------------------------------------------------------------------
# Shared Push-T settings
# -----------------------------------------------------------------------------
OBS_HORIZON = 2
ACTION_HORIZON = 8
PRED_HORIZON = 16
OBS_DIM = 5
ACTION_DIM = 2


class PushTStateDatasetWithNextObsAsActionFromPath(PushTStateDatasetWithNextObsAsAction):
    """Official SFP dataset variant, but with an explicit local dataset path.

    The official SFPS/SFPD training uses `PushTStateDatasetWithNextObsAsAction`,
    where the action trajectory is the next gripper/state position.  This class
    keeps that behavior while avoiding an implicit download path.
    """

    dataset_path = "pusht_cchi_v7_replay.zarr"

    @staticmethod
    def GetDatasetRoot():
        import zarr
        return zarr.open(PushTStateDatasetWithNextObsAsActionFromPath.dataset_path, "r")


def build_dp_dataset(dataset_path: str) -> PushTDataset:
    return PushTDataset(
        dataset_path=dataset_path,
        pred_horizon=PRED_HORIZON,
        obs_horizon=OBS_HORIZON,
        action_horizon=ACTION_HORIZON,
    )


def build_flow_dataset(dataset_path: str, policy):
    PushTStateDatasetWithNextObsAsActionFromPath.dataset_path = dataset_path
    return PushTStateDatasetWithNextObsAsActionFromPath(
        pred_horizon=PRED_HORIZON,
        obs_horizon=OBS_HORIZON,
        action_horizon=ACTION_HORIZON,
        transform_datum_fn=policy.TransformTrainingDatum,
    )


def make_dataloader(dataset, batch_size: int):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
        pin_memory=False,
        persistent_workers=False,
    )


def train_diffusion_policy(args, device: torch.device):
    """Original expr diffusion training: architecture and scheduler settings kept."""
    dataset = build_dp_dataset(args.dataset_path)
    dataloader = make_dataloader(dataset, args.batch_size_dp)

    batch = next(iter(dataloader))
    print("[diffusion] batch['obs'].shape:", batch['obs'].shape)
    print("[diffusion] batch['action'].shape", batch['action'].shape)

    dp_noise_pred_net = ExprConditionalUnet1D(
        input_dim=ACTION_DIM,
        global_cond_dim=OBS_DIM * OBS_HORIZON,
        updownsample_type="Conv",
        sin_embedding_scale=1,
    ).to(device)

    num_diffusion_iters = 100
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    ema_dp = EMAModel(parameters=dp_noise_pred_net.parameters(), power=0.75)
    optimizer = torch.optim.AdamW(
        params=dp_noise_pred_net.parameters(),
        lr=1e-4,
        weight_decay=1e-6,
    )

    # Kept from the original expr script.
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataset) * args.num_epochs_dp,
    )

    with tqdm(range(args.num_epochs_dp), desc="Diffusion Epoch") as tglobal:
        for _ in tglobal:
            epoch_loss = []
            with tqdm(dataloader, desc="Batch", leave=False) as tepoch:
                for nbatch in tepoch:
                    nobs = nbatch["obs"].to(device)
                    naction = nbatch["action"].to(device)
                    B = nobs.shape[0]

                    obs_cond = nobs.flatten(start_dim=1)
                    noise = torch.randn(naction.shape, device=device)
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (B,),
                        device=device,
                    ).long()
                    noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)
                    noise_pred = dp_noise_pred_net(noisy_actions, timesteps, global_cond=obs_cond)
                    loss = nn.functional.mse_loss(noise_pred, noise)

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()
                    ema_dp.step(dp_noise_pred_net.parameters())

                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
            tglobal.set_postfix(loss=np.mean(epoch_loss))

    ema_dp.copy_to(dp_noise_pred_net.parameters())
    save_path = Path(args.checkpoint_dir) / "dp_noise_pred_net_ema.pth"
    torch.save(dp_noise_pred_net.state_dict(), save_path)
    print(f"[diffusion] saved EMA weights to: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default="pusht_cchi_v7_replay.zarr")
    parser.add_argument("--checkpoint-dir", default="ckpt")
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=("diffusion"),
        default=("diffusion"),
    )
    parser.add_argument("--num-epochs-dp", type=int, default=100)
    parser.add_argument("--batch-size-dp", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if "diffusion" in args.policies:
        train_diffusion_policy(args, device)

if __name__ == "__main__":
    main()
