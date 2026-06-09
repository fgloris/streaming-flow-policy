"""
Roll out and compare three Push-T policies:
  1. Diffusion Policy            ckpt/dp_noise_pred_net_ema.pth
  2. SFPS (official stochastic)  ckpt/pusht_sfps_obs_ema.pth
  3. SFPD (official deterministic) ckpt/pusht_sfpd_obs_ema.pth

SFPS/SFPD directly reference:
  - streaming_flow_policy.pusht.sfps.StreamingFlowPolicyStochastic
  - streaming_flow_policy.pusht.sfpd.StreamingFlowPolicyDeterministic
"""

import argparse
import collections
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from tqdm.auto import tqdm

from model import ConditionalUnet1D as ExprConditionalUnet1D
from utils import PushTEnv, PushTDataset, normalize_data, unnormalize_data

from streaming_flow_policy.pusht.dataset import PushTStateDatasetWithNextObsAsAction
from streaming_flow_policy.pusht.dp_state_notebook.network import ConditionalUnet1D as OfficialConditionalUnet1D
from streaming_flow_policy.pusht.sfps import StreamingFlowPolicyStochastic
from streaming_flow_policy.pusht.sfpd import StreamingFlowPolicyDeterministic


OBS_HORIZON = 2
ACTION_HORIZON = 8
PRED_HORIZON = 16
OBS_DIM = 5
ACTION_DIM = 2
NUM_DIFFUSION_ITERS = 100


class PushTStateDatasetWithNextObsAsActionFromPath(PushTStateDatasetWithNextObsAsAction):
    dataset_path = "pusht_cchi_v7_replay.zarr"

    @staticmethod
    def GetDatasetRoot():
        import zarr
        return zarr.open(PushTStateDatasetWithNextObsAsActionFromPath.dataset_path, "r")


def build_dp_stats(dataset_path: str):
    dataset = PushTDataset(
        dataset_path=dataset_path,
        pred_horizon=PRED_HORIZON,
        obs_horizon=OBS_HORIZON,
        action_horizon=ACTION_HORIZON,
    )
    return dataset.stats


def build_flow_stats(dataset_path: str):
    PushTStateDatasetWithNextObsAsActionFromPath.dataset_path = dataset_path
    dataset = PushTStateDatasetWithNextObsAsActionFromPath(
        pred_horizon=PRED_HORIZON,
        obs_horizon=OBS_HORIZON,
        action_horizon=ACTION_HORIZON,
    )
    return dataset.stats


def load_diffusion_policy(ckpt_path: str, device: torch.device):
    model = ExprConditionalUnet1D(
        input_dim=ACTION_DIM,
        global_cond_dim=OBS_DIM * OBS_HORIZON,
        updownsample_type="Conv",
        sin_embedding_scale=1,
    )
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    scheduler = DDPMScheduler(
        num_train_timesteps=NUM_DIFFUSION_ITERS,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    return model, scheduler


def load_sfps_policy(ckpt_path: str, device: torch.device, sigma0: float = 0.1, sigma1: float = 0.1):
    velocity_net = OfficialConditionalUnet1D(
        input_dim=ACTION_DIM,
        global_cond_dim=OBS_DIM * OBS_HORIZON,
        fc_timesteps=2,
    )
    policy = StreamingFlowPolicyStochastic(
        velocity_net=velocity_net,
        action_dim=ACTION_DIM,
        pred_horizon=PRED_HORIZON,
        σ0=sigma0,
        σ1=sigma1,
        device=device,
    )
    policy.load_state_dict(torch.load(ckpt_path, map_location=device))
    policy.to(device).eval()
    return policy


def load_sfpd_policy(ckpt_path: str, device: torch.device, sigma: float = 0.1):
    velocity_net = OfficialConditionalUnet1D(
        input_dim=ACTION_DIM,
        global_cond_dim=OBS_DIM * OBS_HORIZON,
        fc_timesteps=1,
    )
    policy = StreamingFlowPolicyDeterministic(
        velocity_net=velocity_net,
        action_dim=ACTION_DIM,
        pred_horizon=PRED_HORIZON,
        sigma=sigma,
        device=device,
    )
    policy.load_state_dict(torch.load(ckpt_path, map_location=device))
    policy.to(device).eval()
    return policy


def rollout_diffusion(env, model, scheduler, stats, seed: int, max_steps: int, device: torch.device):
    env.seed(seed)
    obs, _ = env.reset()
    obs_deque = collections.deque([obs] * OBS_HORIZON, maxlen=OBS_HORIZON)
    rewards = []
    done = False
    step_idx = 0

    with torch.no_grad():
        while not done and step_idx < max_steps:
            obs_stack = np.stack(obs_deque)
            nobs = normalize_data(obs_stack, stats=stats["obs"])
            nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)
            obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

            na_traj = torch.randn((1, PRED_HORIZON, ACTION_DIM), device=device)
            scheduler.set_timesteps(NUM_DIFFUSION_ITERS)
            for k in scheduler.timesteps:
                noise_pred = model(sample=na_traj, timestep=k, global_cond=obs_cond)
                na_traj = scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=na_traj,
                ).prev_sample

            na_traj = na_traj.detach().cpu().numpy()[0]
            a_traj = unnormalize_data(na_traj, stats=stats["action"])
            action_chunk = a_traj[OBS_HORIZON - 1: OBS_HORIZON - 1 + ACTION_HORIZON]

            for action in action_chunk:
                obs, reward, done, _, _ = env.step(action)
                obs_deque.append(obs)
                rewards.append(reward)
                step_idx += 1
                if step_idx >= max_steps:
                    done = True
                if done:
                    break

    return pad_rewards(rewards, max_steps)


def rollout_flow(env, policy, stats, seed: int, max_steps: int, device: torch.device, integration_steps_per_action: int):
    env.seed(seed)
    obs, _ = env.reset()
    obs_deque = collections.deque([obs] * OBS_HORIZON, maxlen=OBS_HORIZON)
    rewards = []
    done = False
    step_idx = 0

    policy_kwargs = {
        "num_actions": 1 + ACTION_HORIZON,
        "integration_steps_per_action": integration_steps_per_action,
    }

    with torch.no_grad():
        while not done and step_idx < max_steps:
            obs_stack = np.stack(obs_deque)
            nobs = normalize_data(obs_stack, stats=stats["obs"])
            nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)

            naction = policy(nobs, **policy_kwargs).detach().cpu().numpy()[0]
            action_pred = unnormalize_data(naction, stats=stats["action"])
            action_chunk = action_pred[OBS_HORIZON - 1: OBS_HORIZON - 1 + ACTION_HORIZON]

            for action in action_chunk:
                obs, reward, done, _, _ = env.step(action)
                obs_deque.append(obs)
                rewards.append(reward)
                step_idx += 1
                if step_idx >= max_steps:
                    done = True
                if done:
                    break

    return pad_rewards(rewards, max_steps)


def pad_rewards(rewards, max_steps: int):
    # Same convention as the original expr rollout: after success/end, pad with 1.0.
    if len(rewards) < max_steps:
        rewards = rewards + [1.0] * (max_steps - len(rewards))
    return np.asarray(rewards[:max_steps], dtype=np.float32)


def evaluate_policy(name, rollout_fn, num_rollouts: int, max_steps: int):
    env = PushTEnv()
    all_rewards = []
    print(f"\n=== Evaluating {name} ===")
    for trial in tqdm(range(num_rollouts), desc=name):
        seed = 1000 + trial
        rewards = rollout_fn(env, seed, max_steps)
        all_rewards.append(rewards)
    all_rewards = np.stack(all_rewards, axis=0)
    return {
        "name": name,
        "rewards": all_rewards,
        "mean": all_rewards.mean(axis=0),
        "std": all_rewards.std(axis=0),
        "final_mean": float(all_rewards[:, -1].mean()),
        "max_mean": float(all_rewards.max(axis=1).mean()),
    }


def plot_results(results, max_steps: int, output_path: str):
    plt.figure(figsize=(11, 7))
    for item in results:
        x = np.arange(max_steps)
        mean = item["mean"]
        std = item["std"]
        plt.plot(x, mean, label=f"{item['name']} mean", linewidth=2.2)
        plt.fill_between(x, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1), alpha=0.15)

    plt.title(f"Step-Reward Comparison ({results[0]['rewards'].shape[0]} Rollouts)", fontsize=14, fontweight="bold")
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Reward / Target Coverage", fontsize=12)
    plt.xlim(0, max_steps)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize=11)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved comparison curve to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default="pusht_cchi_v7_replay.zarr")
    parser.add_argument("--checkpoint-dir", default="ckpt")
    parser.add_argument("--num-rollouts", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--integration-steps-per-action", type=int, default=1)
    parser.add_argument("--output", default="policy_comparison_curve.png")
    parser.add_argument("--sfps-sigma0", type=float, default=0.1)
    parser.add_argument("--sfps-sigma1", type=float, default=0.1)
    parser.add_argument("--sfpd-sigma", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    ckpt_dir = Path(args.checkpoint_dir)

    dp_stats = build_dp_stats(args.dataset_path)
    flow_stats = build_flow_stats(args.dataset_path)

    dp_model, dp_scheduler = load_diffusion_policy(ckpt_dir / "dp_noise_pred_net_ema.pth", device)
    sfps_policy = load_sfps_policy(
        ckpt_dir / "pusht_sfps_obs_ema.pth",
        device,
        sigma0=args.sfps_sigma0,
        sigma1=args.sfps_sigma1,
    )
    sfpd_policy = load_sfpd_policy(
        ckpt_dir / "pusht_sfpd_obs_ema.pth",
        device,
        sigma=args.sfpd_sigma,
    )

    results = []
    results.append(evaluate_policy(
        "Diffusion Policy",
        lambda env, seed, max_steps: rollout_diffusion(env, dp_model, dp_scheduler, dp_stats, seed, max_steps, device),
        args.num_rollouts,
        args.max_steps,
    ))
    #results.append(evaluate_policy(
    #    "SFPS",
    #    lambda env, seed, max_steps: rollout_flow(env, sfps_policy, flow_stats, seed, max_steps, device, args.integration_steps_per_action),
    #    args.num_rollouts,
    #    args.max_steps,
    #))
    #results.append(evaluate_policy(
    #    "SFPD",
    #    lambda env, seed, max_steps: rollout_flow(env, sfpd_policy, flow_stats, seed, max_steps, device, args.integration_steps_per_action),
    #    args.num_rollouts,
    #    args.max_steps,
    #))

    print("\n=== Summary ===")
    for item in results:
        print(f"{item['name']}: final_mean={item['final_mean']:.4f}, mean_of_episode_max={item['max_mean']:.4f}")

    plot_results(results, args.max_steps, args.output)
    plt.show()


if __name__ == "__main__":
    main()
