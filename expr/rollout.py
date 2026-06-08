import collections
import torch
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from model import *
from utils import PushTEnv, PushTDataset, normalize_data, unnormalize_data
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# --- 超参数设置 ---
obs_horizon = 2
action_horizon = 8
pred_horizon = 16
obs_dim = 5
action_dim = 2
max_steps = 1000
num_rollouts = 100  # 设定 100 次随机 sample

# 0. 创建环境
env = PushTEnv()

# 创建数据集以获取统计信息
dataset = PushTDataset(
    dataset_path="pusht_cchi_v7_replay.zarr",
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    action_horizon=action_horizon,
)
stats = dataset.stats

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------- 1. 初始化模型 -------------------------
# Diffusion Policy
dp_noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon,
    updownsample_type='Conv',
    sin_embedding_scale=1,
)
state_dict_dp = torch.load("ckpt/dp_noise_pred_net_ema.pth", map_location=device)
ema_noise_pred_net_dp = dp_noise_pred_net
ema_noise_pred_net_dp.load_state_dict(state_dict_dp)
ema_noise_pred_net_dp.to(device)
ema_noise_pred_net_dp.eval()

num_diffusion_iters = 100
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    beta_schedule='squaredcos_cap_v2',
    clip_sample=True,
    prediction_type='epsilon'
)

# Streaming Flow Policy
sfp_velocity_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim * obs_horizon,
    updownsample_type='Linear',
    sin_embedding_scale=100,
)
state_dict_sfp = torch.load("ckpt/ema_spf_velocity_net_ema.pth", map_location=device)
ema_spf_velocity_net = sfp_velocity_net
ema_spf_velocity_net.load_state_dict(state_dict_sfp)
ema_spf_velocity_net.to(device)
ema_spf_velocity_net.eval()


# ------------------ 2. SFP Rollout ---------------------------------------
all_rollouts_rewards_sfp = []

print("=== 开始评估 Streaming Flow Policy ===")
for trial in range(num_rollouts):
    env.seed(1000 + trial)
    obs, info = env.reset()
    
    obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
    trial_rewards = []
    done = False
    step_idx = 0

    a = obs[:action_dim]
    na = normalize_data(a, stats=stats['action'])
    na = torch.from_numpy(na).to(device, dtype=torch.float32)
    na_from_prev_chunk = na.unsqueeze(0).unsqueeze(0)

    Δt = 1.0 / (pred_horizon - obs_horizon)

    with torch.no_grad():
        while not done and step_idx < max_steps:
            obs_stack = np.stack(obs_deque)
            nobs = normalize_data(obs_stack, stats=stats['obs'])
            o_test = torch.from_numpy(nobs).to(device, dtype=torch.float32)
            o_test = o_test.flatten().unsqueeze(0)

            na = na_from_prev_chunk

            for i in range(action_horizon):
                a = na.detach().to('cpu').numpy().squeeze(axis=(0, 1))
                a = unnormalize_data(a, stats=stats['action'])
                
                obs, reward, done, _, info = env.step(a)
                obs_deque.append(obs)
                trial_rewards.append(reward)
                
                step_idx += 1
                if step_idx >= max_steps: done = True
                if done: break

                t = torch.tensor(i * Δt, device=device)
                nv = ema_spf_velocity_net(
                    sample=na,
                    timestep=t,
                    global_cond=o_test,
                )
                na = na + nv * Δt

            na_from_prev_chunk = na

    # 填充与截断
    if len(trial_rewards) < max_steps:
        trial_rewards.extend([1.0] * (max_steps - len(trial_rewards)))
    trial_rewards = trial_rewards[:max_steps]
    all_rollouts_rewards_sfp.append(trial_rewards)
    
    print(f"SFP Rollout {trial+1}/{num_rollouts} 已完成. 实际步数: {step_idx}")


# ------------------ 3. DP Rollout ---------------------------------------
all_rollouts_rewards_dp = []

print("\n=== 开始评估 Diffusion Policy ===")
for trial in range(num_rollouts):
    # 【修复】每次 trial 必须重新设置种子和重置环境状态
    env.seed(1000 + trial)
    obs, info = env.reset()
    obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
    
    trial_rewards = []  # 【修复】每一轮独立收集奖励
    done = False
    step_idx = 0
    
    with torch.no_grad():
        while not done and step_idx < max_steps:
            obs_stack = np.stack(obs_deque)
            nobs = normalize_data(obs_stack, stats=stats['obs'])
            nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)
            obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)  # (1, To * O)

            # 初始化高斯噪声
            na_traj = torch.randn((1, pred_horizon, action_dim), device=device)
            noise_scheduler.set_timesteps(num_diffusion_iters)

            # Denoising 循环
            for k in noise_scheduler.timesteps:
                noise_pred = ema_noise_pred_net_dp(
                    sample=na_traj,
                    timestep=k,
                    global_cond=obs_cond
                )
                na_traj = noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=na_traj,
                ).prev_sample

            na_traj = na_traj.detach().to('cpu').numpy()[0]  # (Tp, A)
            a_traj = unnormalize_data(na_traj, stats=stats['action'])

            # 裁剪 action_horizon
            start = obs_horizon - 1
            end = start + action_horizon
            a_traj = a_traj[start:end, :]

            # 执行 Action Chunk
            for action in a_traj:
                obs, reward, done, _, info = env.step(action)
                obs_deque.append(obs)
                trial_rewards.append(reward)  # 【修复】存入当轮的单独列表中

                step_idx += 1
                if step_idx >= max_steps: done = True
                if done: break

    # 【修复】正确填充 DP 的奖励数据
    if len(trial_rewards) < max_steps:
        trial_rewards.extend([1.0] * (max_steps - len(trial_rewards)))
    trial_rewards = trial_rewards[:max_steps]
    all_rollouts_rewards_dp.append(trial_rewards)
    
    print(f"DP Rollout {trial+1}/{num_rollouts} 已完成. 实际步数: {step_idx}")


# ------------------ 4. 数据统计与双曲线绘制 ---------------------------------------
all_rollouts_rewards_sfp = np.array(all_rollouts_rewards_sfp)
mean_rewards_sfp = np.mean(all_rollouts_rewards_sfp, axis=0)
std_rewards_sfp = np.std(all_rollouts_rewards_sfp, axis=0)

all_rollouts_rewards_dp = np.array(all_rollouts_rewards_dp)
mean_rewards_dp = np.mean(all_rollouts_rewards_dp, axis=0)
std_rewards_dp = np.std(all_rollouts_rewards_dp, axis=0)

plt.figure(figsize=(11, 7))

# 绘制 Streaming Flow Policy 曲线及阴影区
plt.plot(mean_rewards_sfp, label='Streaming Flow Policy (Mean)', color='#1f77b4', linewidth=2.5)
plt.fill_between(
    range(max_steps),
    np.clip(mean_rewards_sfp - std_rewards_sfp, 0, 1),
    np.clip(mean_rewards_sfp + std_rewards_sfp, 0, 1),
    color='#1f77b4', alpha=0.15
)

# 绘制 Diffusion Policy 曲线及阴影区
plt.plot(mean_rewards_dp, label='Diffusion Policy (Mean)', color='#ff7f0e', linewidth=2.5)
plt.fill_between(
    range(max_steps),
    np.clip(mean_rewards_dp - std_rewards_dp, 0, 1),
    np.clip(mean_rewards_dp + std_rewards_dp, 0, 1),
    color='#ff7f0e', alpha=0.15
)

# 图表美化
plt.title(f'Step-Reward Comparison ({num_rollouts} Rollouts)', fontsize=14, fontweight='bold')
plt.xlabel('Step', fontsize=12)
plt.ylabel('Reward (Target Coverage)', fontsize=12)
plt.xlim(0, max_steps)
plt.ylim(0, 1.05)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='lower right', fontsize=11)

# 保存并展示
plt.savefig('policy_comparison_curve.png', dpi=300, bbox_inches='tight')
print("\n对比曲线图已成功保存为 'policy_comparison_curve.png'")
plt.show()