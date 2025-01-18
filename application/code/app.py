import warnings
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
from collections import deque
import random
import numpy as np
import gym
from tqdm import tqdm
import glob
import os
import imageio

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class DQN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        # 卷积输出大小：64通道，7x7特征图 => 64*7*7 = 3136维
        self.fc1 = nn.Linear(64 * 7 * 7, 512)

        # 使用dropout
        self.dropout = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # 展平: (N,64,7,7) -> (N,3136)
        x = x.view(x.size(0), -1)

        # 使用dropout后再激活
        x = F.relu(self.dropout(self.fc1(x)))

        # 最终输出Q值
        x = self.fc2(x)
        return x


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_shape, num_actions, lr=1e-4, gamma=0.99, buffer_size=100000, batch_size=256,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=50000):
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.policy_net = DQN(state_shape[0], num_actions).to(device)
        self.target_net = DQN(state_shape[0], num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=lr, weight_decay=1e-5)
        self.memory = ReplayBuffer(buffer_size)

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

    def select_action(self, state, is_exploration=True):
        if is_exploration:
            self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                np.exp(-1. * self.steps_done / self.epsilon_decay)
            self.steps_done += 1
            if random.random() < self.epsilon:
                return random.randrange(self.num_actions)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.max(1)[1].item()

    def push_memory(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        state, action, reward, next_state, done = self.memory.sample(
            self.batch_size)

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.LongTensor(action).unsqueeze(1).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(done).unsqueeze(1).to(device)

        q_values = self.policy_net(state).gather(1, action)
        # Double DQN逻辑
        next_actions = self.policy_net(next_state).max(1)[1].unsqueeze(1)
        next_q_values = self.target_net(
            next_state).gather(1, next_actions).detach()

        expected_q_values = reward + self.gamma * next_q_values * (1 - done)
        loss = F.mse_loss(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


def create_action_space():
    steering_options = [-1.0, 0.0, 1.0]
    gas_options = [0.0, 1.0]
    brake_options = [0.0, 0.8]
    action_space = []
    for steer in steering_options:
        for gas in gas_options:
            for brake in brake_options:
                action_space.append([steer, gas, brake])
    return action_space


def preprocess(observation):
    img = Image.fromarray(observation)
    img = img.resize((84, 84))
    img = img.convert('L')
    img = np.array(img, dtype=np.float32)
    img /= 255.0
    img = np.expand_dims(img, axis=0)
    return img


def train_dqn(env, agent, action_space, scheduler, num_episodes=500, target_update=1000, max_steps=1000, save_interval=25):
    episode_rewards = []
    steps = 0
    with tqdm(total=num_episodes, desc="Training Progress") as pbar:
        for episode in range(1, num_episodes + 1):
            obs, info = env.reset(seed=SEED)
            state = preprocess(obs)
            total_reward = 0
            done = False
            step = 0
            while not done and step < max_steps:
                action_idx = agent.select_action(state)
                action = action_space[action_idx]
                next_obs, reward, terminated, truncated, info = env.step(
                    action)
                done = terminated or truncated
                next_state = preprocess(next_obs)
                agent.push_memory(state, action_idx, reward, next_state, done)
                agent.update()
                state = next_state
                total_reward += reward
                step += 1
                steps += 1
                if steps % target_update == 0:
                    agent.update_target_network()

            episode_rewards.append(total_reward)
            if episode % save_interval == 0:
                weight_path = f"dqn_weights_episode_{episode}.pth"
                torch.save(agent.policy_net.state_dict(), weight_path)
                print(f"Saved weights to {weight_path}")

            pbar.set_postfix({
                "Episode": episode,
                "AvgReward": f"{np.mean(episode_rewards[-10:]):.2f}",
                "Epsilon": f"{agent.epsilon:.2f}",
                "LR": agent.optimizer.param_groups[0]['lr']
            })
            pbar.update(1)

            # 每个episode结束后step一下scheduler以调整学习率
            scheduler.step()

            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(
                    f"Episode {episode}, Average Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.2f}, LR: {agent.optimizer.param_groups[0]['lr']:.6f}")
    return episode_rewards


def test_dqn(env, agent, action_space, num_episodes=3, max_steps=1000, render=True):
    agent.policy_net.eval()
    for episode in range(1, num_episodes + 1):
        obs, info = env.reset(seed=SEED + episode)
        state = preprocess(obs)
        total_reward = 0
        done = False
        step = 0
        while not done and step < max_steps:
            action_idx = agent.select_action(state, is_exploration=False)
            action = action_space[action_idx]
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = preprocess(next_obs)
            total_reward += reward
            step += 1
            if render:
                env.render()
        print(f"Test Episode {episode}: Total Reward: {total_reward:.2f}")
    env.close()


def test_dqn_with_gif(env, agent, action_space, num_episodes=1, max_steps=1000, gif_path="car_racing.gif"):
    agent.policy_net.eval()
    frames = []
    for episode in range(1, num_episodes + 1):
        obs, info = env.reset(seed=SEED + episode)
        state = preprocess(obs)
        total_reward = 0
        done = False
        step = 0
        while not done and step < max_steps:
            action_idx = agent.select_action(state, is_exploration=False)
            action = action_space[action_idx]
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = preprocess(next_obs)
            total_reward += reward
            step += 1
            frame = env.render()
            frames.append(frame)

        print(f"Test Episode {episode}: Total Reward: {total_reward:.2f}")
    env.close()
    imageio.mimsave(gif_path, frames, fps=30)
    print(f"GIF saved to {gif_path}")


def get_q_values_over_time(env, agent, action_space, max_steps=1000):
    agent.policy_net.eval()
    obs, info = env.reset(seed=SEED)
    state = preprocess(obs)
    q_values_list = []
    total_reward = 0
    done = False
    step = 0
    while not done and step < max_steps:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = agent.policy_net(state_tensor).cpu().numpy().flatten()
        q_values_list.append(q_values)
        action_idx = np.argmax(q_values)
        action = action_space[action_idx]
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = preprocess(next_obs)
        total_reward += reward
        step += 1
    env.close()
    return q_values_list, total_reward


if __name__ == "__main__":
    env = gym.make('CarRacing-v2', render_mode="rgb_array")
    action_space = create_action_space()
    num_actions = len(action_space)
    print(f"Number of discrete actions: {num_actions}")
    state_shape = (1, 84, 84)  # 灰度图输入

    agent = DQNAgent(state_shape, num_actions)

    # 定义学习率范围
    initial_lr = 1e-4
    final_lr = 1e-5
    num_training_episodes = 2000  # 总训练episodes

    # 定义学习率调度函数：线性从initial_lr到final_lr
    def lr_lambda(epoch):
        # epoch从0开始，最后一个epoch为num_training_episodes
        # 当epoch=0时返回1.0，当epoch=num_training_episodes时返回final_lr/initial_lr
        return final_lr + (initial_lr - final_lr) * (1 - epoch / num_training_episodes) / initial_lr

    # 创建scheduler，注意epoch的定义我们以episode作为单位
    scheduler = optim.lr_scheduler.LambdaLR(
        agent.optimizer, lr_lambda=lambda e: final_lr/initial_lr + (1 - final_lr/initial_lr)*(1 - e/num_training_episodes))

    weight_files = sorted(glob.glob("dqn_weights_episode_*.pth"))

    # 若有已有权重，则加载最新权重并询问是否从此基础继续训练
    if weight_files:
        latest_weight = weight_files[-1]
        print(f"Found existing weights: {latest_weight}")
        user_input = input("Load these weights and continue training? (y/n): ")
        if user_input.lower() == 'y':
            print(
                f"Loading weights from {latest_weight} and continue training...")
            agent.policy_net.load_state_dict(torch.load(latest_weight))
        else:
            print("Not loading existing weights, start from scratch.")
    else:
        print("No saved weights found. Start training from scratch...")

    print("Starting training...")
    rewards = train_dqn(env, agent, action_space,
                        scheduler=scheduler,
                        num_episodes=num_training_episodes,
                        save_interval=10)

    # 绘制训练奖励曲线
    plt.figure(figsize=(12, 5))
    plt.plot(rewards, label='Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Reward Curve')
    plt.legend()
    plt.show()

    # 在训练完成后，列出所有权重文件供选择
    weight_files = sorted(glob.glob("dqn_weights_episode_*.pth"))
    if not weight_files:
        print("No weights available for testing. Exiting...")
        exit()

    print("Available weights:")
    for idx, weight_file in enumerate(weight_files):
        print(f"{idx + 1}: {weight_file}")
    selected_idx = int(
        input("Enter the number of the weight file to load for testing: ")) - 1
    selected_weight = weight_files[selected_idx]
    print(f"Loading selected weights from {selected_weight}...")
    agent.policy_net.load_state_dict(torch.load(selected_weight))

    # 测试并展示GIF
    print("Starting testing...")
    test_dqn(env, agent, action_space, num_episodes=3, render=False)

    print("Starting testing with GIF generation...")
    test_dqn_with_gif(env, agent, action_space, gif_path="car_racing.gif")

    # 最后展示Q值曲线
    q_values_list, total_reward = get_q_values_over_time(
        env, agent, action_space)
    q_values_array = np.array(q_values_list)
    plt.figure(figsize=(12, 6))
    for i in range(agent.num_actions):
        plt.plot(q_values_array[:, i], label=f"Action {i}")
    plt.xlabel("Time Step")
    plt.ylabel("Q-value")
    plt.title(
        f"Q-values over Time ({selected_weight}, Total Reward: {total_reward:.2f})")
    plt.legend()
    plt.show()
