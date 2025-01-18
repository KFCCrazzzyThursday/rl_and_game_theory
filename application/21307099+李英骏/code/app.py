import os
import glob
from tqdm import tqdm
import gym
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import warnings

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
        self.conv1 = nn.Conv2d(
            input_channels, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4,
                               stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3,
                               stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.dropout(self.fc1(x)))
        return self.fc2(x)





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
    def __init__(self, state_shape, num_actions, lr=1e-4, gamma=0.99, buffer_size=100000, batch_size=64,
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=50000):
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size

        self.policy_net = DQN(state_shape[0], num_actions).to(device)
        self.target_net = DQN(state_shape[0], num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
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
        state = torch.FloatTensor(state).unsqueeze(0).to(device)  # (1, C, H, W)
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

        next_actions = self.policy_net(next_state).max(1)[1].unsqueeze(1)
        next_q_values = self.target_net(
            next_state).gather(1, next_actions).detach()

        expected_q_values = reward + self.gamma * next_q_values * (1 - done)

        loss = F.mse_loss(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())



def create_action_space():
    steering_options = [-1.0, 0.0, 1.0]  # 左转、直行、右转
    gas_options = [0.0, 1.0]            # 无加速、加速
    brake_options = [0.0, 0.8]          # 无刹车、刹车

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
    img /= 255.0  # 归一化到[0,1]
    img = np.expand_dims(img, axis=0)
    return img

def train_dqn(env, agent, action_space, num_episodes=500, target_update=1000, max_steps=1000, save_interval=10):
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
                "Epsilon": f"{agent.epsilon:.2f}"
            })
            pbar.update(1)
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(
                    f"Episode {episode}, Average Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
    return episode_rewards


def test_dqn(env, agent, action_space, num_episodes=10, max_steps=1000, render=True):
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


import imageio
def test_dqn_with_gif(env, agent, action_space, num_episodes=1, max_steps=1000, gif_path="car_racing.gif"):
    agent.policy_net.eval()
    frames = []  # 用于保存帧
    for episode in range(1, num_episodes + 1):
        obs, info = env.reset(seed=SEED + episode)
        state = preprocess(obs)
        total_reward = 0
        done = False
        step = 0
        while not done and step < max_steps:
            action_idx = agent.select_action(
                state, is_exploration=False)  # 禁用探索
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


if __name__ == "__main__":
    env = gym.make('CarRacing-v2', render_mode="rgb_array")
    action_space = create_action_space()
    num_actions = len(action_space)
    print(f"Number of discrete actions: {num_actions}")
    state_shape = (1, 84, 84)  # 灰度图，84x84
    agent = DQNAgent(state_shape, num_actions)
    weight_files = sorted(glob.glob("dqn_weights_episode_*.pth"))
    if weight_files:
        latest_weight = weight_files[-1]
        print(f"Loading weights from {latest_weight}...")
        agent.policy_net.load_state_dict(torch.load(latest_weight))
        start_episode = int(latest_weight.split('_')[-1].split('.')[0])
        print(f"Resuming training from episode {start_episode + 1}...")
    else:
        print("No saved weights found. Starting training from scratch...")
        start_episode = 1
        
        
    # 训练智能体
    # num_training_episodes = 5000
    # print("Starting training...")
    # rewards = train_dqn(env, agent, action_space,
    #                     num_episodes=num_training_episodes,
    #                     save_interval=25)

    # # 绘制训练奖励曲线
    # plt.figure(figsize=(12, 5))
    # plt.plot(rewards, label='Reward per Episode')
    # plt.xlabel('Episode')
    # plt.ylabel('Reward')
    # plt.title('Training Reward Curve')
    # plt.legend()
    # plt.show()

    print("Starting testing...")
    print("Available weights:")
    for idx, weight_file in enumerate(weight_files):
        print(f"{idx + 1}: {weight_file}")
    selected_idx = int(
        input("Enter the number of the weight file to load for testing: ")) - 1
    selected_weight = weight_files[selected_idx]
    print(f"Loading selected weights from {selected_weight}...")
    agent.policy_net.load_state_dict(torch.load(selected_weight))
    test_dqn(env, agent, action_space, num_episodes=3, render=True)
    print("Starting testing with GIF generation...")
    test_dqn_with_gif(env, agent, action_space, gif_path="car_racing.gif")
