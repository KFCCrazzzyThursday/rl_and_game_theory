import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# SARSA 算法


def sarsa(env, num_episodes, alpha, gamma, epsilon):
    """
    Q(S, A) ← Q(S, A) + α · [ R + γ · Q(S', A') − Q(S, A) ]
    return:
    学习到的 Q 表
    episode_rewards: 每回合总奖励
    """
    # 初始化 Q 表，每个sa对的初始值为 0
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    episode_rewards = []

    for episode in range(num_episodes):
        # 初始化状态 S
        state, info = env.reset()

        # 根据 ε-贪婪策略选择动作 A
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # 随机选择
        else:
            action = np.argmax(Q[state])  # 选择 Q 值最大的

        total_reward = 0

        while True:
            # 执行动作 A，得到下一状态 S'、即时奖励 R 和是否终止标志
            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward  # 累计奖励

            # 根据 ε-贪婪策略选择下一动作 A'
            if np.random.rand() < epsilon:
                next_action = env.action_space.sample()  # 随机选择动作
            else:
                next_action = np.argmax(Q[next_state])  # 选择 Q 值最大的动作

            # 更新 Q 值
            # 对应公式：Q(S, A) <- Q(S, A) + α · [ R + γ · Q(S', A') − Q(S, A) ]
            Q[state, action] += alpha * (
                reward + gamma * Q[next_state, next_action] - Q[state, action]
            )

            # 状态更新：S <- S', A <- A'
            state = next_state
            action = next_action
            # 如果到达终止状态或被截断，则退出
            if terminated or truncated:
                break

        # 记录本回合的总奖励
        episode_rewards.append(total_reward)

    return Q, episode_rewards


# Q-learning 算法
def q_learning(env, num_episodes, alpha, gamma, epsilon):
    """
    Q(S, A) ← Q(S, A) + α · [ R + γ · max_a Q(S', a) − Q(S, A) ]
    
    return:
    学习到的 Q 表
    episode_rewards: 每回合的总奖励
    """
    # 初始化 Q 表，每个状态-动作对的初始值为 0
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    episode_rewards = []

    for episode in range(num_episodes):
        # 初始化状态 S
        state, info = env.reset()
        total_reward = 0

        while True:
            # 根据 ε-贪婪策略选择动作
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # 随机选择
            else:
                action = np.argmax(Q[state])  # 选择 Q 值最大的

            # 执行动作 A，得到下一状态 S'、即时奖励 R 和是否终止标志
            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward  # 累计奖励

            # 找到 S' 下的最优动作对应的 Q 值 max_a Q(S', a)
            best_next_action = np.argmax(Q[next_state])

            # 更新 Q 值
            # 对应公式：Q(S, A) <- Q(S, A) + α · [ R + γ · max_a Q(S', a) − Q(S, A) ]
            Q[state, action] += alpha * (
                reward + gamma * Q[next_state,
                                   best_next_action] - Q[state, action]
            )

            # 状态更新：S <- S'
            state = next_state

            # 如果到达终止状态或被截断 退出
            if terminated or truncated:
                break

        # 记录本回合总奖励
        episode_rewards.append(total_reward)

    return Q, episode_rewards

def extract_policy(Q):
    policy = np.argmax(Q, axis=1)
    return policy


def run_episode(env, policy):
    state, info = env.reset()
    states = [state]
    total_reward = 0
    while True:
        action = policy[state]
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        states.append(next_state)
        state = next_state
        if terminated or truncated:
            break
    return states, total_reward


def plot_route(states, title):
    grid_shape = (4, 12)
    grid = np.zeros(grid_shape)

    cmap = mcolors.ListedColormap(
        ['white', 'black', 'red', 'green', 'blue', 'yellow'])
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # 悬崖
    cliff = np.arange(37, 47)
    grid[np.unravel_index(cliff, grid_shape)] = -1  # 悬崖用 -1 表示

    start_state = 36
    goal_state = 47
    grid[np.unravel_index(start_state, grid_shape)] = 2  # 起点2
    grid[np.unravel_index(goal_state, grid_shape)] = 3  # 终点3

    for idx, state in enumerate(states):
        if state != start_state and state != goal_state and grid[np.unravel_index(state, grid_shape)] == 0:
            grid[np.unravel_index(state, grid_shape)] = 4  # 路径4
    plt.figure(figsize=(12, 4))
    plt.imshow(grid, cmap=cmap, norm=norm)
    plt.title(title)
    plt.axis('off')

    for i in range(-1, grid_shape[1]):
        plt.plot([i + 0.5, i + 0.5], [-0.5, grid_shape[0] - 0.5],
                 color='gray', linewidth=1)
    for i in range(-1, grid_shape[0]):
        plt.plot([-0.5, grid_shape[1] - 0.5],
                 [i + 0.5, i + 0.5], color='gray', linewidth=1)

    start_coords = np.unravel_index(start_state, grid_shape)
    goal_coords = np.unravel_index(goal_state, grid_shape)
    plt.text(start_coords[1], start_coords[0], 'S', ha='center',
             va='center', color='white', fontsize=14, fontweight='bold')
    plt.text(goal_coords[1], goal_coords[0], 'G', ha='center',
             va='center', color='white', fontsize=14, fontweight='bold')


    for i in range(len(states) - 1):
        curr_state = states[i]
        next_state = states[i + 1]
        curr_coords = np.unravel_index(curr_state, grid_shape)
        next_coords = np.unravel_index(next_state, grid_shape)
        plt.arrow(curr_coords[1], curr_coords[0], next_coords[1] - curr_coords[1], next_coords[0] - curr_coords[0],
                  head_width=0.2, length_includes_head=True, color='blue')

    plt.show()


env = gym.make('CliffWalking-v0')

# 参数
num_episodes = 5000  
alpha = 0.2
gamma = 0.99
epsilon = 0.1

# SARSA
Q_sarsa, rewards_sarsa = sarsa(env, num_episodes, alpha, gamma, epsilon)
policy_sarsa = extract_policy(Q_sarsa)

# Q-learning
Q_qlearning, rewards_qlearning = q_learning(
    env, num_episodes, alpha, gamma, epsilon)
policy_qlearning = extract_policy(Q_qlearning)

states_sarsa, _ = run_episode(env, policy_sarsa)
states_qlearning, _ = run_episode(env, policy_qlearning)
plot_route(states_sarsa, 'SARSA PATH')
plot_route(states_qlearning, 'Q-learning PATH')

plt.figure(figsize=(10, 6))
plt.plot(rewards_sarsa, label='SARSA')
plt.plot(rewards_qlearning, label='Q-learning')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Reward per Episode')
plt.legend()
plt.grid(True)
plt.show()



def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


# 平滑奖励曲线
window_size = 50
smoothed_rewards_sarsa = moving_average(rewards_sarsa, window_size)
smoothed_rewards_qlearning = moving_average(rewards_qlearning, window_size)
plt.figure(figsize=(10, 6))
plt.plot(smoothed_rewards_sarsa, label='SARSA (Smoothed)')
plt.plot(smoothed_rewards_qlearning, label='Q-learning (Smoothed)')
plt.xlabel('Episode')
plt.ylabel('Average Total Reward')
plt.title('Reward per Episode (Smoothed)')
plt.legend()
plt.grid(True)
plt.show()
