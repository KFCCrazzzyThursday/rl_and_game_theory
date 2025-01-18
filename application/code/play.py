import pygame
import gym


def play_carracing_manually(env):
    pygame.init()
    window_width, window_height = 800, 600  # 窗口大小
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Car Racing Manual Control")
    clock = pygame.time.Clock()

    action = [0.0, 0.0, 0.0]  # [steering, gas, brake]

    print("Manual control started. Use the following keys:")
    print("W: Accelerate | S: Brake | A: Steer Left | D: Steer Right | Q: Quit")

    done = False
    obs, info = env.reset(seed=42)
    total_reward = 0

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break

        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]:  # Quit
            done = True
            break
        # Steering
        action[0] = - \
            1.0 if keys[pygame.K_a] else (1.0 if keys[pygame.K_d] else 0.0)
        action[1] = 1.0 if keys[pygame.K_w] else 0.0  # Gas
        action[2] = 0.8 if keys[pygame.K_s] else 0.0  # Brake

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Render the environment
        frame = env.render()  # 获取环境的渲染帧
        frame_surface = pygame.surfarray.make_surface(
            frame.swapaxes(0, 1))  # 转换为 Pygame 表面
        frame_surface = pygame.transform.scale(
            frame_surface, (window_width, window_height))  # 调整大小
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()
        clock.tick(30)

    print(f"Game over! Total reward: {total_reward:.2f}")
    env.close()
    pygame.quit()


if __name__ == "__main__":
    # 初始化环境
    env = gym.make("CarRacing-v2", render_mode="rgb_array")
    play_carracing_manually(env)