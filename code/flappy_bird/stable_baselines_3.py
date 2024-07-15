import flappy_bird_gymnasium
import gymnasium as gym
from stable_baselines3.common.callbacks import EvalCallback

from stable_baselines3 import DQN, PPO

# eval_env = gym.make("FlappyBird-v0", render_mode="rgb_array")

# eval_callback = EvalCallback(
#     eval_env, deterministic=True, render=False, best_model_save_path="./gymnasium/"
# )
# env = gym.make("FlappyBird-v0", render_mode="rgb_array")
# model = DQN("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=1000000, callback=eval_callback, log_interval=1000)

model = DQN.load("./gymnasium/best_model.zip")
env = gym.make("FlappyBird-v0", render_mode="human")

obs, info = env.reset()
num = 0
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print(num)
        num = 0
        obs, info = env.reset()
    num += 1