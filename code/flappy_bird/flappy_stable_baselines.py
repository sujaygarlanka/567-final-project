import flappy_bird_gymnasium
import gymnasium as gym
from stable_baselines3.common.callbacks import EvalCallback

from stable_baselines3 import DQN, PPO

# env = gym.make("FlappyBird-v0", render_mode="rgb_array")
eval_env = gym.make("CartPole-v1", render_mode="rgb_array")
eval_callback = EvalCallback(eval_env, n_eval_episodes=4, 
                             best_model_save_path="./logs/best_model",
                             log_path="./logs/results",
                             deterministic=True, render=False)


# model = DQN("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=1000000, log_interval=4)
# model.save("dqn_cartpole")

env = gym.make("CartPole-v1", render_mode="rgb_array")
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000, callback=eval_callback, log_interval=4)
model.save("dqn_cartpole")

# del model # remove to demonstrate saving and loading

# model = DQN.load("dqn_cartpole")

# obs, info = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = env.step(action)
#     if terminated or truncated:
#         obs, info = env.reset()