from stable_baselines3.common.callbacks import EvalCallback

from stable_baselines3 import DQN, PPO
import flappy_bird_gym
import time

eval_env = flappy_bird_gym.make("FlappyBird-v0")

eval_callback = EvalCallback(
    eval_env, deterministic=True, render=False, best_model_save_path="./another/"
)
env = flappy_bird_gym.make("FlappyBird-v0")
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000000, callback=eval_callback, log_interval=1000)

model = DQN.load("./gym/best_model.zip")
env = flappy_bird_gym.make("FlappyBird-v0")

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    time.sleep(1 / 30)
    if done:
        obs = env.reset()
