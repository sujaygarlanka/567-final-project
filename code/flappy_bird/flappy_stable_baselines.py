import flappy_bird_gymnasium
import gymnasium as gym
from stable_baselines3.common.callbacks import EvalCallback
# from wandb.integration.sb3 import WandbCallback
# import wandb

# model.learn(..., callback=WandbCallback())

from stable_baselines3 import DQN, PPO


eval_env = gym.make("FlappyBird-v0", render_mode="rgb_array")
# eval_env = gym.make("CartPole-v1", render_mode="rgb_array")


# config = {
#     "policy_type": "MlpPolicy",
#     "total_timesteps": 100000,
#     "env_id": "CartPole-v1",
# }

# run = wandb.init(
#     project="cartpole-v1-stable-baselines3",
#     config=config,
#     sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
#     # monitor_gym=True,  # auto-upload the videos of agents playing the game
#     # save_code=True,  # optional
# )

eval_callback = EvalCallback(
    eval_env,
    deterministic=True,
    render=False,
)
# env = gym.make("CartPole-v1", render_mode="rgb_array")
env = gym.make("FlappyBird-v0", render_mode="rgb_array")
model = DQN("MlpPolicy", env, verbose=1)
# model.learn(
#     total_timesteps=100000, callback=WandbCallback(verbose=1, log="all"), log_interval=4
# )
model.learn(
    total_timesteps=1000000, callback=eval_callback, log_interval=4
)
# model.save("dqn_cartpole")

# del model # remove to demonstrate saving and loading

# model = DQN.load("dqn_cartpole")

# obs, info = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = env.step(action)
#     if terminated or truncated:
#         obs, info = env.reset()
