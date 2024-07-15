import gymnasium
import numpy as np
import tensorflow as tf

import flappy_bird_gymnasium
from flappy_bird_gymnasium.envs.utils import MODEL_PATH


class DuelingDQN(tf.keras.Model):
    def __init__(self, action_space):
        super(DuelingDQN, self).__init__()

        self.fc1 = tf.keras.layers.Dense(
            512,
            activation="elu",
            kernel_initializer=tf.keras.initializers.Orthogonal(tf.sqrt(2.0)),
        )
        self.fc2 = tf.keras.layers.Dense(
            256,
            activation="elu",
            kernel_initializer=tf.keras.initializers.Orthogonal(tf.sqrt(2.0)),
        )
        self.V = tf.keras.layers.Dense(
            1,
            activation=None,
            kernel_initializer=tf.keras.initializers.Orthogonal(0.01),
        )
        self.A = tf.keras.layers.Dense(
            action_space,
            activation=None,
            kernel_initializer=tf.keras.initializers.Orthogonal(0.01),
        )

    def call(self, inputs, training=None):
        x = self.fc1(inputs, training=training)
        x = self.fc2(x, training=training)
        V = self.V(x, training=training)
        A = self.A(x, training=training)
        adv_mean = tf.reduce_mean(A, axis=-1, keepdims=True)
        return V + (A - adv_mean)

    def get_action(self, state):
        q_value = self(state)
        print("Q value: ", q_value, tf.math.argmax(q_value, axis=-1))
        return tf.math.argmax(q_value, axis=-1)[0]


def play(epoch=10, audio_on=True, render_mode="human", use_lidar=False):
    env = gymnasium.make(
        "FlappyBird-v0", audio_on=audio_on, render_mode=render_mode, use_lidar=use_lidar
    )

    # init models
    if use_lidar:
        q_model.build((None, *env.observation_space.shape))
        q_model.load_weights(MODEL_PATH + "/model_lidar.h5")
    else:
        q_model = DuelingDQN(env.action_space.n)
        q_model.build((None, *env.observation_space.shape))
        q_model.load_weights(MODEL_PATH + "/model.h5")

    # run
    for _ in range(epoch):
        state, _ = env.reset(seed=123)
        state = np.expand_dims(state, axis=0)
        while True:
            # Getting action
            action = q_model.get_action(state)
            action = np.array(action, copy=False, dtype=env.env.action_space.dtype)

            # Processing action
            next_state, _, done, _, info = env.step(action)

            state = np.expand_dims(next_state, axis=0)
            print(f"Obs: {state}\n" f"Action: {action}\n" f"Score: {info['score']}\n")

            if done:
                break

    env.close()
    assert state.shape == (1,) + env.observation_space.shape
    assert info["score"] > 0


def test_play():
    play(epoch=1, audio_on=False, render_mode=None, use_lidar=False)
    # play(epoch=1, audio_on=False, render_mode=None, use_lidar=True)


if __name__ == "__main__":
    play()
