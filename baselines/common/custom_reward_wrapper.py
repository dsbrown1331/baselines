import gym
import numpy as np
from baselines.common.vec_env import VecEnvWrapper

class VecLiveLongReward(VecEnvWrapper):
    def __init__(self, venv):
        VecEnvWrapper.__init__(self, venv)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()

        rews = np.ones_like(rews)

        print(obs.shape)
        # obs shape: [num_env,84,84,4] in case of atari games

        return obs, rews, news, infos

    def reset(self, **kwargs):
        obs = self.venv.reset()

        ##############
        # If the reward is based on LSTM or something, then please reset internal state here.
        ##############

        return obs


import tensorflow as tf
class VecTFRandomReward(VecEnvWrapper):
    def __init__(self, venv):
        VecEnvWrapper.__init__(self, venv)

        self.graph = tf.Graph()

        config = tf.ConfigProto(
            device_count = {'GPU': 0}) # Run on CPU
        #config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph,config=config)

        with self.graph.as_default():
            with self.sess.as_default():
                self.obs = tf.placeholder(tf.float32,[None,84,84,4])

                self.rewards = tf.reduce_mean(
                    tf.random_normal(tf.shape(self.obs)),axis=[1,2,3])


    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()

        rews = self.sess.run(self.rewards,feed_dict={self.obs:obs})

        return obs, rews, news, infos

    def reset(self, **kwargs):
        obs = self.venv.reset()

        ##############
        # If the reward is based on LSTM or something, then please reset internal state here.
        ##############

        return obs
