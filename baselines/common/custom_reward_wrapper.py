import gym
import numpy as np
from baselines.common.vec_env import VecEnvWrapper

class VecLiveLongReward(VecEnvWrapper):
    def __init__(self, venv):
        VecEnvWrapper.__init__(self, venv)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()

        rews = np.ones_like(rews)

        #print(obs.shape)
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

class VecTFPreferenceReward(VecEnvWrapper):
    def __init__(self, venv, num_models, model_dir):
        VecEnvWrapper.__init__(self, venv)

        self.graph = tf.Graph()

        config = tf.ConfigProto(
            device_count = {'GPU': 0}) # Run on CPU
        #config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph,config=config)

        with self.graph.as_default():
            with self.sess.as_default():
                import os, sys
                dir_path = os.path.dirname(os.path.realpath(__file__))
                sys.path.append(os.path.join(dir_path,'..','..','..','..'))
                from preference_learning import Model

                print(os.path.realpath(model_dir))

                self.models = []
                for i in range(num_models):
                    with tf.variable_scope('model_%d'%i):
                        model = Model(self.venv.observation_space.shape[0])
                        model.saver.restore(self.sess,model_dir+'/model_%d.ckpt'%(i))
                    self.models.append(model)

        """
        try:
            self.save = venv.save
            self.load = venv.load
        except AttributeError:
            pass
        """

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()

        with self.graph.as_default():
            with self.sess.as_default():
                r_hat = np.zeros_like(rews)
                for model in self.models:
                    r_hat += model.get_reward(obs)

        rews = r_hat / len(self.models)

        return obs, rews, news, infos

    def reset(self, **kwargs):
        obs = self.venv.reset()

        return obs

if __name__ == "__main__":
    pass
