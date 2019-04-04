import gym
import numpy as np
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.running_mean_std import RunningMeanStd

import torch
import torch.nn as nn
import torch.nn.functional as F

class AtariNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 16, 7, stride=3)
        self.conv2 = nn.Conv2d(16, 16, 5, stride=2)
        self.conv3 = nn.Conv2d(16, 16, 3, stride=1)
        self.conv4 = nn.Conv2d(16, 16, 3, stride=1)
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 1)


    def cum_return(self, traj):
        '''calculate cumulative return of trajectory'''
        sum_rewards = 0
        for x in traj:
            x = x.permute(0,3,1,2) #get into NCHW format
            #compute forward pass of reward network
            x = F.leaky_relu(self.conv1(x))
            x = F.leaky_relu(self.conv2(x))
            x = F.leaky_relu(self.conv3(x))
            x = F.leaky_relu(self.conv4(x))
            x = x.view(-1, 784)
            x = F.leaky_relu(self.fc1(x))
            #r = torch.sigmoid(self.fc2(x)) #clip reward?
            r = self.fc2(x) #clip reward?
            sum_rewards += r
        ##    y = self.scalar(torch.ones(1))
        ##    sum_rewards += y
        #print(sum_rewards)
        return sum_rewards



    def forward(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''
        #print([self.cum_return(traj_i), self.cum_return(traj_j)])
        return torch.cat([self.cum_return(traj_i), self.cum_return(traj_j)])

class VecRLplusIRLAtariReward(VecEnvWrapper):
    def __init__(self, venv, reward_net_path, combo_param):
        VecEnvWrapper.__init__(self, venv)
        self.reward_net = AtariNet()
        self.reward_net.load_state_dict(torch.load(reward_net_path))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.reward_net.to(self.device)

        self.lamda = combo_param #how much weight to give to IRL verus RL combo_param \in [0,1] with 0 being RL and 1 being IRL
        self.rew_rms = RunningMeanStd(shape=())
        self.epsilon = 1e-8
        self.cliprew = 10.

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
##Testing network to see why always giving zero rewards....
        #import pickle
        #filename = 'rand_obs.pkl'
        #infile = open(filename,'rb')
        #rand_obs = pickle.load(infile)
        #infile.close()
        #traj = [obs / 255.0] #normalize!
        #import matplotlib.pyplot as plt
        #plt.figure(1)
        #plt.imshow(obs[0,:,:,0])
        #plt.figure(2)
        #plt.imshow(rand_obs[0,:,:,0])
        #plt.show()
        #print(obs.shape)
        with torch.no_grad():
            rews_network = self.reward_net.cum_return(torch.from_numpy(np.array(obs)).float().to(self.device)).cpu().numpy().transpose()[0]
            #rews2= self.reward_net.cum_return(torch.from_numpy(np.array([rand_obs])).float().to(self.device)).cpu().numpy().transpose()[0]
        #self.rew_rms.update(rews_network)
        #r_hat = rews_network
        #r_hat = np.clip((r_hat - self.rew_rms.mean) / np.sqrt(self.rew_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        #print(rews1)
        #   print(rews2)

        #print(obs.shape)
        # obs shape: [num_env,84,84,4] in case of atari games

        #combine IRL and RL rewards using lambda parameter like Yuke Zhu's paper "Reinforcement and Imitation Learningfor Diverse Visuomotor Skills"
        reward_combo = self.lamda * rews_network + (1-self.lamda) * rews

        return obs, reward_combo , news, infos

    def reset(self, **kwargs):
        obs = self.venv.reset()

        ##############
        # If the reward is based on LSTM or something, then please reset internal state here.
        ##############

        return obs


class VecPyTorchAtariReward(VecEnvWrapper):
    def __init__(self, venv, reward_net_path):
        VecEnvWrapper.__init__(self, venv)
        self.reward_net = AtariNet()
        self.reward_net.load_state_dict(torch.load(reward_net_path))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.reward_net.to(self.device)

        self.rew_rms = RunningMeanStd(shape=())
        self.epsilon = 1e-8
        self.cliprew = 10.

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        traj = [obs] #normalize!
        
        with torch.no_grad():
            rews_network = self.reward_net.cum_return(torch.from_numpy(np.array(traj)).float().to(self.device)).cpu().numpy().transpose()[0]

        # obs shape: [num_env,84,84,4] in case of atari games

        return obs, rews_network, news, infos

    def reset(self, **kwargs):
        obs = self.venv.reset()

        ##############
        # If the reward is based on LSTM or something, then please reset internal state here.
        ##############

        return obs



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
