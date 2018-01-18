#!/usr/bin/env python
from OpenGL import GLU
import gym, roboschool
import argparse
import os
from baselines import bench, logger

def test(env_id, num_timesteps, seed, model_path):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.ppo2 import ppo2
    from baselines.ppo2.policies import MlpPolicy, CnnPolicy
    import gym
    from gym import wrappers
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True
    tf.Session(config=config).__enter__()
    def make_env():
        import numpy as np
        import colorsys
        COLOR_SET = [ tuple(int(c*255) for c in colorsys.hsv_to_rgb(h/360.,1,1))
                    for h in range(0,360,20) ]

        np.random.seed(0)
        np.random.shuffle(COLOR_SET)
        COLOR_SET = COLOR_SET[:4]

        env = gym.make(env_id)
        env.unwrapped.set_goals( [0] )
        env.unwrapped.set_targets_color( COLOR_SET )
        #env.unwrapped.set_fixed(100)
        #env = wrappers.Monitor(env,os.path.join('/tmp/videos/'),force=True)
        return env
    env = DummyVecEnv([make_env],render=False)
    #env = DummyVecEnv([make_env],render=True)
    env = VecNormalize(env,True,True)
    #env = VecNormalize(env,False,False)
    env.load(model_path)

    #set_global_seeds(seed)
    policy = MlpPolicy
    #policy = CnnPolicy
    states,actions,images = ppo2.test(policy=policy, env=env, nsteps=num_timesteps, model_dir=model_path)

    print(states.shape, actions.shape, images.shape)
    import numpy as np
    np.savez('test.npz', states=states,actions=actions,images=images)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    #parser.add_argument('--env', help='environment ID', default='Hopper-v1')
    parser.add_argument('--env', help='environment ID', default='RoboschoolReacher-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(2048))
    args = parser.parse_args()

    test(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
         model_path='/tmp/openai-2018-01-17-21-14-33-709102/checkpoints/00340')
         #model_path='/tmp/openai-2018-01-16-21-13-42-297830/checkpoints/00100')
         #model_path='/tmp/openai-2018-01-16-20-46-52-719658/checkpoints/00220')
         #model_path='/tmp/openai-2018-01-16-20-27-25-798771/checkpoints/00150')
         #model_path='/tmp/openai-2018-01-16-20-19-14-061814/checkpoints/00110')
         #model_path='/tmp/openai-2018-01-16-20-08-11-760768/checkpoints/00230')
         #model_path='/tmp/openai-2018-01-16-20-08-11-760768/checkpoints/00230')
         #model_path='/tmp/openai-2018-01-16-17-09-00-358335/checkpoints/00140')
         #model_path='/tmp/openai-2018-01-16-16-41-53-640746/checkpoints/00160')
         #model_path='/tmp/openai-2018-01-15-16-04-04-089676/checkpoints/00230')
         #model_path='/tmp/openai-2018-01-15-15-49-32-706043/checkpoints/00170')
         #model_path='/tmp/openai-2017-12-18-18-05-34-607331/checkpoints/00020')
         #model_path='/tmp/openai-2017-12-18-17-38-55-610551/checkpoints/00060')

if __name__ == '__main__':
    main()

