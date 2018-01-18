#!/usr/bin/env python
from OpenGL import GLU
import gym, roboschool
import argparse
from baselines import bench, logger

def train(env_id, num_timesteps, seed):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.ppo2 import ppo2
    from baselines.ppo2.policies import MlpPolicy
    import gym
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True
    tf.Session(config=config).__enter__()

    # 4-1. Shuffle And Learn Reward
    from baselines.ppo2.tf_reward import ShuffleAndLearn
    from functools import partial
    sal = ShuffleAndLearn(tf.get_default_session())
    sal_load_fn = partial(sal.load,
                          '/home/wonjoon/workspace/tcn-ppo/log/shffule-and-learn/2018-01-17 12:43:26/model.ckpt-8000')

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

        # 1. Sparse Reward
        #env.unwrapped.set_sparse_reward()

        # 2. Pixel based Reward
        #env.unwrapped.set_fixed(100)
        #demo = np.load('demo.npz')['images']
        #env.unwrapped.set_demo(demo)

        # 3. Simple Reward (Distance Based Reward)
        #env.unwrapped.set_simple_reward()

        # 4-1. Shuffle And Learn Reward
        env.unwrapped.set_tf_reward_fn(sal.reward_fn)

        env = bench.Monitor(env, logger.get_dir())
        return env
    env = DummyVecEnv([make_env])
    env = VecNormalize(env)
    #env = VecNormalize(env,False,False) #normalize observ, normalize ret.

    set_global_seeds(seed)
    policy = MlpPolicy
    ppo2.learn(policy=policy, env=env, nsteps=2048, nminibatches=32,
        lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
        ent_coef=0.0,
        lr=3e-4,
        cliprange=0.2,
        total_timesteps=num_timesteps,
        save_interval=10,
        post_tf_init_fn=sal_load_fn)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='RoboschoolReacher-v1')
    #parser.add_argument('--env', help='environment ID', default='Hopper-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    args = parser.parse_args()
    logger.configure()

    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == '__main__':
    main()

