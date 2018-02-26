#!/usr/bin/env python3
import argparse
from baselines.common.cmd_util import mujoco_arg_parser
from baselines import bench, logger
from baselines.common import set_global_seeds
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import MlpPolicy
import gym
import tensorflow as tf
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

def train(env_id, num_timesteps, seed):
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()
    def make_env():
        env = gym.make(env_id)
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
        save_interval=10)

def test(env_id, seed, model_path):
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    with sess.as_default():
        def make_env(): # pylint: disable=C0111
            env = gym.make(env_id)
            #env.seed(seed)
            from gym import wrappers
            env = wrappers.Monitor(env,logger.get_dir(),force=True)
            return env
        #env = make_env()
        env = DummyVecEnv([make_env])
        env = VecNormalize(env)

        set_global_seeds(seed)
        policy = MlpPolicy
        ppo2.test(policy=policy, env=env, model_dir=model_path)
    sess.close()


def main():
    from pathlib import Path
    current_path = Path(__file__).parent

    parser= mujoco_arg_parser()
    parser.add_argument('--model', help='Model for run a pretrained model', default='')
    args = parser.parse_args()
    logger.configure()

    if args.model == '' :
        train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)
    else:
        test(args.env,
             seed=args.seed,
             model_path=args.model)

if __name__ == '__main__':
    main()
