#!/usr/bin/env python3
import sys
from baselines import logger
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.cmd_util import make_atari_env, atari_arg_parser
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy
import multiprocessing
import tensorflow as tf


def train(env_id, num_timesteps, seed, policy):

    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    tf.Session(config=config).__enter__()

    env = VecFrameStack(make_atari_env(env_id, 8, seed), 4)
    policy = {'cnn' : CnnPolicy, 'lstm' : LstmPolicy, 'lnlstm' : LnLstmPolicy}[policy]
    ppo2.learn(policy=policy, env=env, nsteps=128, nminibatches=4,
        lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
        ent_coef=.01,
        lr=lambda f : f * 2.5e-4,
        cliprange=lambda f : f * 0.1,
        total_timesteps=int(num_timesteps * 1.1),
        save_interval=1000)

def test(env_id, seed, policy, model_path):
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    from gym import wrappers
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    with sess.as_default():
        def make_env(): # pylint: disable=C0111
            env = make_atari(env_id)
            env.seed(seed)
            env = wrappers.Monitor(env,logger.get_dir(),force=True)
            return wrap_deepmind(env, clip_rewards=False, frame_stack=False)
        #env = make_env()
        env = DummyVecEnv([make_env])
        env = VecFrameStack(env, 4)

        policy = {'cnn' : CnnPolicy, 'lstm' : LstmPolicy, 'lnlstm' : LnLstmPolicy}[policy]

        ppo2.test(policy=policy, env=env, model_dir=model_path)
    sess.close()

def main():
    from pathlib import Path
    current_path = Path(__file__).parent

    parser = atari_arg_parser()
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    parser.add_argument('--model', help='Model for run a pretrained model', default=current_path/'models'/'pong'/'10000')
    args = parser.parse_args()
    logger.configure()

    if args.model == '' :
        train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
            policy=args.policy)
    else:
        test(args.env,
            seed=args.seed,
            policy=args.policy,
            model_path=args.model)

if __name__ == '__main__':
    main()
