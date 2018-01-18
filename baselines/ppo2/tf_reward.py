import sys
sys.path.append('/home/wonjoon/workspace/tcn-ppo')

import numpy as np
import tensorflow as tf
from functools import partial
from model import Aligner, _reacher_arch

class ShuffleAndLearn(object):
    def __init__(self,sess):
        self.sess = sess
        self.x = tf.placeholder(tf.float32,[None,64,64,3])
        self.y = tf.placeholder(tf.float32,[None,64,64,3])
        self.gt = tf.placeholder(tf.bool,[None])

        with tf.variable_scope('train'):
            with tf.variable_scope('params') as params:
                pass
        with tf.variable_scope('valid'):
            self.net = \
                Aligner(self.x,self.y,self.gt,
                        partial(_reacher_arch,32), # embedding vector length
                        None,
                        None, # l2_lambda
                        None,
                        params,is_training=False)

    def load(self, model_file):
        self.net.load(self.sess,model_file)

    def reward_fn(self, p_img, c_img):
        """
        p_img: past image. image one frame before
        c_img: current image. image of current frame
        """
        logits = \
            self.sess.run(self.net.logits,
                          feed_dict={self.x: p_img[None].astype(np.float32)/255.0, self.y: c_img[None].astype(np.float32)/255.0,
                                     self.gt:np.array([1])})
        return logits[0]


if __name__ == "__main__":
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    tf.Session(config=config).__enter__()

    sess = tf.get_default_session()
    sal = ShuffleAndLearn(sess)

    tf.global_variables_initializer().run(session=sess) #pylint: disable=E1101
    sal.load('/home/wonjoon/workspace/tcn-ppo/log/shffule-and-learn/2018-01-17 12:43:26/model.ckpt-8000')

    reward_fn = sal.reward_fn
    for _ in range(10):
        print(reward_fn(np.random.rand(64,64,3),np.random.rand(64,64,3)))

