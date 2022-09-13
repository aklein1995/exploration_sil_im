import numpy as np
import tensorflow as tf
# from tensorflow.keras import datasets, layers, models
# from baselines.a2c.utils import fc, conv, conv_to_fc
# from baselines.common.distributions import make_pdtype
from rapid.baselines_utils.model_and_distributions_utils import fc,make_pdtype,conv, conv_to_fc

class MlpPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False):
        self.pdtype = make_pdtype(ac_space)
        with tf.variable_scope("model", reuse=reuse):
            X = tf.placeholder(tf.float32, shape=(None,)+ob_space.shape)
            activ = tf.tanh
            processed_x = tf.layers.flatten(X)
            pi_h1 = activ(fc(processed_x, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
            pi_h2 = activ(fc(pi_h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
            vf_h1 = activ(fc(processed_x, 'vf_fc1', nh=64, init_scale=np.sqrt(2)))
            vf_h2 = activ(fc(vf_h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
            vf = fc(vf_h2, 'vf', 1)[:,0]

            self.pd, self.pi = self.pdtype.pdfromlatent(pi_h2, init_scale=0.01)


        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            tf.set_random_seed(0)
            a, v, neglogp, = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        def neg_log_prob(actions):
            return self.pd.neglogp(actions)

        self.X = X
        self.vf = vf
        self.step = step
        self.value = value
        self.neg_log_prob = neg_log_prob
        self.entropy = self.pd.entropy()

class MlpPolicy_unique(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False):
        self.pdtype = make_pdtype(ac_space)
        with tf.variable_scope("model", reuse=reuse):
            X = tf.placeholder(tf.float32, shape=(None,)+ob_space.shape)
            activ = tf.tanh
            processed_x = tf.layers.flatten(X)
            h1 = activ(fc(processed_x, 'fc1', nh=256, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'fc2', nh=64, init_scale=np.sqrt(2)))

            # value/critic head
            vf = fc(h2, 'vf', 1)[:,0]
            # policy/actor head
            self.pd, self.pi = self.pdtype.pdfromlatent(h2, init_scale=0.01)


        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            tf.set_random_seed(0)
            a, v, neglogp, = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        def neg_log_prob(actions):
            return self.pd.neglogp(actions)

        self.X = X
        self.vf = vf
        self.step = step
        self.value = value
        self.neg_log_prob = neg_log_prob
        self.entropy = self.pd.entropy()

class MlpPolicy_unique_cnn(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False):
        print('\nUsing shared-network for AC')
        self.pdtype = make_pdtype(ac_space)
        with tf.variable_scope("model", reuse=reuse):
            X = tf.placeholder(tf.float32, shape=(None,)+ob_space.shape)
            activ = tf.nn.relu
            h = activ(conv(X, 'c1', nf=32, rf=3, stride=2, pad='SAME', init_scale=np.sqrt(2)))
            h2 = activ(conv(h, 'c2', nf=32, rf=3, stride=2, pad='SAME',init_scale=np.sqrt(2)))
            h3 = activ(conv(h2, 'c3', nf=32, rf=3, stride=2, pad='SAME',init_scale=np.sqrt(2)))
            ctf = conv_to_fc(h3)
            common = activ(fc(ctf, 'fc1', nh=256, init_scale=np.sqrt(2)))

            # policy-head
            self.pd, self.pi = self.pdtype.pdfromlatent(common, init_scale=0.01)
            # value-head
            vf = fc(common, 'vf', 1)[:,0]

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            tf.set_random_seed(0)
            a, v, neglogp, = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.vf = vf
        self.step = step
        self.value = value

class CnnPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False):
        self.pdtype = make_pdtype(ac_space)
        with tf.variable_scope("model", reuse=reuse):
            X = tf.placeholder(tf.float32, shape=(None,)+ob_space.shape)
            activ = tf.tanh
            processed_x = tf.layers.flatten(X)
            pi_h1 = activ(fc(processed_x, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
            pi_h2 = activ(fc(pi_h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
            vf_h1 = activ(fc(processed_x, 'vf_fc1', nh=64, init_scale=np.sqrt(2)))
            vf_h2 = activ(fc(vf_h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
            vf = fc(vf_h2, 'vf', 1)[:,0]

            self.pd, self.pi = self.pdtype.pdfromlatent(pi_h2, init_scale=0.01)


        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            tf.set_random_seed(0)
            a, v, neglogp, = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.vf = vf
        self.step = step
        self.value = value
