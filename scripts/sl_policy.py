from __future__ import division
import tensorflow as tf
import numpy as np
from itertools import count
import sys

from networks import policy_nn
from utils import *
from env import Env, getIdAndEntity
from BFS.KB import KB
from BFS.BFS import BFS
import time
import ipdb
import networkx as nx
from crime_graph import CrimeGraph

# import osmgraph

relation = sys.argv[1]
file_name = "../" + sys.argv[2]
first_retrain = int(sys.argv[3])
# episodes = int(sys.argv[2])
graphpath = dataPath + int(sys.argv[4])
edges = int(sys.argv[5])
relationPath = dataPath + 'train_pos'


class SupervisedPolicy(object):
    """docstring for SupervisedPolicy"""

    def __init__(self, learning_rate=0.001):
        self.initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope('supervised_policy'):
            self.state = tf.placeholder(tf.float32, [None, state_dim], name='state')
            self.history = tf.placeholder(tf.float32, [None, history_dim], name='state')
            self.action = tf.placeholder(tf.int32, [None], name='action')
            self.action_prob = policy_nn(self.state, self.history,  state_dim, history_dim, action_space, self.initializer)

            action_mask = tf.cast(tf.one_hot(self.action, depth=action_space), tf.bool)
            self.picked_action_prob = tf.boolean_mask(self.action_prob, action_mask)

            self.loss = tf.reduce_sum(-tf.log(self.picked_action_prob) * 1) + sum(
                tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='supervised_policy'))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action_prob, {self.state: state})

    def update(self, state, history, action, sess=None):
        sess = sess or tf.get_default_session()
        _, loss = sess.run([self.train_op, self.loss], {self.state: state, self.history: history, self.action: action})
        return loss


def train():
    f = open(dataPath + edges)
    kb_all = f.readlines()
    f.close()

    kb = []
    for line in kb_all:
        kb.append(line)
    # networkxGraph = getGraph()
    tf.reset_default_graph()
    policy_nn = SupervisedPolicy()

    f = open(file_name)
    train_data = f.readlines()
    train_data = list(np.random.permutation(train_data).tolist())
    f.close()

    num_samples = len(train_data)

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        if not first_retrain:

            saver.restore(sess, 'models/policy_supervised_' + relation)
            print "sl_policy restored"
        else:
            sess.run(tf.global_variables_initializer())
        if num_samples > 50000:
            num_samples = 50000

        entity2id_, id2entity_ = getIdAndEntity()
        crime_graph = CrimeGraph(id2entity_, dataPath)

        rewards_gained = 0.0
        min_dist_gained = 0.0
        avg_length = 0.0
        #ipdb.set_trace()
        for episode in xrange(num_samples):
            print "Episode %d" % episode
            print 'Training Sample:', train_data[episode % num_samples][:-1]

            env = Env(dataPath, crime_graph)
            sample = train_data[episode % num_samples].split()

            good_episodes, avg_reward, min_dist, avg_len = teacher(sample[0], sample[1], env, crime_graph, graphpath)
            print "reward: ", avg_reward

            rewards_gained += avg_reward
            min_dist_gained += min_dist
            avg_length += avg_len
            print "CURRENT PATH MIN DIST: {}".format(min_dist)
            print 'Average reward so far: ', rewards_gained / (episode + 1)
            print 'Min dist so far: ', min_dist_gained / (episode + 1)
            print 'Length so far: ', avg_length / (episode + 1)
            for item in good_episodes:
                state_batch = []
                action_batch = []
                history_batch = []
                for t, transition in enumerate(item):
                    state_batch.append(transition.state)
                    action_batch.append(transition.action)
                    history_batch.append(transition.history)
                state_batch = np.squeeze(state_batch)
                state_batch = np.reshape(state_batch, [-1, state_dim])
                history_batch = np.squeeze(history_batch)
                history_batch = np.reshape(history_batch, [-1, history_dim])
                policy_nn.update(state_batch, history_batch, action_batch)

        saver.save(sess, 'models/policy_supervised_' + relation)
        print 'Model saved'


def test(test_episodes):
    tf.reset_default_graph()
    policy_nn = SupervisedPolicy()

    f = open(relationPath)
    test_data = f.readlines()
    f.close()

    test_data = test_data[-test_episodes:]
    print len(test_data)

    success = 0

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver.restore(sess, 'models/policy_supervised_' + relation)
        print 'Model reloaded'
        for episode in xrange(len(test_data)):
            print 'Test sample %d: %s' % (episode, test_data[episode][:-1])
            env = Env(dataPath, test_data[episode])
            sample = test_data[episode].split()
            state_idx = [env.entity2id_[sample[0]], env.entity2id_[sample[1]], 0]
            for t in count():
                state_vec = env.idx_state(state_idx)
                action_probs = policy_nn.predict(state_vec)
                action_chosen = np.random.choice(np.arange(action_space), p=np.squeeze(action_probs))
                reward, new_state, done = env.interact(state_idx, action_chosen)
                if done or t == max_steps_test:
                    if done:
                        print 'Success'
                        success += 1
                    print 'Episode ends\n'
                    break
                state_idx = new_state

    print 'Success persentage:', success / test_episodes


if __name__ == "__main__":
    train()
# test(50)
