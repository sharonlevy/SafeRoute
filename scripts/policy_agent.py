from __future__ import division
import tensorflow as tf
import numpy as np
import collections
from itertools import count
from sklearn.metrics.pairwise import cosine_similarity
import time
import sys
import base64
import ipdb
import IPython
import select
import bisect
import math
from geopy.distance import great_circle

from crime_graph import CrimeGraph

from networks import policy_nn, value_nn
from utils import *
from env import Env, getIdAndEntity

print sys.argv
relation = sys.argv[1]
task = sys.argv[2]
file_name = "../" + sys.argv[3]
first_retrain = int(sys.argv[4])
valid_file = "../" + sys.argv[5]
graphpath = dataPath + sys.argv[6]
man_test = dataPath + 'man_test.txt'
out_test = dataPath + 'out_test.txt'

np.seterr(all='raise')

START_TIME = time.time()
CMD_LINE = '\n'.join(sys.argv)
RUN_IDENTIFIER = base64.b64encode(CMD_LINE + '\n' + str(START_TIME))

# source https://stackoverflow.com/questions/292095/polling-the-keyboard-detect-a-keypress-in-python/41083602
def heardEnter():
    i, o, e = select.select([sys.stdin], [], [], 0.0001)
    for s in i:
        if s == sys.stdin:
            _ = sys.stdin.readline()
            return True
    return False


debug = False


def check_debug():
    global debug

    if debug:
        ipdb.set_trace()
        # IPython.embed()
        debug = False

    if heardEnter():
        debug = True

def node_dist(g, a, b):
    return great_circle(g.node[a]['coordinate'], g.node[b]['coordinate']).miles

def random_path(g, src_id, num_steps=10):
    path = [src_id]
    cur_node = src_id
    for i in range(num_steps):
        options = g.adj[cur_node].keys()
        selected = np.random.choice(options)
        sel_dir = g.adj[cur_node][selected]['direction']
        path.extend((sel_dir, selected))

    return path


def playout_moving_closer_weights(g, src_id, dst_id, num_playouts=100, num_steps=10):
    '''
    :param g:
    :type g: nx.DiGraph
    :param src_id:
    :type src_id: int
    :param dst_id:
    :type dst_id: int
    :param num_playouts: how many paths to randomly sample
    :param num_steps: how long each randomly sampled path should be
    :return:
    '''

    random_paths = [random_path(g, src_id, num_steps=num_steps) for _ in range(num_playouts)]
    closer = collections.defaultdict(int)
    total = collections.defaultdict(int)
    for p in random_paths:
        initial_dir = p[1]
        final_node = p[-1]
        initial_dist = node_dist(g, src_id, dst_id)
        final_dist = node_dist(g, final_node, dst_id)

        total[initial_dir] += 1
        if initial_dist >= final_dist:
            closer[initial_dir] += 1

    weights = {}
    for outgoing, data in g.adj[src_id].iteritems():
        d = data['direction']
        weights[d] = closer[d] / (total[d] or 1)

    return weights


class PolicyNetwork(object):

    def __init__(self, scope='policy_network', learning_rate=0.0005):
        self.initializer = tf.contrib.layers.xavier_initializer()
        self.variable_scope = scope

        self.collected_stats = {
            'avg_crime_local': tf.float32,
            'avg_crime_min_local': tf.float32,
            'avg_crime_global': tf.float32,
            'length': tf.float32,

            'avg_crime_local_running': tf.float32,
            'avg_crime_min_local_running': tf.float32,
            'avg_crime_global_running': tf.float32,
            'length_running': tf.float32,

            'success_rate': tf.float32,
        }

        with tf.variable_scope(scope):
            with tf.device("/gpu:0"):
                self.state = tf.placeholder(tf.float32, [None, state_dim], name='state')
                self.history = tf.placeholder(tf.float32, [None, history_dim], name='history')
                self.action = tf.placeholder(tf.int32, [None], name='action')
                self.target = tf.placeholder(tf.float32, name='target')

            # self.stats_vars = {}
            # self.stats_placeholders = {}
            # self.update_vars = {}
            # for name, stat_type in self.collected_stats.iteritems():
            #     self.stats_placeholders[name] = tf.placeholder(stat_type, [], name=name)
            #     self.stats_vars[name] = tf.Variable(0, 'tensorboardVar_{}'.format(name), dtype=stat_type)
            #     self.update_vars[name] = self.stats_vars[name].assign(self.stats_placeholders[name])
            #     tf.summary.scalar(name, self.update_vars[name])
            #
            # self.merged_summaries = tf.summary.merge_all()

            with tf.device("/gpu:0"):
                self.action_prob = policy_nn(self.state, self.history, state_dim, history_dim,  action_space, self.initializer)

                action_mask = tf.cast(tf.one_hot(self.action, depth=action_space), tf.bool)
            self.picked_action_prob = tf.boolean_mask(self.action_prob, action_mask)
            with tf.device("/gpu:0"):
                self.loss = tf.reduce_sum(-tf.log(self.picked_action_prob + 1e-6) * self.target) + sum(
                    tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope))
                self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                self.train_op = self.optimizer.minimize(self.loss)

        self.summary_dir_prefix = '/home/sharon/SafeRoute/scripts/tensorboard/'
        self.summary_writer = None

    def predict(self, state, history,  sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action_prob, {self.state: state, self.history: history})

    def update(self, state, target, history, action, sess=None):
        sess = sess or tf.get_default_session()

        feed_dict = {self.state: state, self.target: target, self.history: history, self.action: action}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        # ipdb.set_trace()
        return loss

    def update_summary_stats(self, step, collected_stats):
        #feed_dict = {self.stats_placeholders[name]: np.array(v) for name, v in collected_stats.iteritems()}
        #summary = session.run(self.merged_summaries, feed_dict=feed_dict)
        #self.summary_writer.add_summary(summary, step)
        if step % 20 != 0 or step == 0:
            return
        for name, val in collected_stats.iteritems():
            summary = tf.Summary()
            summary.value.add(tag=name, simple_value=val)
            self.summary_writer.add_summary(summary, step)
            #self.summary_writer.add_summary(tf.summary.scalar(name, np.array(val)).eval())

    def update_for_epoch(self, epoch):
        self.summary_writer = tf.summary.FileWriter(self.summary_dir_prefix + RUN_IDENTIFIER + '_' + str(epoch),
                                                    graph=tf.get_default_graph())

# noinspection PyUnusedLocal
def REINFORCE(training_pairs, policy_nn, num_episodes):
    entity2id_, id2entity_ = getIdAndEntity()
    crimegraph = CrimeGraph(id2entity_, dataPath)
    train = training_pairs
    train = list(np.random.permutation(train).tolist())

    success = 0
    avg_combo_gained = 0.0
    avg_local_rewards = 0.0
    rewards_gained = 0.0
    min_dist_gained = 0.0
    avg_crime_dist_gained = 0.0
    avg_length = 0.0
    avg_count = 0.0

    curr_step = 0

    # path_found = set()
    path_found_entity = []
    path_relation_found = []
    best_path_lengths = {}
    avg_rewards = 0.
    for i_episode in range(len(train)):
        best = -1.0
        best_state_batch = []
        best_history_batch = []
        best_node_batch = []
        best_action_batch = []
        best_env = Env(dataPath, crimegraph)
        best_length = 10000.0
        best_dist = 0.0
        best_count = 1000.0
        best_min_dist = 0.0
        best_risk = 10000000000.0
        best_combo = 0.0
        # best_reward = 0.0
        best_path = []
        best_len = 1000
        # current_paths = []
        for i in range(5):
            tot_crime_reward = 0
            start = time.time()
            print 'Episode %d' % i_episode
            print 'Training sample: ', train[i_episode][:-1]

            env = Env(dataPath, crimegraph)

            sample = train[i_episode].split()
            state_idx = [env.entity2id_[sample[0]], env.entity2id_[sample[1]], 0]

            episode = []
            state_batch_negative = []
            path_length = 0.0

            for t in count():
                state_vec = env.idx_state(state_idx)
                #import ipdb; ipdb.set_trace()
                history_vec = env.history_state()
                action_probs = policy_nn.predict(state_vec, history_vec)
                #ipdb.set_trace()
                choices = set()
                for line in env.kb:
                    triple = line.rsplit()
                    e1_idx = env.entity2id_[triple[0]]
                    if state_idx[0] == e1_idx and triple[1] in env.entity2id_:

                        # don't choose an action that has been chosen before in the same state
                        if triple[0] in env.visited and env.relation2id_[triple[2]] not in env.visited[triple[0]]:
                            choices.add(triple[2])
                        elif triple[0] not in env.visited:
                            choices.add(triple[2])

                choices_idx = []
                action_probs_avail = []
                for c in choices:
                    choices_idx.append(env.relation2id_[c])
                    action_probs_avail.append(action_probs[0][env.relation2id_[c]])
                norm_action_probs = [float(i) / sum(action_probs_avail) for i in action_probs_avail]
                #ipdb.set_trace()
                # print norm_action_probs
                # print choices_idx
                # print env.id2entity_[state_idx[0]]
                # ipdb.set_trace()
                if len(choices_idx) != 0:
                    if random.random() < 0.1:
                        action_chosen = np.random.choice(np.asarray(choices_idx))
                    else:
                        action_chosen = np.random.choice(np.asarray(choices_idx), p=np.asarray(norm_action_probs))

                    reward, new_state, length, done = env.interact(state_idx, action_chosen, 0, norm_action_probs[choices_idx.index(action_chosen)])
                    tot_crime_reward += reward

                # intermediate rewards
                # policy_nn.update(np.reshape([state_vec], (-1,state_dim)), reward, [action_chosen])

                path_length += length
                new_state_vec = env.idx_state(new_state)
                episode.append(Transition(state=state_vec, history=history_vec, action=action_chosen, next_state=new_state_vec, reward=reward))

                if done or t == max_steps:
                    break
                state_idx = new_state

            # Discourage the agent when it choose an invalid step
            if len(state_batch_negative) != 0:
                print 'Penalty to invalid steps:', len(state_batch_negative)
                # policy_nn.update(np.reshape(state_batch_negative, (-1, state_dim)), -0.05, action_batch_negative)

            # If the agent success, do one optimization
            if done == 1:
                # if env.curr_num_crimes != 0.0:
                #     avg_crime_reward = env.curr_crime_dist/env.curr_num_crimes
                #     avg_crime_reward /= float(env.path_length)
                # else:
                #     avg_crime_reward = 10.0

                print 'Success'

                # success += 1
                # path_length = len(env.path)
                # length_reward = 1 / path_length
                # global_reward = 1
                #
                # total_reward = 0.1 * global_reward + 0.9 * length_reward
                state_batch = []
                action_batch = []
                history_batch = []
                node_batch = []
                curr_node = sample[0]
                new_path = []
                new_path_crime_length = 0.0
                new_path_crime_count = 0.0
                new_path_crime_dist = 0.0
                new_path_min_dist = 0.0
                new_path_avg_crime_dist = 0.0
                new_path_risk = 0.0
                new_path_combo = 0.0
                # post-process path to remove cycles before rewarding
                while True:
                    a = env.last_visited[curr_node][0]

                    state_batch.append(env.idx_state([env.entity2id_[curr_node], env.entity2id_[sample[1]], 0]))
                    history_batch.append(env.history_state(env.last_visited[curr_node][2]))
                    node_batch.append(curr_node)
                    action_batch.append(a)
                    new_path.append(curr_node)
                    env.visited[curr_node].discard(a)
                    crime_transition = env.crime_transition_rewards[curr_node][env.last_visited[curr_node][1]]
                    new_path_crime_dist += crime_transition[0]
                    new_path_crime_count += crime_transition[1]
                    new_path_crime_length += crime_transition[2]
                    new_path_min_dist += crime_transition[3]
                    new_path_avg_crime_dist += crime_transition[4]
                    new_path_risk += crime_transition[5]
                    curr_node = env.last_visited[curr_node][1]

                    # for line in env.kb:
                    # 	triple = line.rsplit()
                    # 	#e1_idx = self.entity2id_[triple[0]]
                    # 	if curr_node == triple[0] and env.relation2id_[triple[2]] == a and (triple[1] in env.visited or triple[1] == sample[1]) and triple[1] not in new_visited:
                    # 		state_batch.append(env.idx_state([env.entity2id_[curr_node], env.entity2id_[sample[1]], 0]))
                    # 		action_batch.append(a)
                    # 		new_path.append(curr_node)
                    # 		env.visited[curr_node].discard(a)
                    # 		new_visited.append(curr_node)
                    # 		curr_node = triple[1]
                    # 		#print curr_node
                    # 		break
                    if curr_node == sample[1]:
                        new_path.append(curr_node)
                        break

                # calculate avg distance from crimes reward
                if new_path_crime_count != 0.0:
                    new_path_avg_reward = new_path_crime_dist / new_path_crime_count
                    new_path_avg_reward /= float(new_path_crime_length)
                    new_path_combo = new_path_crime_dist / (new_path_crime_count*new_path_crime_count * float(new_path_crime_length))
                else:
                    new_path_avg_reward = 2.0
                    new_path_combo = 2.0

                # check if this path is better than current best path for the episode
                # if (new_path_avg_reward > best and done) or (
                #     done and new_path_avg_reward >= best and len(state_batch) < best_len):
                if (new_path_crime_count < best_count and done) or (
                        done and new_path_crime_count <= best_count and len(state_batch) < best_len):

                    best_env = env.copy()
                    best_len = len(state_batch)
                    best = new_path_avg_reward
                    best_length = new_path_crime_length
                    best_dist = new_path_crime_dist
                    best_count = new_path_crime_count
                    best_min_dist = new_path_min_dist
                    best_avg_crime_dist = new_path_avg_crime_dist
                    best_risk = new_path_risk
                    best_state_batch = state_batch
                    best_history_batch = history_batch
                    best_node_batch = node_batch
                    best_action_batch = action_batch
                    best_combo = new_path_combo
                    # best_reward = total_reward
                    # best_dist = path_length
                    print "CRIMES--------------------------------------------------------------------------"
                    print new_path_avg_reward
                    print new_path
                    print env.path_no_relations
                    print "crime count: %f" % new_path_crime_count
                    print "crime length: %f" % new_path_crime_length
                    best_path = new_path
                    # path = path_clean(' -> '.join(env.path))
                else:
                    print [sample[0]] + env.path_no_relations
                    print new_path_avg_reward
                    print "crime count: %f" % new_path_crime_count
                    print "crime length: %f" % new_path_crime_length
                print "state_batch len: %d" % len(state_batch)
            else:
                global_reward = -1
                # length_reward = 1/len(env.path)

                state_batch = []
                action_batch = []
                history_batch = []
                total_reward = global_reward
                for t, transition in enumerate(episode):
                    if transition.reward == 0:
                        state_batch.append(transition.state)
                        action_batch.append(transition.action)
                        history_batch.append(transition.history)

                # punish the path that has not reached the target
                # policy_nn.update(np.reshape(state_batch, (-1, state_dim)), total_reward, action_batch)
                print 'Failed'
                if env.curr_num_crimes != 0.0:
                    avg_reward = env.curr_crime_dist / env.curr_num_crimes
                    avg_reward /= float(env.path_length)
                else:
                    avg_reward = 2.0
                print 'CRIME REWARDS', avg_reward
            print 'Episode time: ', time.time() - start
            print '\n'

        # update the best path found
        if best != -1.0:
            curr_dist = 0.0
            curr_count = 0.0
            curr_length = 0.0
            curr_risk = 0.0
            curr_min_dist = 0.0
            curr_num_hops = best_len
            #discounted_rewards = best*best_length*0.5 + best_min_dist*0.25 + best_risk/best_len * 0.25
            discounted_rewards = best_count
            #discounted_rewards = best_risk / best_len
            print "state: ", len(best_state_batch)
            print "node: ", len(best_node_batch)
            print "action: ", len(best_action_batch)
            # ipdb.set_trace()
            for s, node, a, h in zip(best_state_batch, best_node_batch, best_action_batch, best_history_batch):
                print "Curr rewards: ", discounted_rewards
                final_reward = discounted_rewards - avg_rewards/(success or 1)
                final_reward = -1 * (best_count - avg_rewards/(success or 1))
                #policy_nn.update(np.reshape([s], (-1, state_dim)), 10 * (discounted_rewards), [a])
                #policy_nn.update(np.reshape([s], (-1, state_dim)), 10 * (discounted_rewards), [a])
                if final_reward > 0.:
                    policy_nn.update(np.reshape([s], (-1, state_dim)), 10 * (final_reward), h,[a])
                    print "subtracted: ", discounted_rewards - avg_rewards/(success or 1)
                #ipdb.set_trace()
                transition_rewards = best_env.crime_transition_rewards[node][best_env.last_visited[node][1]]
                curr_risk += transition_rewards[5]
                curr_min_dist += transition_rewards[3]
                curr_num_hops -= 1
                if curr_num_hops == 0:
                    continue
                if transition_rewards[1] != 0.0:
                    curr_dist += transition_rewards[0]
                    curr_count += transition_rewards[1]
                    curr_length += transition_rewards[2]
                    if best_count-curr_count == 0:
                        #discounted_rewards = (2.0)*0.5 + ((best_risk - curr_risk) / curr_num_hops) * 0.25 + ((best_min_dist - curr_min_dist) / curr_num_hops) * 0.25
                        discounted_rewards = (2.0)
                    else:
                        #discounted_rewards = ((best_dist-curr_dist) / (best_count-curr_count)) * 0.5 + ((best_risk - curr_risk) / curr_num_hops) * 0.25 + ((best_min_dist - curr_min_dist) / curr_num_hops) * 0.25
                        discounted_rewards = ((best_dist - curr_dist) / (best_count - curr_count))
                else:
                    if best_count == 0.:
                        # discounted_rewards = (2.0) * 0.5 + (
                        #             (best_risk - curr_risk) / curr_num_hops) * 0.25 + (
                        #             (best_min_dist - curr_min_dist) / curr_num_hops) * 0.25
                        discounted_rewards = (2.0)
                    else:
                        # discounted_rewards = ((best_dist) / (best_count)) * 0.5 + (
                        #         (best_risk - curr_risk) / curr_num_hops) * 0.25 + (
                        #         (best_min_dist - curr_min_dist) / curr_num_hops) * 0.25
                        discounted_rewards = ((best_dist) / (best_count))
                    #ipdb.set_trace()
                   # discounted_rewards += avg_reward/(best_length-curr_length)
                #ipdb.set_trace()
                #curr_risk += transition_rewards[5]
                #curr_num_hops -= 1
                #discounted_rewards = (best_risk - curr_risk) / curr_num_hops

            # policy_nn.update(np.reshape(best_state_batch, (-1, state_dim)), 10 * discounted_rewards, best_action_batch)
            avg_rewards += best_count
            success += 1
            if best_count == 0.0:
                avg_local_rewards = best
                rewards_gained += best
                avg_combo_gained += best_combo
            else:
                avg_local_rewards = best * best_length
                rewards_gained += best * best_length
                avg_combo_gained += best_combo * best_length
            # import ipdb; ipdb.set_trace()
            min_dist_gained += best_min_dist/(len(best_path) - 1)
            avg_crime_dist_gained += best_avg_crime_dist / (len(new_path) - 1)
            avg_length += best_length
            avg_count += best_count
            print best_path
            print 'FINAL PATH LENGTH', len(best_path)
            print 'FINAL CRIME REWARDS', best_combo
            print '----------------------------------------------------------------'

        best_path_lengths[i_episode] = best
        print 'Success so far:', success / (i_episode + 1)
        print 'Average reward so far: ', rewards_gained / (success or 1)
        print 'Average combo so far: ', avg_combo_gained / (success or 1)
        print 'Average min_dist so far: ', min_dist_gained / (success or 1)
        print 'Average crime_dist valid: ', avg_crime_dist_gained / (float(success) or 1)
        print 'Average count so far: ', avg_count / (float(success) or 1)
        print 'Average length so far: ', avg_length / (success or 1)

        summary_stats = {'avg_crime_local': avg_local_rewards,
            'avg_crime_min_local': best_min_dist/(len(best_path) - 1),
            'avg_crime_global': best_min_dist/(len(best_path) - 1),
            'length': best_length,

            'avg_crime_local_running': rewards_gained / (success or 1),
            'avg_crime_min_local_running': min_dist_gained / (success or 1),
            'avg_crime_global_running': avg_crime_dist_gained / (float(success) or 1),
            'length_running': avg_length / (success or 1),

            'success_rate': success / (i_episode + 1),
        }

        policy_nn.update_summary_stats(curr_step, summary_stats)
        curr_step += 1
    for k, v in best_path_lengths.items():
        print "Episode: %d   path length: %f" % (k, v)
    print 'Success percentage:', success / len(train)
    print 'Average reward so far: ', rewards_gained / (success or 1)
    print 'Average combo so far: ', avg_combo_gained / (success or 1)
    print 'Average min_dist so far: ', min_dist_gained / (success or 1)
    print 'Average crime_dist valid: ', avg_crime_dist_gained / (float(success) or 1)
    print 'Average count so far: ', avg_count / (float(success) or 1)
    print 'Average length so far: ', avg_length / (success or 1)

    # for path in path_found_entity:
    #     rel_ent = path.split(' -> ')
    #     path_relation = []
    #     for idx, item in enumerate(rel_ent):
    #         if idx % 2 == 0:
    #             path_relation.append(item)
    #     path_relation_found.append(' -> '.join(path_relation))
    #
    # relation_path_stats = collections.Counter()
    # relation_path_stats.update(path_relation_found)
    #
    # sorted_relation_path_stats = sorted(relation_path_stats.items(), key=lambda x: x[1], reverse=True)
    #
    # f = open(dataPath + 'path_stats.txt', 'w')
    # for item in sorted_relation_path_stats:
    #     f.write(item[0] + '\t' + str(item[1]) + '\n')
    # f.close()
    # print 'Path stats saved'

    return


def retrain():
    print 'Start retraining'
    tf.reset_default_graph()
    policy_network = PolicyNetwork(scope='supervised_policy')

    f = open(file_name)
    training_pairs = f.readlines()
    f.close()
    saver = tf.train.Saver()
    for i in range(20):
        policy_network.update_for_epoch(i)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            if i == 0 and first_retrain:
                saver.restore(sess, 'models/policy_supervised_' + relation)
                print "sl_policy restored"
            else:
                saver.restore(sess, 'models/policy_retrained' + relation)
                print 'Model reloaded'
            episodes = len(training_pairs)
            if episodes > 500000:
                episodes = 500000
            REINFORCE(training_pairs, policy_network, episodes)
            test_valid(valid_file, policy_network)
            saver.save(sess, 'models/policy_retrained' + relation)
    print 'Retrained model saved'


def getActions(state_idx, env, policy_network):
    state_vec = env.idx_state(state_idx)
    action_probs = policy_network.predict(state_vec, env.history_state())
    choices = set()
    new_states = []
    for line in env.kb:
        triple = line.rsplit()
        e1_idx = env.entity2id_[triple[0]]
        if state_idx[0] == e1_idx and triple[1] in env.entity2id_:
            if triple[0] in env.visited and env.relation2id_[triple[2]] not in env.visited[triple[0]]:
                choices.add(triple[2])
            elif triple[0] not in env.visited:
                choices.add(triple[2])
                # else:
                #     action_probs[0][env.relation2id_[triple[2]]] /= 1.2
            #choices.add(triple[2])
            new_states.append(triple[1])
    return action_probs, choices, new_states


def _normalize_probs(p):
    if all(prob == 0 for prob in p):
        return [1.0 / len(p) for prob in p]
    return [float(i) / sum(p) for i in p]

def _create_beam_search_candidate_paths_all(s, env, choices, scores):
    if len(choices) == 0:
        ipdb.set_trace()
    candidate_paths = []
    norm_action_probs = _normalize_probs(scores)
    for action_chosen_idx, action_chosen in enumerate(choices):
        new_env = env.copy()
        reward, new_state, length, done = new_env.interact(s, action_chosen, 0, norm_action_probs[action_chosen_idx])
        candidate_paths.append((new_state, new_env))

    return candidate_paths

def _create_beam_search_candidate_paths_prob(s, env, choices, scores):
    candidate_paths = []
    norm_action_probs = _normalize_probs(scores)
    chosen_actions = set()
    # x = np.random.rand()
    # if x <= 0.0:
    #     action_chosen_idx = np.random.choice(len(choices))
    #     action_chosen = choices[action_chosen_idx]
    # else:
    #     action_chosen_idx = np.argmax(np.asarray(norm_action_probs))
    #     action_chosen = choices[action_chosen_idx]
    # #ipdb.set_trace()
    # new_env = env.copy()
    # reward, new_state, length, done = new_env.interact(s, action_chosen, 0, norm_action_probs[action_chosen_idx])
    # candidate_paths.append((new_state, new_env))
    for i in range(10):
        action_chosen_idx = np.random.choice(len(choices), p=np.asarray(norm_action_probs))
        if action_chosen_idx in chosen_actions:
            continue
        chosen_actions.add(action_chosen_idx)
        new_env = env.copy()
        if len(choices) == 0:
            ipdb.set_trace()

        action_chosen = choices[action_chosen_idx]
        reward, new_state, length, done = new_env.interact(s, action_chosen, 0, norm_action_probs[action_chosen_idx])
        candidate_paths.append((new_state, new_env))

    return candidate_paths



# beam search with crime rewards as heuristic
def test_valid(test_file, policy_network):
    entity2id_, id2entity_ = getIdAndEntity()
    crimegraph = CrimeGraph(id2entity_, dataPath)
    f = open(test_file)
    all_data = f.readlines()
    f.close()

    test_data = all_data
    test_num = len(test_data)

    success = 0

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    print 'Model reloaded'


    best_path_lengths = {}
    rewards_gained = 0.0
    min_dist_gained = 0.0
    avg_crime_dist_gained = 0.0
    combo_gained = 0.0
    avg_len = 0.0
    avg_count = 0.0
    looped = 0
    no_success = []
    for episode in xrange(test_num):
        print 'Test sample %d: %s' % (episode, test_data[episode][:-1])
        env = Env(dataPath, crimegraph)
        sample = test_data[episode].split()
        state_idx = [env.entity2id_[sample[0]], env.entity2id_[sample[1]], 0]
        # path_length = 0.0
        curr_paths = [(state_idx, env)]
        ended_paths = []
        for step_count in range(max_steps):
            next_paths = []
            for s, e in curr_paths:
                check_debug()

                # agent has reached target, calculate avg crime score
                if e.done:
                    ended_paths.append((e.curr_prob, s, e))
                    continue
                action_probs, choices, new_states = getActions(s, e, policy_network)
                if len(choices) == 0:
                    break
                choices_idx = []
                action_probs_avail = []
                for c in choices:
                    choices_idx.append(e.relation2id_[c])
                    action_probs_avail.append(action_probs[0][e.relation2id_[c]])

                candidate_paths = _create_beam_search_candidate_paths_prob(s, e, choices_idx, action_probs_avail)
                next_paths.extend(candidate_paths)

            pq = PriorityQueue()
            for candidate, cand_env in next_paths:
                pq.put((cand_env.curr_prob, candidate, cand_env))
                if pq.qsize() > 10 - len(ended_paths):
                    pq.get()

            curr_paths = [(candidate, env) for prob, candidate, env in pq.queue]
            if len(ended_paths) == 10 or len(curr_paths) == 0:
                break
        if len(ended_paths) > 0:
            highest_prob, best_state, best_env = max([(p, s, e) for p, s, e in ended_paths])
            success += 1
            curr_node = sample[0]
            new_path = []
            print best_env.path
            print 'FINAL PATH LENGTH', len(best_env.path)
            print '----------------------------------------------------------------'
            new_path_probs = 1.0
            new_path_crime_length = 0.0
            new_path_crime_count = 0.0
            new_path_crime_dist = 0.0
            new_path_min_dist = 0.0
            new_path_avg_crime_dist = 0.0
            new_path_combo = 0.0
            while True:
                check_debug()
                # a = best_env.last_visited[curr_node][0]
                new_path.append(curr_node)
                new_path_probs *= best_env.crime_transition_probs[curr_node][best_env.last_visited[curr_node][1]]
                crime_transition = best_env.crime_transition_rewards[curr_node][
                    best_env.last_visited[curr_node][1]]
                new_path_crime_dist += crime_transition[0]
                new_path_crime_count += crime_transition[1]
                new_path_crime_length += crime_transition[2]
                new_path_min_dist += crime_transition[3]
                new_path_avg_crime_dist += crime_transition[4]
                # if crime_transition[1]:
                #     new_path_combo += crime_transition[0] / (crime_transition[1]*crime_transition[1])
                # else:
                #     new_path_combo += crime_transition[3]
                curr_node = best_env.last_visited[curr_node][1]
                if curr_node == sample[1]:
                    new_path.append(curr_node)
                    break
            #new_path_combo /= (new_path_crime_length * (len(new_path)-1))
            if new_path_crime_count != 0.0:
                new_path_avg_reward = new_path_crime_dist / new_path_crime_count
                new_path_avg_reward /= float(new_path_crime_length)
                new_path_combo = new_path_crime_dist / (new_path_crime_count*new_path_crime_count)
            else:
                new_path_avg_reward = 2.0
                new_path_avg_reward = new_path_min_dist / (len(new_path) - 1)
                new_path_combo = new_path_min_dist / (len(new_path) - 1)
            print new_path
            print 'FINAL PATH LENGTH', len(new_path)
            print 'FINAL PROB', new_path_probs
            print 'FINAL CRIME', new_path_avg_reward
            print '----------------------------------------------------------------'
            if new_path_crime_count == 0.0:
                rewards_gained += new_path_avg_reward
                best_path_lengths[episode] = new_path_avg_reward
            else:
                rewards_gained += new_path_avg_reward * new_path_crime_length
                best_path_lengths[episode] = new_path_avg_reward * new_path_crime_length
            print "CURRENT PATH MIN DIST: {}".format(new_path_min_dist / (len(new_path) - 1))
            if len(new_path) != len(best_env.path) + 1:
                looped += 1
            min_dist_gained += new_path_min_dist / (len(new_path) - 1)
            combo_gained += new_path_combo
            avg_count += new_path_crime_count
            avg_crime_dist_gained += new_path_avg_crime_dist / (len(new_path) - 1)
            new_path_crime_length += new_path_crime_length / (len(new_path) - 1)
            avg_len += float(new_path_crime_length)
            #best_path_lengths[episode] = len(new_path)
        else:
            no_success.append(episode)
    for k, v in best_path_lengths.items():
        print "Episode: %d   path length: %d" % (k, v)
    print 'Success percentage:', float(success) / float(test_num)
    print 'Average reward valid: ', rewards_gained / (float(success) or 1)
    print 'Average combo valid: ', combo_gained / (float(success) or 1)
    print 'Average min_dist valid: ', min_dist_gained / (float(success) or 1)
    print 'Average crime_dist valid: ', avg_crime_dist_gained / (float(success) or 1)
    print 'Average path length so far: ', avg_len / (success or 1)
    print 'Average count so far: ', avg_count / (float(success) or 1)
    print 'Paths with loops: ', looped / float(test_num)
    print no_success


# beam search with policy probabilities as heuristic
def test(test_file):
    entity2id_, id2entity_ = getIdAndEntity()
    crimegraph = CrimeGraph(id2entity_, dataPath)
    # ipdb.set_trace()
    tf.reset_default_graph()
    policy_network = PolicyNetwork(scope='supervised_policy')
    f = open(test_file)
    all_data = f.readlines()
    f.close()

    test_data = all_data
    test_num = len(test_data)

    success = 0

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    no_success = []
    with tf.Session(config=config) as sess:
        saver.restore(sess, 'models/policy_retrained' + relation)
        print 'Model reloaded'

        if test_num > 20000:
            test_num = 20000
        best_path_lengths = {}
        rewards_gained = 0.0
        min_dist_gained = 0.0
        num_gained = 0.0
        avg_crime_dist_gained = 0.0
        combo_gained = 0.0
        diff_local_avg_gained = 0.0
        diff_combo_gained = 0.0
        avg_len = 0.0
        looped = 0
        for episode in xrange(test_num):
            print 'Test sample %d: %s' % (episode, test_data[episode][:-1])
            env = Env(dataPath, crimegraph)
            sample = test_data[episode].split()
            state_idx = [env.entity2id_[sample[0]], env.entity2id_[sample[1]], 0]
            # path_length = 0.0
            curr_paths = [(state_idx, env)]
            ended_paths = []
            #past_coor = tuple([float(x) for x in env.id2entity_[state_idx[0]].split(",")])
            target_coor = tuple([float(x) for x in env.id2entity_[state_idx[1]].split(",")])
            for step_count in range(max_steps):
                next_paths = []

                removePaths = []
                # if step_count % 5 == 0 and step_count != 0:
                #     #ipdb.set_trace()
                #     for i, (s, e) in enumerate(curr_paths):
                #         curr_coor = tuple([float(x) for x in e.id2entity_[s[0]].split(",")])
                #         curr_length = great_circle(curr_coor, target_coor).miles
                #         past_length = great_circle(e.past_coor, target_coor).miles
                #         if curr_length >= past_length:
                #             #ipdb.set_trace()
                #             removePaths.append(i)
                #             #ipdb.set_trace()
                #         e.past_coor = curr_coor
                #     for i in reversed(removePaths):
                #         del curr_paths[i]
                #     removePaths = []
                for s, e in curr_paths:
                    check_debug()

                    # agent has reached target, calculate avg crime score
                    if e.done:
                        ended_paths.append((e.curr_prob, s, e))
                        continue
                    action_probs, choices, new_states = getActions(s, e, policy_network)
                    if len(choices) == 0:
                        break
                    choices_idx = []
                    action_probs_avail = []
                    for c in choices:
                        choices_idx.append(e.relation2id_[c])
                        action_probs_avail.append(action_probs[0][e.relation2id_[c]])
                    src_id = s[0]
                    dst_id = s[1]
                    #direction_weights = playout_moving_closer_weights(e.crime_graph.g, src_id, dst_id, num_playouts=20, num_steps=5)
                    direction_weights = {d: 1.0 for d in choices}

                    action_probs_weighted = [p * direction_weights[c] for c, p in zip(choices, action_probs_avail)]

                    candidate_paths = _create_beam_search_candidate_paths_prob(s, e, choices_idx, action_probs_weighted)
                    next_paths.extend(candidate_paths)

                pq = PriorityQueue()
                for candidate, cand_env in next_paths:
                    pq.put((cand_env.curr_prob, candidate, cand_env))
                    if pq.qsize() > 5 - len(ended_paths):
                        pq.get()

                curr_paths = [(candidate, env) for prob, candidate, env in pq.queue]
                if len(ended_paths) == 5 or len(curr_paths) == 0:
                    break
            if len(ended_paths) > 0:
                #highest_prob, best_state, best_env = max([(p, s, e) for p, s, e in ended_paths])
                best_env = None
                max_avg = 0.0
                for (p,s,e) in ended_paths:
                    if e.curr_num_crimes != 0.0:
                        avg = e.curr_crime_dist/e.curr_num_crimes
                    else:
                        avg = 0.1
                    if avg > max_avg:
                        max_avg = avg
                        best_env = e
                    # avg = e.curr_num_crimes
                    # if avg <= max_avg:
                    #     max_avg = avg
                    #     best_env = e
                success += 1
                curr_node = sample[0]
                new_path = []
                print best_env.path
                print 'FINAL PATH LENGTH', len(best_env.path)
                print '----------------------------------------------------------------'
                new_path_probs = 1.0
                new_path_crime_length = 0.0
                new_path_crime_count = 0.0
                new_path_crime_dist = 0.0
                new_path_min_dist = 0.0
                new_path_avg_crime_dist = 0.0
                new_path_combo = 0.0
                new_path_diff_local = 0.0
                new_path_diff_combo = 0.0
                while True:
                    check_debug()
                    # a = best_env.last_visited[curr_node][0]
                    new_path.append(curr_node)
                    new_path_probs *= best_env.crime_transition_probs[curr_node][
                        best_env.last_visited[curr_node][1]]
                    crime_transition = best_env.crime_transition_rewards[curr_node][
                        best_env.last_visited[curr_node][1]]
                    new_path_crime_dist += crime_transition[0]
                    new_path_crime_count += crime_transition[1]
                    new_path_crime_length += crime_transition[2]
                    new_path_min_dist += crime_transition[3]
                    new_path_avg_crime_dist += crime_transition[4]
                    if crime_transition[1]:
                        new_path_diff_combo += crime_transition[0] / (crime_transition[1] * crime_transition[1])
                        new_path_diff_local += crime_transition[0] / (crime_transition[1])
                    else:
                        new_path_diff_combo += crime_transition[3]
                        new_path_diff_local += crime_transition[3]
                    curr_node = best_env.last_visited[curr_node][1]
                    if curr_node == sample[1]:
                        new_path.append(curr_node)
                        break

                #new_path_combo /= (len(new_path) - 1)
                if new_path_crime_count != 0.0:
                    new_path_avg_reward = new_path_crime_dist / new_path_crime_count
                    print new_path_avg_reward
                    new_path_avg_reward /= float(new_path_crime_length)
                    new_path_combo = new_path_crime_dist / (new_path_crime_count*new_path_crime_count)
                    print new_path_combo

                else:
                    #new_path_avg_reward = 2.0
                    new_path_avg_reward = 0.5
                    new_path_avg_reward = new_path_min_dist / (len(new_path) - 1)
                    new_path_combo = new_path_min_dist / (len(new_path) - 1)
                num_gained += new_path_crime_count
                combo_gained += new_path_combo
                diff_local_avg_gained += new_path_diff_local / (len(new_path) - 1)
                diff_combo_gained += new_path_diff_combo / (len(new_path) - 1)
                if new_path_crime_count == 0.0:
                    rewards_gained += new_path_avg_reward
                    best_path_lengths[episode] = (new_path_avg_reward, new_path_min_dist / (len(new_path)-1))
                else:
                    rewards_gained += new_path_avg_reward * new_path_crime_length
                    best_path_lengths[episode] = (new_path_avg_reward * new_path_crime_length, new_path_min_dist / (len(new_path)-1))
                min_dist_gained += new_path_min_dist / (len(new_path)-1)
                #avg_crime_dist_gained += (1.0/math.sqrt(new_path_avg_crime_dist / (len(new_path) - 1)))
                avg_crime_dist_gained += (new_path_avg_crime_dist) / (len(new_path) - 1)
                avg_len += float(new_path_crime_length)
                #print new_path
                print 'FINAL PATH LENGTH', len(new_path)
                print 'Success so far:', success / (episode + 1)
                print 'Average reward so far: ', rewards_gained / (success or 1)
                print 'Average combo so far: ', combo_gained / (success or 1)
                print 'Average min_dist so far: ', min_dist_gained / (success or 1)
                print 'Average crime_dist valid: ', avg_crime_dist_gained / (float(success) or 1)
                print 'Average length so far: ', avg_len / (success or 1)
                print 'Average count so far: ', num_gained / (success or 1)
                print 'Average diff local so far: ', diff_local_avg_gained / (success or 1)
                print 'Average diff combo so far: ', diff_combo_gained / (success or 1)
                #print 'FINAL PROB', new_path_probs
                #print 'FINAL CRIME', new_path_avg_reward
                print '----------------------------------------------------------------'
                #if orig_path_len != best_len:
                if len(best_env.path) != len(new_path) - 1:
                    looped += 1
                    # #print "CURRENT PATH MIN DIST: {}".format(new_path_min_dist / (len(new_path)-1))
                    # min_dist_gained += new_path_min_dist / (len(new_path)-1)
                    # avg_crime_dist_gained += new_path_avg_crime_dist / (len(new_path) - 1)
                    # avg_len += float(new_path_crime_length)
                #best_path_lengths[episode] = len(new_path)
            else:
                no_success.append(episode)
        #import ipdb; ipdb.set_trace()
        for k, (a,m) in best_path_lengths.items():
            print "Episode: {}   avg: {}   min: {}".format(k, a, m)
        print 'Success percentage:', float(success) / float(test_num)
        print 'Average reward valid: ', rewards_gained / (float(success) or 1)
        print 'Average combo so far: ', combo_gained / (success or 1)
        print 'Average min_dist valid: ', min_dist_gained / (float(success) or 1)
        print 'Average crime_dist valid: ', avg_crime_dist_gained / (float(success) or 1)
        print 'Average path length so far: ', avg_len / (success or 1)
        print 'Average diff local so far: ', diff_local_avg_gained / (success or 1)
        print 'Average count so far: ', num_gained / (success or 1)
        print 'Average diff combo so far: ', diff_combo_gained / (success or 1)
        print 'Paths with loops: ', looped/float(test_num)
        print no_success


if __name__ == "__main__":
    if task == 'test':
        test(file_name)
        # test2()
    elif task == 'retrain':
        retrain()
    else:
        retrain()
        test2(file_name)
    # retrain()
