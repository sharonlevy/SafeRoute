import numpy as np
import random
from utils import *
from geopy.distance import great_circle
import bisect
import ipdb
import copy

from crime_graph import CrimeGraph

entity2idInput = 'entity2id_ny.txt'
node2vec = 'node2vec_ny.emd'
edges = 'edges_ny.txt'

def getIdAndEntity():
    f1 = open(dataPath + entity2idInput)
    entity2id = f1.readlines()
    f1.close()
    entity2id_ = {}
    id2entity_ = {}
    for line in entity2id:
        entity2id_[line.split()[0]] = int(line.split()[1])
        id2entity_[int(line.split()[1])] = line.split()[0]
    return entity2id_, id2entity_

class Env(object):
    """knowledge graph environment definition"""

    def __init__(self, dataPath, crime_graph):
        f2 = open(dataPath + 'relation2id.txt')
        self.relation2id = f2.readlines()
        f2.close()
        self.entity2id_, self.id2entity_ = getIdAndEntity()
        self.relation2id_ = {}
        self.relations = []
        for line in self.relation2id:
            self.relation2id_[line.split()[0]] = int(line.split()[1])
            self.relations.append(line.split()[0])
        self.node2vec_ = {}
        self.node2vec = open(dataPath + node2vec).readlines()
        for line in self.node2vec:
            self.node2vec_[int(line.split()[0])] = [float(x) for x in line.split()[1:]]
        # self.relation2vec = np.loadtxt(dataPath + 'relation2vec.bern')

        # Knowledge Graph for path finding
        f = open(dataPath + edges)
        kb_all = f.readlines()
        f.close()

        self.kb = []
        for line in kb_all:
            self.kb.append(line)
        # if task != None:
        # 	relation = task.split()[2]
        # 	for line in kb_all:
        # 		rel = line.split()[2]
        # 		if rel != relation and rel != relation + '_inv':
        # 			self.kb.append(line)

        self.crime_graph = crime_graph

        self.path = []
        self.path_no_relations = []
        self.path_relations = []
        self.coords_path = []
        self.curr_rewards = 0.0
        self.curr_prob = 1.0
        self.crime_transition_rewards = {}
        self.crime_transition_probs = {}
        self.curr_num_crimes = 0.0
        self.curr_crime_dist = 0.0
        self.die = 0  # record how many times does the agent choose an invalid path
        self.last_visited = {}
        self.visited = {}
        self.done = 0
        self.steps = 0
        self.path_length = 0.0
        self.past_coor = (0.0,0.0)

    def copy(self):
        o = Env.__new__(Env)
        o.relation2id = self.relation2id
        o.relation2id_ = self.relation2id_
        o.entity2id_ = self.entity2id_
        o.id2entity_ = self.id2entity_
        o.relations = self.relations
        o.node2vec_ = self.node2vec_
        o.kb = self.kb
        o.crime_graph = self.crime_graph

        o.path = list(self.path)
        o.path_relations = list(self.path_relations)
        o.path_no_relations = list(self.path_no_relations)
        o.coords_path = list(self.coords_path)
        o.curr_rewards = self.curr_rewards
        o.curr_prob = self.curr_prob
        o.curr_num_crimes = self.curr_num_crimes
        o.curr_crime_dist = self.curr_crime_dist
        o.crime_transition_rewards = {k: dict(v.items()) for k, v in self.crime_transition_rewards.iteritems()}
        o.crime_transition_probs = {k: dict(v.items()) for k, v in self.crime_transition_probs.iteritems()}

        o.die = self.die  # record how many times does the agent choose an invalid path
        o.last_visited = {k: tuple(v) for k, v in self.last_visited.iteritems()}
        o.visited = {k: set(v) for k, v in self.visited.iteritems()}
        o.done = self.done
        o.steps = self.steps
        o.path_length = self.path_length
        o.past_coor = self.past_coor

        return o

    def interact(self, state, action, rollout, prob):
        """
        This function process the interact from the agent
        state: is [current_position, target_position]
        action: an integer
        return: (reward, [new_postion, target_position], done)
        """
        done = 0  # Whether the episode has finished
        curr_pos = state[0]
        target_pos = state[1]
        chosen_relation = self.relations[action]
        choices = []

        if self.past_coor == (0.0,0.0):
            self.past_coor = self.id2entity_[curr_pos]
        # ipdb.set_trace()
        for line in self.kb:
            triple = line.rsplit()
            e1_idx = self.entity2id_[triple[0]]

            if curr_pos == e1_idx and triple[2] == chosen_relation and triple[1] in self.entity2id_:
                choices.append(triple)
        if len(choices) == 0:
            assert False
        else:  # find a valid step
            path = random.choice(choices)
            self.path.append(path[2] + ' -> ' + path[1])
            self.path_no_relations.append(path[1])
            self.coords_path.append(self.id2entity_[curr_pos])
            # ipdb.set_trace()

            # add action taken to dictionary of state-action pairs
            self.last_visited[self.id2entity_[curr_pos]] = (self.relation2id_[path[2]], path[1], len(self.path) - 1)
            if self.id2entity_[curr_pos] in self.visited:

                self.visited[self.id2entity_[curr_pos]].add(self.relation2id_[path[2]])
            else:
                self.visited[self.id2entity_[curr_pos]] = {self.relation2id_[path[2]]}

            self.die = 0
            new_pos = self.entity2id_[path[1]]
            new_state = [new_pos, target_pos, self.die]

            reward = self.crime_graph.crime_reward(curr_pos, new_pos)
            self.curr_num_crimes += self.crime_graph.crime_count(curr_pos, new_pos)
            self.curr_crime_dist += self.crime_graph.crime_length(curr_pos, new_pos)


            if new_pos == target_pos and not rollout:
                done = 1
                self.done = 1
        # reward = action_length_reward
        # if looped_action:
        # ipdb.set_trace()

        # save above crime data for each node visited for future rewards
        if self.id2entity_[curr_pos] in self.crime_transition_rewards:
            if self.id2entity_[new_pos] not in self.crime_transition_rewards[self.id2entity_[curr_pos]]:
                self.crime_transition_rewards[self.id2entity_[curr_pos]][self.id2entity_[new_pos]] = (
                    self.crime_graph.crime_length(curr_pos, new_pos), self.crime_graph.crime_count(curr_pos, new_pos),
                    self.crime_graph.length(curr_pos, new_pos), self.crime_graph.min_dist(curr_pos, new_pos),
                    self.crime_graph.avg_crime_dist(curr_pos, new_pos), self.crime_graph.risk(curr_pos, new_pos))
        else:
            self.crime_transition_rewards[self.id2entity_[curr_pos]] = {
                self.id2entity_[new_pos]: (self.crime_graph.crime_length(curr_pos, new_pos), self.crime_graph.crime_count(curr_pos, new_pos),
                                           self.crime_graph.length(curr_pos, new_pos), self.crime_graph.min_dist(curr_pos, new_pos),
                                           self.crime_graph.avg_crime_dist(curr_pos, new_pos), self.crime_graph.risk(curr_pos, new_pos))}

        # save probabilities for each node taken
        if self.id2entity_[curr_pos] in self.crime_transition_probs:
            if self.id2entity_[new_pos] not in self.crime_transition_probs[self.id2entity_[curr_pos]]:
                self.crime_transition_probs[self.id2entity_[curr_pos]][self.id2entity_[new_pos]] = prob
        else:
            self.crime_transition_probs[self.id2entity_[curr_pos]] = {
                self.id2entity_[new_pos]: prob}

        self.curr_rewards += reward
        self.curr_prob *= prob
        self.steps += 1
        self.path_length += self.crime_graph.length(curr_pos, new_pos)
        return reward, new_state, self.crime_graph.length(curr_pos, new_pos), done

    def idx_state(self, idx_list):
        if idx_list is not None:
            curr = [float(x) for x in self.id2entity_[idx_list[0]].split(",")]
            targ = [float(x) for x in self.id2entity_[idx_list[1]].split(",")]
            curr = self.node2vec_[idx_list[0]]
            targ = self.node2vec_[idx_list[1]]
            # curr = [2 * (curr[0] + 90)/180 - 1, 2 * (curr[1] + 180) / 360 - 1]
            # targ = [2 * (targ[0] + 90) / 180 - 1, 2 * (targ[1] + 180) / 360 - 1]
            return np.expand_dims(np.concatenate((np.asarray(curr), np.asarray(targ) - np.asarray(curr))), axis=0)
        else:
            return None

    def history_state(self, action_idx=None, count=5):
        node_vec = []
        action_vec = []
        if action_idx is None:
            start = len(self.path_no_relations) - 1
        else:
            start = action_idx

        for i in range(start, start - count, -1):
            if i < 0 or i >= len(self.path_no_relations):
                node_vec.append([0.] * 128)
                action_vec.append([0.] * 8)
            else:
                node = self.path_no_relations[i]
                node_vec.append(self.node2vec_[self.entity2id_[node]])
                action_id = self.relation2id_[self.path[i].split("->")[0].strip()]
                action = [0.] * 8
                action[action_id] = 1
                action_vec.append(action)

        return np.expand_dims(np.concatenate((np.asarray(node_vec[0]), action_vec[0], np.asarray(node_vec[1]), action_vec[1], np.asarray(node_vec[2]),
                                              action_vec[2], np.asarray(node_vec[3]), action_vec[3], np.asarray(node_vec[4]), action_vec[4])), axis=0)

    # if idx_list != None:
    # 	curr = self.entity2vec[idx_list[0],:]
    # 	targ = self.entity2vec[idx_list[1],:]
    # 	return np.expand_dims(np.concatenate((curr, targ - curr)),axis=0)
    # else:
    # 	return None

    def get_valid_actions(self, entityID):
        actions = set()
        for line in self.kb:
            triple = line.split()
            e1_idx = self.entity2id_[triple[0]]
            if e1_idx == entityID:
                actions.add(self.relation2id_[triple[2]])
        return np.array(list(actions))

    def path_embedding(self, path):
        embeddings = [self.relation2vec[self.relation2id_[relation], :] for relation in path]
        embeddings = np.reshape(embeddings, (-1, embedding_dim))
        path_encoding = np.sum(embeddings, axis=0)
        return np.reshape(path_encoding, (-1, embedding_dim))
