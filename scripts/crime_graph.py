import networkx as nx
from geopy.distance import great_circle
import bisect
from scipy.stats import gaussian_kde
import numpy as np
from osm_utils import name_angle, calculate_initial_compass_bearing
import ipdb

nxGraphFile = "nxGraph_ny.gpickle"
crimeFile = "crime_coors_ny.txt"

def getGraph():
    # g = osmgraph.parse_file('boston_map_with_res.osm').to_undirected()  # or .osm or .pbf
    g = nx.read_gpickle(dataPath + nxGraphFile)
    for u, v in g.edges(data=False):
        c1 = g.nodes[u]['coordinate']
        c2 = g.nodes[v]['coordinate']
        g.edges[u, v]['midpoint'] = (np.array(c1) + np.array(c2)) / 2.0
        g.edges[u, v]['length'] = great_circle(c1, c2).miles

        angle = calculate_initial_compass_bearing(tuple(reversed(c1)), tuple(reversed(c2)))
        direction = name_angle(angle)
        g.edges[u, v]['angle'] = direction
        g.edges[u, v]['direction'] = direction

    return g


class CrimeGraph(object):
    def __init__(self, id2entity, dataPath):
        self.g = getGraph()
        self.id2entity = id2entity

        with open(dataPath + crimeFile) as f:
            self.crimes_rev = np.array([map(float, reversed(l.split(' '))) for l in f.read().strip().split('\n')])

        f = open(dataPath + crimeFile)
        crimes_all = f.readlines()
        f.close()

        self.crimes = []
        for line in crimes_all:
            self.crimes.append(tuple([float(x) for x in line.split()]))

        self._populate_crime_info()

    def allCrimeDist(self, midpoint):
        tot_dist = 0.0
        for crime in self.crimes:
            #tot_dist += 1.0/((great_circle(midpoint, crime).miles)*(great_circle(midpoint, crime).miles))
            tot_dist += (great_circle(midpoint, crime).miles)
        return tot_dist / len(self.crimes)

    def minCrimeDist(self, midpoint):
        min = 100.0
        for crime in self.crimes:
            if great_circle(midpoint, crime).miles < min:
                min = great_circle(midpoint, crime).miles
        return min

    def _populate_crime_info(self):
        crime_estimation = gaussian_kde(self.crimes_rev.T)
        total_density = 0.0
        for src_id, dst_id, edge_data in self.g.edges(data=True):
            midpoint = edge_data['midpoint']
            density = crime_estimation.evaluate(midpoint)
            self.g.edges[src_id, dst_id]['density'] = density * edge_data['length']
            total_density += density

            curr_coor = self.g.nodes[src_id]['coordinate']
            new_pos_coor = self.g.nodes[dst_id]['coordinate']

            # distance moved from curr location to new location
            length = great_circle(curr_coor, new_pos_coor).miles
            midpoint = ((curr_coor[1] + new_pos_coor[1]) / 2, (curr_coor[0] + new_pos_coor[0]) / 2)

            # distance in degrees lat and long radii
            # lat_rad = 0.00289855072
            lat_rad = length / 69
            # lon_rad = 0.2 / (new_pos_coor[0] * 3.1415927 / 180.0 * 69.172)
            # lon_rad = 0.0032
            lon_rad = length / 69

            # coordinates for radius around new location
            top_coor = midpoint[0] + lat_rad
            bottom_coor = midpoint[0] - lat_rad
            left_coor = midpoint[1] - lon_rad
            right_coor = midpoint[1] + lon_rad

            crime_count = 0.0
            crime_length = 0.0
            lat, lon = zip(*self.crimes)
            top_index = bisect.bisect_right(lat, top_coor) - 1
            bottom_index = bisect.bisect_left(lat, bottom_coor)
            #min_dist = 2.0
            min_dist = 0.5
            # if src_id == 42437644 and dst_id == 42434946:
            #     ipdb.set_trace()
            # get # crimes and distances from new location in radius
            for i in range(bottom_index, top_index + 1):
                if left_coor <= self.crimes[i][1] <= right_coor:
                    crime_length_check = great_circle(midpoint, self.crimes[i]).miles
                    if crime_length_check <= length:
                        crime_length += crime_length_check
                        crime_count += 1.0
                        if crime_length_check < min_dist:
                            min_dist = crime_length_check

            # intermediate reward
            if crime_count > 0:
                crime_reward = crime_length / crime_count
            # crime_reward = 0.9/crime_count
            else:
                crime_reward = 0.65
            reward = crime_reward
            avg_crime_dist = self.allCrimeDist(midpoint)
            # if src_id == 42434946 and dst_id == 42434948:
            #     ipdb.set_trace()
            min_dist = self.minCrimeDist(midpoint)
            self.g.adj[src_id][dst_id]['crime_count'] = crime_count
            self.g.adj[src_id][dst_id]['crime_length'] = crime_length
            self.g.adj[src_id][dst_id]['crime_reward'] = reward
            self.g.adj[src_id][dst_id]['min_dist'] = min_dist
            self.g.adj[src_id][dst_id]['avg_crime_dist'] = avg_crime_dist

        for src_id, dst_id in self.g.edges():
            self.g[src_id][dst_id]['risk'] = self.g[src_id][dst_id]['density'] / total_density

    def crime_count(self, u, v):
        return self.g.adj[u][v]['crime_count']

    def crime_length(self, u, v):
        return self.g.adj[u][v]['crime_length']

    def length(self, u, v):
        return self.g.adj[u][v]['length']

    def crime_reward(self, u, v):
        return self.g.adj[u][v]['crime_reward']

    def min_dist(self, u, v):
        return self.g.adj[u][v]['min_dist']

    def avg_crime_dist(self, u, v):
        return self.g.adj[u][v]['avg_crime_dist']

    def risk(self, u, v):
        return self.g.adj[u][v]['risk']
