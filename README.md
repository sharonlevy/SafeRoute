# SafeRoute
This repository contains the dataset used in our paper: SafeRoute: Learning to Navigate Streets Safely in an Urban Environment

When using our dataset/code, please cite our paper:
```
@article{10.1145/3402818,
author = {Levy, Sharon and Xiong, Wenhan and Belding, Elizabeth and Wang, William Yang},
title = {SafeRoute: Learning to Navigate Streets Safely in an Urban Environment},
year = {2020},
issue_date = {December 2020},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {11},
number = {6},
issn = {2157-6904},
url = {https://doi.org/10.1145/3402818},
doi = {10.1145/3402818},
journal = {ACM Trans. Intell. Syst. Technol.},
month = {sep},
articleno = {66},
numpages = {17},
keywords = {Safe routing, deep reinforcement learning, multi-preference routing}
}
```

# Format of the dataset
1. edges_\*.txt: list of all connections (streets) between two intersections on a map. Relations are the compass direction between the two points (e.g. North, South, etc)
2. entity2id_\*.txt: mapping of intersections to graph node id. 
3. crime_coors_\*.txt: lat/long coordinates of crimes in map area, sorted by latitude for easy search.
4. node2vec_\*.emd: node2vec graph embeddings of intersections by graph node id.
5. nxGraph_\*.emd: networkx graph of street map.
6. test_\*: testing start and end intersections for 5 and 10 hop paths
7. train_\*: training start and end intersections for 5 and 10 hop paths


## Train
First, go into ```scripts/env.py``` and replace the entity2idInput, node2vec, and edges files at the top with the correct files for the city you train on. 
Second, go into ```scripts/env.py``` and replace the nxGraphFile and crimeFile files at the top with the correct files for the city you train on. 


To pre-train the network with shortest paths: ```python scripts/sl_policy.py $model_name$ $train_file$ $first_train$ $graph_path$ $edges$```
  * you may choose a model_name for your network
  * train_file indicates the train_\* files
  * first_train is a boolean, where 1 is the first time you pre-train the network and 0 is for retraining an exisiting model with the shortest paths
  * graph_path is the correct graph_\* file
  * edges is the correct edges_\* files

To retrain/test the network with rewards: ```python scripts/policy_agent.py $model_name$ $task$ $train_or_test_file$ $first_train$ $valid_file$ $graph_path$```
  * you may choose a model_name for your network
  * task indicates retrain (retrain network with rewards) or test (test fully-trained network)
  * train_or_test_file indicates the train_\* files if retraining and the train_\* if testing
  * first_train is a boolean, where 1 is the first time you retrain the network and 0 is for retraining a model that has already been retrained with rewards
  * valid_file indicates the valid_\* files
  * graph_path is the correct graph_\* file

