# SafeRoute

SafeRoute allows users to specify and a start and end location on a map and finds a safe and short path between the two points. 

# Format of the dataset
1. edges_\*.txt: list of all connections (streets) between two intersections on a map. Relations are the compass direction between the two points (e.g. North, South, etc)
2. entity2id_\*.txt: mapping of intersections to graph node id. 
3. crime_coors_\*.txt: lat/long coordinates of crimes in map area, sorted by latitude for easy search.
4. node2vec_\*.emd: node2vec graph embeddings of intersections by graph node id.
5. nxGraph_\*.emd: networkx graph of street map.
6. test_\*: testing start and end intersections for 5 and 10 hop paths
7. train_\*: training start and end intersections for 5 and 10 hop paths
