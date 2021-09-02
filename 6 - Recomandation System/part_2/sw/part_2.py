import networkx as nx
import csv

def compute_top_k(map__node_id__score, k=20):
    list__node_id__score = [(node_id, score) for node_id, score in map__node_id__score.items()]
    list__node_id__score.sort(key=lambda x: (-x[1], x[0]))
    return list__node_id__score[:k]


complete_input_graph_file_name = "./Part_2/dataset/pkmn_graph_data.tsv"
k = 6

# Graph creation by reading the list of weighted edges from file...
input_file_handler = open(complete_input_graph_file_name, 'r', encoding="utf-8")
csv_reader = csv.reader(input_file_handler, delimiter='\t', quotechar='"', quoting=csv.QUOTE_NONE)
list__u_v_weight = []
list_u_v = []
for record in csv_reader:
    u = record[0]
    v = record[1]
    weight = int(1)
    list__u_v_weight.append((u, v, weight))
    list_u_v.append((u, v))
input_file_handler.close()

# graph with all edges has weight 1 (will be used for 2.1)
graph = nx.Graph()
graph.add_weighted_edges_from(list__u_v_weight)
# graph with no edge weight (will be used for 2.2)
graph_2 = nx.Graph()
graph_2.add_edges_from(list_u_v)

# Part 2.1
################################################################################
### The following `topic_specific` function takes as input set of pokemon name(s)
### It retrieves the Top-K nodes in the graph
### using as score the Topic-Specific-PageRank score of a node.
################################################################################

def topic_specific(pokemon_names):
    #
    # Creation of the teleporting probability distribution for the selected Topic
    set__all_x_node_ids = set()
    set__all_NOT_x_node_ids = set()
    for node_id in graph:
        if node_id in pokemon_names:
            set__all_x_node_ids.add(node_id)
        else:
            set__all_NOT_x_node_ids.add(node_id)

    map_teleporting_probability_distribution__node_id__probability = {node_id: 1. / len(set__all_x_node_ids) for
                                                                      node_id in set__all_x_node_ids}
    for node_id in set__all_NOT_x_node_ids:
        map_teleporting_probability_distribution__node_id__probability[node_id] = 0.

    map__node_id__node_pagerank_value = nx.pagerank(graph, alpha=0.33,
                                                    personalization=map_teleporting_probability_distribution__node_id__probability,
                                                    weight='weight')

    # Extract the Top-K node identifiers according to the PageRank score.
    top_k__node_id__node_pagerank_value = compute_top_k(map__node_id__node_pagerank_value, k)
    # nodes set is created to return only the members according to topic-specific pagerank algorithm
    nodes = set()
    for i in range(len(top_k__node_id__node_pagerank_value)):
        nodes.add(top_k__node_id__node_pagerank_value[i][0])
    return nodes

Set_A = {"Pikachu"}
Set_B = set(["Venusaur", "Charizard", "Blastoise"])
Set_C = set(["Excadrill", "Dracovish", "Whimsicott", "Milotic"])
g_a = topic_specific(Set_A)
g_b = topic_specific(Set_B)
g_c = topic_specific(Set_C)
g_1 = topic_specific("Charizard")
g_2 = topic_specific("Venusaur")
g_3 = topic_specific("Kingdra")
g_4 = topic_specific(set(["Charizard", "Venusaur"]))
g_5 = topic_specific(set(["Charizard", "Kingdra"]))
g_6 = topic_specific(set(["Venusaur", "Kingdra"]))

# Compute the number of team members inside the Team(Charizard, Venusaur) that are neither
# in Team(Charizard) nor in Team(Venusaur)
u_1_2 = set().union(g_1, g_2)
print(len(g_4.difference(u_1_2)))

# Compute the number of team members inside the Team(Charizard, Kingdra) that are neither in Team(Charizard)
# nor in Team(Kingdra)
u_1_3 = set().union(g_1, g_3)
print(len(g_5.difference(u_1_3)))

# Compute the number of team members inside the Team(Venusaur, Kingdra) that are
# neither in Team(Venusaur) nor in Team(Kingdra)
u_2_3 = set().union(g_2, g_3)
print(len(g_6.difference(u_2_3)))


# PART 2.2

def compute_good_local_community(graph, seed_node_id, alpha=0.9):
    # Creation of the teleporting probability distribution for the selected node...
    map_teleporting_probability_distribution__node_id__probability = {}
    for node_id in graph:
        map_teleporting_probability_distribution__node_id__probability[node_id] = 0.
    map_teleporting_probability_distribution__node_id__probability[seed_node_id] = 1.
    # Computation of the PageRank vector.
    map__node_id__node_pagerank_value = nx.pagerank(graph, alpha=alpha,
                                                    personalization=map_teleporting_probability_distribution__node_id__probability)
    # Put all nodes in a list and sort the list in descending order of the â€œnormalized_scoreâ€.
    sorted_list__node_id__normalized_score = [(node_id, score / graph.degree[node_id])
                                              for node_id, score in map__node_id__node_pagerank_value.items()]
    sorted_list__node_id__normalized_score.sort(key=lambda x: (-x[1], x[0]))
    # LET'S SWEEP!
    index_representing_the_set_of_node_ids_with_maximum_conductance = -1
    min_conductance_value = float("+inf")
    set__node_ids_in_the_candidate_community = set()
    set__node_ids_in_the_COMPLEMENT_of_the_candidate_community_to_the_entire_set_of_nodes = set(graph.nodes())
    for sweep_index in range(0, len(sorted_list__node_id__normalized_score) - 1):
        # Creation of the set of nodes representing the candidate community and
        # its complement to the entire set of nodes in the graph.
        current_node_id = sorted_list__node_id__normalized_score[sweep_index][0]
        set__node_ids_in_the_candidate_community.add(current_node_id)
        set__node_ids_in_the_COMPLEMENT_of_the_candidate_community_to_the_entire_set_of_nodes.remove(current_node_id)
        #
        # Evaluation of the quality of the candidate community according to its conductance value.
        conductance_value = nx.algorithms.cuts.conductance(graph,
                                                           set__node_ids_in_the_candidate_community,
                                                           set__node_ids_in_the_COMPLEMENT_of_the_candidate_community_to_the_entire_set_of_nodes)
        # Discard local communities with conductance 0 or 1.
        if conductance_value == 0. or conductance_value == 1.:
            continue
        # Discard the nodes in local community if it is the member in the greater than 140's position.
        if len(set__node_ids_in_the_candidate_community) > 140:
            continue
        # Update the values of variables representing the best solution generated so far.
        if conductance_value < min_conductance_value:
            min_conductance_value = conductance_value
            index_representing_the_set_of_node_ids_with_maximum_conductance = sweep_index

    # Creation of the set of nodes representing the best local community generated by the sweeping procedure.
    set__node_ids_with_minimum_conductance = set([node_id for node_id, normalized_score in
                                                  sorted_list__node_id__normalized_score[
                                                  :index_representing_the_set_of_node_ids_with_maximum_conductance + 1]])

    return set__node_ids_with_minimum_conductance, min_conductance_value

# different alpha values will be tried and the best will be considered as the best community
alphas = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
nodes = graph_2.nodes
# `best_conductance_lists` is created to add the best community's conductance value for each pokemon
best_conductance_lists = []
# `local_community_list` is created to add the best community's members for each pokemon
local_community_list = []
for node in nodes:
    # `conductance_list` and `local_community` are created temporarily to choose the best local community
    # for each pokemon at the end of each alpha value tried
    conductance_list = []
    local_community = []
    for a in alphas:
        set_local_community_for_node, conductance_value_for_local_community_for_node = compute_good_local_community(
            graph_2, node, alpha=a)
        conductance_list.append(conductance_value_for_local_community_for_node)
        local_community.append(set_local_community_for_node)
    min_index = conductance_list.index(min(conductance_list))
    best_conductance_lists.append((conductance_list[min_index]))
    local_community_list.append(local_community[min_index])

# `community_frequency` dictionary is created to store pokemon names as key and community_frequency of key(pokemon) as a value
community_frequency = {}
for i in graph_2.nodes:
    community_frequency[i] = 0

for community in local_community_list:
    for pokemon in community:
        community_frequency[pokemon] = community_frequency[pokemon] + 1

sort_com_frequency = sorted(community_frequency.items(), key=lambda x: x[1], reverse=True)
# The most 5 frequent pokemon and its' frequency value
print(sort_com_frequency[:5])
# The least 5 frequent pokemon and its' frequency value
print(sort_com_frequency[-5:])

# Writing the tsv file
with open('./Part_2/output.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(["pokemon_name", "number_of_nodes_in_the_local_comunity", "conductance_value_of_the_local_comunity"])
    for node in sorted(graph_2.nodes):
        index = list(graph_2.nodes).index(node)
        result = [node, len(local_community_list[index]), best_conductance_lists[index]]
        tsv_writer.writerow(result)
