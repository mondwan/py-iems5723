"""
File: 1155002613_hw3.py
Author: Me
Email: 0
Github: 0
Description: Compute betweenness centrality and produce the network graph
"""

import os

import pydot
import networkx as nx

with open(os.path.join('.', 'input.csv'), 'r') as f:
    lines = [l.rstrip('\n') for l in f.readlines()]

# Process graph
graph = pydot.Dot(graph_type='graph')
columns = (lines[0].split(','))[1:]
for col_node_label in columns:
    col_index = columns.index(col_node_label) + 1
    rows = lines[1:]
    for row in rows:
        values = row.split(',')
        row_node_label = values[0]
        connected = True if values[col_index] == '1' else False
        if (connected):
            edge = pydot.Edge(
                col_node_label,
                row_node_label
            )
            graph.add_edge(edge)

# Calculate nodes' betweenness
nx_graph = nx.from_pydot(graph)
results = nx.edge_betweenness_centrality(nx_graph)

# Sort them by descending order
link_betweenness_pairs = sorted(
    results.items(),
    key=lambda x: x[1],
    reverse=True
)

# Add edge's label and change font's colour according to the specificaiton
final_graph = pydot.Dot(graph_type='graph')
max_val = -1
for (edge, betweenness_val) in link_betweenness_pairs:
    # Pen's width should be 5 times of the betweenness_val
    penwidth = betweenness_val * 5

    # Update edge's label and penwidth
    e = graph.get_edge(edge[0], edge[1])
    e.obj_dict['attributes']['label'] = str(betweenness_val)
    e.obj_dict['attributes']['penwidth'] = str(penwidth)

    # Change font's color to red if it is the max betweenness
    # Change font's color to blue otherwise
    if (link_betweenness_pairs.index((edge, betweenness_val)) == 0):
        e.obj_dict['attributes']['fontcolor'] = 'red'
        max_val = betweenness_val
    else:
        if (betweenness_val == max_val):
            e.obj_dict['attributes']['fontcolor'] = 'red'
        else:
            e.obj_dict['attributes']['fontcolor'] = 'blue'

    # Add the edge to the finalized graph
    final_graph.add_edge(e)

# Output required png and txt
final_graph.write('1155002613.png', prog='neato', format='png')
with open(os.path.join('.', '1155002613.txt'), 'w') as f:
    for (edge, betweenness_val) in link_betweenness_pairs:
        # The edge ('A', '2')'s betweenness is 0.4
        f.write(
            "The edge ('%s', '%s')'s betweenness is %0.1f\n" %
            (edge[0], edge[1], betweenness_val)
        )
