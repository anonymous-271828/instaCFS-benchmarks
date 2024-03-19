import os, sys
import torch
import json
import numpy as np
import networkx as nx
import random

from collections import defaultdict
from pomegranate.distributions import Categorical, ConditionalCategorical
from pomegranate.bayesian_network import BayesianNetwork


def process_conditional(cond_info):
    # Iterate through to get max index to get size of array
    parent_idxs = []
    child_idxs = []
    for parent_state, child_state_info in cond_info.items():
        parent_idxs.append(int(parent_state))
        for child_state, prob in child_state_info.items():
            child_idxs.append(int(child_state))
    # Initialize array
    probs = np.zeros((max(child_idxs)+1, max(parent_idxs)+1))
    # Go through again to fill out the CPT
    for parent_state, child_state_info in cond_info.items():
        parent_state = int(parent_state)
        for child_state, prob in child_state_info.items():
            child_state = int(child_state)
            probs[child_state, parent_state] = prob
    return ConditionalCategorical(np.expand_dims(probs, 0))

def process_categorical(info):
    parent_idxs = []
    for parent_state in info:
        parent_idxs.append(int(parent_state))
    probs = np.zeros(max(parent_idxs)+1)
    for parent_state, prob in info.items():
        probs[int(parent_state)] = prob
    return Categorical(np.expand_dims(probs, 0))

def process_additional_dists(dist, nodes_to_add):
    # We start at X30 and will add a 110 variables
    i = 30
    dists = {}
    edges = defaultdict(list)
    for X in nodes_to_add:
        for _ in range(10):
            Z = f'X{i}'
            probs = np.zeros((3,3))
            for parent_state, child_state_info in dist.items():
                parent_state = int(parent_state)
                for child_state, prob in child_state_info.items():
                    child_state = int(child_state)
                    probs[child_state, parent_state] = prob
            dists[Z] = ConditionalCategorical(np.expand_dims(probs, 0))
            edges[X].append(Z)
            i += 1
    return dists, edges

def merge_edges(e1, e2):
    for p, clist in e2.items():
        if p in e1:
            e1[p].extend(clist)
        else:
            e1[p] = clist
    return e1

def extract_edge_list(edges):
    edge_list = []
    for p, clist in edges.items():
        for c in clist:
            edge_list.append((p, c))
    return list(set(edge_list))

def link_edges_to_dists_vars(edge_list, dists):
    edge_list_dists = []
    for p, c in edge_list:
        edge_list_dists.append(
            (dists[p], dists[c])
        )
    return edge_list_dists

def flatten_dists(dists):
    flattened = []
    flat_names = []
    for name, dist in dists.items():
        flattened.append(dist)
        flat_names.append(name)
    return flattened, flat_names

def make_bn(dists, edges):
    edge_list = extract_edge_list(edges)
    edge_list_dists = link_edges_to_dists_vars(edge_list, dists)
    flattened, flat_names = flatten_dists(dists)
    bn = BayesianNetwork(flattened, edge_list_dists)
    return bn, flat_names

def make_random_detached_variables(num_nodes, start_idx=140):
    i = start_idx
    dists = {}
    for _ in range(num_nodes):
        name = f'X{i}'
        # Choose a random number of states or 3 or 4
        rand_num_states = random.choice([3,4])
        probs = []
        up_limit = 1
        for j in range(rand_num_states):
            # Ensure complete distribution
            if j == rand_num_states - 1:
                probs.append(up_limit)
                continue
            prob = random.uniform(0, up_limit)
            up_limit -= prob
            probs.append(prob)
        probs = np.expand_dims(np.array(probs), 0)
        dists[name] = Categorical(probs)
        i += 1
    return dists

def make_TIE(dists_info, edges, nodes_to_add_10_children=None, isTIE1000=True):
    dists = {}
    for parent_name, info in dists_info.items():
        if nodes_to_add_10_children and parent_name == "Z|X":
            _dists, _edges = process_additional_dists(info, nodes_to_add_10_children)
            dists.update(_dists)
            edges = merge_edges(edges, _edges)
            continue
        try:
            dists[parent_name] = process_conditional(info)
        except AttributeError:
            # Process if this is not a conditional distribution
            dists[parent_name] = process_categorical(info)
    if isTIE1000:
        # Add in 860 detached variables
        dists.update(make_random_detached_variables(860, 140))
    return make_bn(dists, edges)

def generate_tie1000(
        path='../data/TIE1000.json',
        num_samples=1000
        ):
    # Load the TIE 1000 information file
    with open(path, 'r') as tie_jsonfile:
        tie_info = json.load(tie_jsonfile)

    # Create TIE 1000 network with pomegranate
    tie, names = make_TIE(tie_info["dists"], tie_info["edges"], tie_info["add_10_children"])

    # Generate samples
    samples = tie.sample(num_samples).type(torch.float)
    arr = samples.detach().numpy().astype(np.int16)
    
    return arr, names


def generate_tie_small(
        path='../data/TIE-SMALL.json',
        num_samples=1000
        ):
    # Load the TIE small information file
    with open(path, 'r') as tie_jsonfile:
        tie_info = json.load(tie_jsonfile)

    # Create TIE small network with pomegranate
    tie, names = make_TIE(tie_info["dists"], tie_info["edges"], isTIE1000=False)

    # Generate samples
    samples = tie.sample(num_samples).type(torch.float)
    arr = samples.detach().numpy().astype(np.int16)

    return arr, names


if __name__ == '__main__':
    tie1000, names1000 = generate_tie1000(num_samples=3750)
    tie_small, names_small = generate_tie_small()
    # Save the generated datasets
    np.save('../data/tie1000.npy', tie1000)
    np.save('../data/tie_small.npy', tie_small)
    with open('../data/tie1000_names.json', 'w') as f:
        json.dump(names1000, f)
    with open('../data/tie_small_names.json', 'w') as f:
        json.dump(names_small, f)
    print("Datasets saved")