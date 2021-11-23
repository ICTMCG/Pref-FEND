from argparse import ArgumentParser
import json
from tqdm import tqdm
import torch
import pickle
import os
import numpy as np


def pytorch_cos_sim(a: torch.Tensor, b: torch.Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    This function can be used as a faster replacement for 1-scipy.spatial.distance.cdist(a,b)
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def normalized_correlation_matrix(nodes):
    # Nodes: (sz, 768)
    A = pytorch_cos_sim(nodes, nodes)
    A = (A + 1) / 2

    D = torch.diag((torch.sum(A, dim=1) + len(nodes)) ** (-0.5))
    D.masked_fill_(D == float('inf'), 0.)
    A = D @ A @ D

    return A


def get_bert_embedding(tokens):
    # tokens: a list, such as [1,2,3,4]
    # return: (num_tokens, 768)
    return bert_token_embeddings(torch.tensor(tokens)).detach().numpy()


def parse_graph(g):
    graph = g['graph'][:max_nodes]
    sz = len(graph)
    type2nidxs = dict()

    # === Nodes ===
    nodes = []
    for i, (t, _, tokens) in enumerate(graph):
        if t not in type2nidxs:
            type2nidxs[t] = [i]
        else:
            type2nidxs[t].append(i)

        feat = np.mean(get_bert_embedding(tokens), axis=0)

        # list, every item's shape is (768,)
        nodes.append(feat)

    # tensor, shape is (sz, 768)
    nodes = torch.tensor(nodes, dtype=torch.float)

    # === Adjacency Matrix ===
    # the shape is (sz, sz)
    A = normalized_correlation_matrix(nodes)

    # === Edges ===
    def _get_edges(node_type):
        edge_index = []
        edge_attr = []

        # self-loop
        edge_index += [(nidx, nidx) for nidx in range(sz)]
        edge_attr += [1.0 for _ in range(sz)]

        # bi-directed connections
        if node_type in type2nidxs:
            for selected_nidx in type2nidxs[node_type]:
                edge_index += [(nidx, selected_nidx)
                               for nidx in range(sz) if nidx != selected_nidx]
                edge_attr += [A[nidx][selected_nidx]
                              for nidx in range(sz) if nidx != selected_nidx]

                edge_index += [(selected_nidx, nidx)
                               for nidx in range(sz) if nidx != selected_nidx]
                edge_attr += [A[selected_nidx][nidx]
                              for nidx in range(sz) if nidx != selected_nidx]

        # (2, num_edges)
        edge_index = torch.tensor(
            [[e[0] for e in edge_index], [e[1] for e in edge_index]], dtype=torch.long)
        # (num_edges, 1)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        return {'index': edge_index, 'weight': edge_attr}

    edges2entity = _get_edges('ENTITY')
    edges2pattern = _get_edges('PATTERN')
    edges2others = _get_edges('OTHERS')

    return nodes, edges2entity, edges2pattern, edges2others, type2nidxs


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--max_nodes', type=int, default=100)
    args = parser.parse_args()

    dataset = args.dataset
    max_nodes = args.max_nodes

    save_dir = 'data'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, dataset)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    bert_token_embeddings = torch.load('../tokenize/data/{}/BertTokenEmbedding.pt'.format(dataset))
    print('Bert Embeddings: \n{}\n'.format(bert_token_embeddings))

    for dataset_type in ['train', 'val', 'test']:
        graph_file = '../tokenize/data/{}/post/graph_{}.json'.format(
            dataset, dataset_type)
        with open(graph_file, 'r') as f:
            graphs = json.load(f)
        print('[Dataset File]\t{}, sz = {}'.format(
            graph_file, len(graphs)))

        # ======== Init Graph Data ========
        nodes_features = dict()
        edges2entity = dict()
        edges2pattern = dict()
        edges2others = dict()
        type2nidxs = dict()

        for idx, g in enumerate(tqdm(graphs)):
            nodes, edges2entity, edges2pattern, edges2others, type2nidxs = parse_graph(
                g)
            nodes_features[idx] = nodes
            edges2entity[idx] = edges2entity
            edges2pattern[idx] = edges2pattern
            edges2others[idx] = edges2others
            type2nidxs[idx] = type2nidxs

        graph_data_file = 'data/{}/graph_max_nodes_{}_{}.pkl'.format(
            dataset, max_nodes, dataset_type)
        with open(graph_data_file, 'wb') as f:
            pickle.dump([nodes_features, edges2entity,
                         edges2pattern, edges2others, type2nidxs], f)
