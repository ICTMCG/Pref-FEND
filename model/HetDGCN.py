import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.nn import GCNConv, inits

from utils import ZERO, normalized_correlation_matrix
from config import MAX_TOKENS_OF_A_POST


class HetDGCN(nn.Module):
    def __init__(self, args):
        super(HetDGCN, self).__init__()
        self.args = args

        self.gnn_layers = []
        self.gnn_dynamic_update_weights = []
        for _ in range(args.num_gnn_layers):
            entity_conv = GCNConv(args.dim_node_features,
                                  args.dim_node_features, add_self_loops=False, normalize=False)
            pattern_conv = GCNConv(
                args.dim_node_features, args.dim_node_features, add_self_loops=False, normalize=False)
            others_conv = GCNConv(args.dim_node_features,
                                  args.dim_node_features, add_self_loops=False, normalize=False)
            self.gnn_layers.append(nn.ModuleDict(
                {'ENTITY': entity_conv, 'PATTERN': pattern_conv, 'OTHERS': others_conv}))

            t = nn.Parameter(torch.Tensor(
                args.dim_node_features, args.dim_node_features))
            inits.glorot(t)

            self.gnn_dynamic_update_weights.append(t)

        self.gnn_layers = nn.ModuleList(self.gnn_layers)
        self.gnn_dynamic_update_weights = nn.ParameterList(
            self.gnn_dynamic_update_weights)

    def forward_GCN(self, GCN, x, graphs, A, layer_num):
        if layer_num == 0:
            edge_index, edge_weight = graphs.edge_index, graphs.edge_attr
        else:
            # --- Update edge_weights in graphs_* ---
            try:
                # (2, E)
                edge_index = graphs.edge_index
                E, N = len(graphs.edge_attr), len(A)
                # (E, N)
                start = F.one_hot(edge_index[0], num_classes=N)
                # (N, E)
                end = F.one_hot(edge_index[1], num_classes=N).t()

                # (E)
                edge_weight = torch.diag(start.float() @ A @ end.float())
                del start, end

            except:
                print('\n[Out of Memory] There are too much edges in this batch (num = {}), so it executes as a for-loop for this batch.\n'.format(len(graphs.edge_attr)))
                # (2, E)
                edge_index = graphs.edge_index
                edges_num = len(graphs.edge_attr)
                edge_weight = torch.zeros(
                    edges_num, device=self.args.device, dtype=torch.float)

                for e in tqdm(range(edges_num)):
                    a, b = graphs.edge_index[:, e]
                    edge_weight[e] = A[a, b]

        # （num_nodes_of_batch, 768)
        out = GCN(x=x, edge_index=edge_index, edge_weight=edge_weight)
        return out

    def forward(self, graphs_entity, graphs_pattern, graphs_others, nums_nodes, type2nidxs):
        # graphs_*:
        #   torch_geometric.data.batch.Batch. It is loaded by PyG's DataLoader, so it has "batch" attribute.
        # nums_nodes:
        #   tensor. The size is batch_size. Each elem is the num of nodes in the "sample".
        # type2nidxs:
        #   list. The size is batch_size. Each elem is a dict, eg: {'ENTITY': [6], 'PATTERN':[9, 10], 'OTHERS': [1, 2, ...]}.

        # --- Tensorize ---
        graphs_entity.to(self.args.device)
        graphs_pattern.to(self.args.device)
        graphs_others.to(self.args.device)

        # "x" of entity, pattern, others are same
        H = torch.clone(graphs_entity.x)
        A = normalized_correlation_matrix(H)

        for i, gnn in enumerate(self.gnn_layers):
            H_entity = self.forward_GCN(
                gnn['ENTITY'], x=H, graphs=graphs_entity, A=A, layer_num=i)
            H_pattern = self.forward_GCN(
                gnn['PATTERN'], x=H, graphs=graphs_pattern, A=A, layer_num=i)
            H_others = self.forward_GCN(
                gnn['OTHERS'], x=H, graphs=graphs_others, A=A, layer_num=i)

            # (num_nodes_in_batches, 768)
            H = F.relu(H_entity + H_pattern + H_others)

            # --- Update adjacency_matrix ---
            A_hat = torch.sigmoid(
                H @ self.gnn_dynamic_update_weights[i] @ H.t())
            A = (1 - self.args.updated_weights_for_A) * \
                A + self.args.updated_weights_for_A * A_hat

        # --- Preference Maps Readout ---
        map_entity = []
        map_pattern = []

        curr = 0
        for j, num in enumerate(nums_nodes):
            entity_nodes_idxs = []
            pattern_nodes_idxs = []

            if 'ENTITY' in type2nidxs[j]:
                entity_nodes_idxs = type2nidxs[j]['ENTITY']
            if 'PATTERN' in type2nidxs[j]:
                pattern_nodes_idxs = type2nidxs[j]['PATTERN']

            # (num, num)
            curr_A = A[curr:curr+num, curr:curr+num]

            if torch.any(torch.isnan(curr_A)):
                print()
                print('curr_A: ', curr_A)
                # exit()

            # (num)
            A_sum = torch.sum(curr_A, dim=1)

            if len(entity_nodes_idxs) > 0:
                # (num, num_of_entity_nodes)
                A_entity = curr_A[:, torch.tensor(list(entity_nodes_idxs))]
                map_pattern.append(A_sum - torch.sum(A_entity, dim=1))
            else:
                map_pattern.append(A_sum)

            if len(pattern_nodes_idxs) > 0:
                # (num, num_of_pattern_nodes)
                A_pattern = curr_A[:, torch.tensor(list(pattern_nodes_idxs))]
                map_entity.append(A_sum - torch.sum(A_pattern, dim=1))
            else:
                map_entity.append(A_sum)

            curr += num

        def _scale(t):
            if len(t) == 0:
                return t
            m, M = min(t), max(t)
            return (t - m) / (M - m + ZERO)

        # Scale
        map_entity = [_scale(m) for m in map_entity]
        map_pattern = [_scale(m) for m in map_pattern]

        # Normalize
        map_entity = [m/(torch.sum(m) + ZERO) for m in map_entity]
        map_pattern = [m/(torch.sum(m) + ZERO) for m in map_pattern]

        # Padding to (batch_size, max_nodes)
        map_entity = self.padding(map_entity)
        map_pattern = self.padding(map_pattern)

        return map_entity, map_pattern

    def padding(self, map):
        # map:
        #   list. The size is batch_size. Each elem is a (nodes_num) tensor
        padding_map = torch.zeros(
            len(map), MAX_TOKENS_OF_A_POST, device=self.args.device, dtype=torch.float)
        for i, m in enumerate(map):
            sz = min(MAX_TOKENS_OF_A_POST, len(m))
            padding_map[i, :sz] = m[:sz]
        return padding_map
