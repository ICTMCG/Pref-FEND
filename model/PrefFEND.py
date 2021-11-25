import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from HetDGCN import HetDGCN
from PatternBasedModels import BiLSTM, BERT_Emo, EANN_Text
from FactBasedModels import DeClarE, MAC, EVIN

from config import MAX_TOKENS_OF_A_POST
from utils import ZERO


class PrefFEND(nn.Module):
    def __init__(self, args):
        super(PrefFEND, self).__init__()
        self.args = args

        # Fact-based Models
        if args.use_fact_based_model:
            self.FactBasedModel = eval(
                '{}(args)'.format(args.fact_based_model))
        else:
            args.output_dim_of_fact_based_model = 0

        # Pattern-based Models
        if args.use_pattern_based_model:
            self.PatternBasedModel = eval(
                '{}(args)'.format(args.pattern_based_model))
        else:
            args.output_dim_of_pattern_based_model = 0

        # MLP layers
        last_output = args.output_dim_of_pattern_based_model + \
            args.output_dim_of_fact_based_model
        self.fcs = nn.ModuleList(self.init_MLP_layers(last_output))

        # HetDGCN
        if args.use_preference_map:
            self.HetDGCN = HetDGCN(args)
            self.reversed_fcs = nn.ModuleList(
                self.init_MLP_layers(last_output))

    def init_MLP_layers(self, last_output):
        fcs = []
        for _ in range(self.args.num_mlp_layers - 1):
            curr_output = int(last_output / 2)
            fcs.append(nn.Linear(last_output, curr_output))
            last_output = curr_output

        fcs.append(nn.Linear(last_output, self.args.category_num))
        return fcs

    def forward_PreferencedDetector(self, idxs, dataset, tokens_features, map_for_fact_detector, map_for_pattern_detector, fcs):
        fact_model_out, pattern_model_out = None, None

        if self.args.use_fact_based_model:
            fact_model_out = self.FactBasedModel(
                idxs, dataset, tokens_features, map_for_fact_detector)
        if self.args.use_pattern_based_model:
            pattern_model_out = self.PatternBasedModel(
                idxs, dataset, tokens_features, map_for_pattern_detector)

            # EANN_Text
            if self.args.pattern_based_model == 'EANN_Text':
                pattern_model_out, event_out = pattern_model_out

        # (batch_size, output_dim_fact + output_dim_pattern)
        models_out = torch.cat(
            [x for x in [fact_model_out, pattern_model_out] if x is not None], dim=1)

        # (batch_size, category_num)
        for fc in fcs:
            models_out = F.gelu(fc(models_out))

        # EANN_Text
        if self.args.pattern_based_model == 'EANN_Text':
            return models_out, event_out
        return models_out

    def forward(self, idxs, dataset, graphs_entity, graphs_pattern, graphs_others, nums_nodes):
        # ========= Fetch tokens_features from graphs =========
        nodes_features = []
        curr = 0
        for num in nums_nodes:
            nodes_features.append(graphs_entity.x[curr:curr+num])
            curr += num
        tokens_features = self.padding(nodes_features)

        # ========= Get Preferenced Maps =========
        if self.args.use_preference_map:
            type2nidxs = [dataset.type2nidxs[idx.item()] for idx in idxs]
            map_entity, map_pattern = self.HetDGCN(
                graphs_entity, graphs_pattern, graphs_others, nums_nodes, type2nidxs)
        else:
            map_entity, map_pattern = None, None

        # ========= Normal Forward =========
        model_out = self.forward_PreferencedDetector(
            idxs, dataset, tokens_features, map_for_fact_detector=map_entity, map_for_pattern_detector=map_pattern, fcs=self.fcs)
        # EANN_Text
        if self.args.pattern_based_model == 'EANN_Text':
            model_out, event_out = model_out

        # ========= Reverse Predictions =========
        if self.args.use_preference_map:
            model_reversed_out = self.forward_PreferencedDetector(
                idxs, dataset, tokens_features, map_for_fact_detector=map_pattern, map_for_pattern_detector=map_entity, fcs=self.reversed_fcs)

            # EANN_Text
            if self.args.pattern_based_model == 'EANN_Text':
                model_reversed_out, event_reversed_out = model_reversed_out
        else:
            model_reversed_out, event_reversed_out = None, None

        if self.args.pattern_based_model == 'EANN_Text':
            return model_out, model_reversed_out, map_entity, map_pattern, event_out, event_reversed_out

        return model_out, model_reversed_out, map_entity, map_pattern

    def padding(self, nodes_features):
        # nodes_features:
        #   type is list, the size is batch_size. Each elem is a (num_nodes, 768) tensor.
        # Return:
        #   padding_nodes_features: (batch_size, max_nodes, 768)
        #   mask: (batch_size, max_nodes)

        dim = nodes_features[0].shape[-1]
        padding_nodes_features = torch.zeros(
            (len(nodes_features), MAX_TOKENS_OF_A_POST, dim), device=self.args.device)
        mask = torch.zeros(
            (len(nodes_features), MAX_TOKENS_OF_A_POST), device=self.args.device)
        for i, x in enumerate(nodes_features):
            sz = min(len(x), MAX_TOKENS_OF_A_POST)
            padding_nodes_features[i, :sz] = x[:sz]
            mask[i, :sz] = 1 / (sz + ZERO)

        return padding_nodes_features, mask
