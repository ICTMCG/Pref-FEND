import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import numpy as np
from tqdm import tqdm
import time
import json
import os
import pickle
from config import INDEX_OF_LABEL, MAX_TOKENS_OF_A_POST, MAX_TOKENS_OF_A_DOC, MAX_RELEVANT_ARTICLES


class GraphDataset(Dataset):
    def __init__(self, args, dataset_type):
        self.args = args
        experimental_dataset = args.dataset

        # === Fake News Detector ===
        graph_file = '../preprocess/tokenize/data/{}/post/graph_{}.json'.format(
            experimental_dataset, dataset_type)
        with open(graph_file, 'r') as f:
            self.graphs = json.load(f)
        print('[Dataset File]\t{}, sz = {}'.format(
            graph_file, len(self.graphs)))

        self.labels = [p['label'] for p in self.graphs]
        self.labels = torch.tensor([INDEX_OF_LABEL[l] for l in self.labels])

        # === HetDGCN ===
        t = time.time()
        print('\nLoading Graph data...')
        graph_data_file = '../preprocess/graph_init/data/{}/graph_max_nodes_{}_{}.pkl'.format(
            experimental_dataset, MAX_TOKENS_OF_A_POST, dataset_type)
        self.init_graph_data(graph_data_file)
        print('Done, it took {:.2}s\n'.format(time.time() - t))

        # === Fact-based Model ===
        if args.use_fact_based_model:
            t = time.time()
            print('\nLoading Relevant Articles data...')
            articles_tokens_file = '../preprocess/tokenize/data/{}/articles/articles_tokens.pkl'.format(
                experimental_dataset)
            bm25_results_file = '../preprocess/bm25/data/{}/top10_articles_{}.npy'.format(
                experimental_dataset, dataset_type)
            self.init_articles_data(articles_tokens_file, bm25_results_file)
            print('Done, it took {:.2}s\n'.format(time.time() - t))

        # === TODO: BERT_Emo ===
        # if args.pattern_based_model == 'BERT_Emo' and args.bert_emotion_dim > 0:
        #     t = time.time()
        #     print('\nLoading Emotion data...')
        #     emotion_features_file = os.path.join(
        #         PROJECT_DIR, 'Baselines/BERT_Emo/code/preprocess/data', experimental_dataset, 'emotions/{}.npy'.format(dataset_type))
        #     self.BERT_Emo_features = np.load(emotion_features_file)
        #     print('Done, the shape is {}, it took {:.2}s\n'.format(
        #         self.BERT_Emo_features.shape, time.time() - t))

        # === TODO: EANN_Text ===
        # if args.pattern_based_model == 'EANN_Text':
        #     t = time.time()
        #     print('\nLoading Event data...')
        #     event_label_file = os.path.join(
        #         PROJECT_DIR, 'Baselines/EANN_Text/data', experimental_dataset, '{}_event_labels.npy'.format(dataset_type))
        #     self.event_labels = np.load(event_label_file).tolist()
        #     print('Done, the size is {}, it took {:.2}s\n'.format(
        #         len(self.event_labels), time.time() - t))

    def init_graph_data(self, graph_data_file):
        with open(graph_data_file, 'rb') as f:
            self.nodes_features, self.edges2entity, self.edges2pattern, self.edges2others, self.type2nidxs = pickle.load(
                f)

    def init_articles_data(self, articles_tokens_file, bm25_results_file):
        with open(articles_tokens_file, 'rb') as f:
            self.articles_tokens = pickle.load(f)
            self.articles_tokens = [t[:MAX_TOKENS_OF_A_DOC]
                                    for t in self.articles_tokens]
        print('There are {} articles, MAX_TOKENS_OF_A_DOC = {}.'.format(
            len(self.articles_tokens), MAX_TOKENS_OF_A_DOC))

        # (dataset_size, MAX_RELEVANT_ARTICLES)
        self.top_articles_idxs = np.load(bm25_results_file)[
            :, :MAX_RELEVANT_ARTICLES]
        print('Top {} resutls arr: {}'.format(
            MAX_RELEVANT_ARTICLES, self.top_articles_idxs.shape))

        bert_token_embeddings = torch.load(
            '../preprocess/tokenize/data/{}/BertTokenEmbedding.pt'.format(self.args.dataset))
        print('\nBert Token Embedding Layer: {}\n'.format(bert_token_embeddings))

        def _get_bert_embedding(tokens):
            # tokens: a list, such as [1,2,3,4]
            # return: (num_tokens, 768)
            return bert_token_embeddings(torch.tensor(tokens)).detach().numpy()

        # self.articles_features: key = article's index, value = (num_tokens, 768)
        self.articles_features = dict()
        for idx in set([x for y in self.top_articles_idxs.tolist() for x in y]):
            self.articles_features[idx] = torch.tensor(
                _get_bert_embedding(self.articles_tokens[idx]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        nodes = self.nodes_features[idx]
        edges2entity = self.edges2entity[idx]
        edges2pattern = self.edges2pattern[idx]
        edges2others = self.edges2others[idx]

        sample = (
            idx,
            self.get_graph_data(nodes, edges2entity, 'ENTITY'),
            self.get_graph_data(nodes, edges2pattern, 'PATTERN'),
            self.get_graph_data(nodes, edges2others, 'OTHERS'),
            # num_of_nodes
            len(nodes),
            self.labels[idx]
        )
        return sample

    def get_graph_data(self, nodes, edges, node_type):
        return Data(x=nodes, edge_index=edges['index'], edge_attr=edges['weight'], y=node_type)
