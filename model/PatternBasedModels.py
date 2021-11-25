import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from torch.autograd import Function
from transformers import BertModel


class BiLSTM(nn.Module):
    def __init__(self, args):
        super(BiLSTM, self).__init__()

        self.args = args

        self.max_sequence_length = args.bilstm_input_max_sequence_length
        self.num_layers = args.bilstm_num_layer
        self.hidden_size = args.bilstm_hidden_dim

        self.lstm = nn.LSTM(args.bilstm_input_dim, self.hidden_size, self.num_layers,
                            bidirectional=True, batch_first=True, dropout=args.bilstm_dropout)
        self.fc = nn.Linear(self.hidden_size * 2,
                            args.output_dim_of_pattern_based_model)

    def forward(self, idxs, dataset, tokens_features, maps=None):
        # tokens_features: (batch_size, max_nodes, 768)
        # masks: (batch_size, max_nodes)
        # maps: (batch_size, max_nodes) or None

        tokens_features, masks = tokens_features
        # (batch_size, max_nodes, hidden_size*2)
        bilstm_out, _ = self.lstm(tokens_features)

        if maps is None:
            # (batch_size, hidden_size*2)
            attention_out = torch.sum(masks[:, :, None] * bilstm_out, dim=1)
        else:
            # (batch_size, hidden_size*2)
            attention_out = torch.sum(maps[:, :, None] * bilstm_out, dim=1)

        out = self.fc(attention_out)
        return out


class BERT_Emo(nn.Module):
    def __init__(self, args) -> None:
        super(BERT_Emo, self).__init__()

        self.args = args

        if args.dataset in DATASETS_CHINESE:
            self.bert = BertModel.from_pretrained(
                args.bert_pretrained_model_chinese, return_dict=False)
        else:
            self.bert = BertModel.from_pretrained(
                args.bert_pretrained_model_english, return_dict=False)

        for name, param in self.bert.named_parameters():
            # finetune the pooler layer
            if name.startswith("pooler"):
                if 'bias' in name:
                    param.data.zero_()
                elif 'weight' in name:
                    param.data.normal_(
                        mean=0.0, std=self.bert.config.initializer_range)
                param.requires_grad = True

            # finetune the last encoder layer
            elif name.startswith('encoder.layer.11'):
                param.requires_grad = True

            # the embedding layer
            elif name.startswith('embeddings'):
                param.requires_grad = args.bert_training_embedding_layers

            # the other transformer layers (intermediate layers)
            else:
                param.requires_grad = args.bert_training_inter_layers

        fixed_layers = []
        for name, param in self.bert.named_parameters():
            if not param.requires_grad:
                fixed_layers.append(name)

        print('\n', '*'*15, '\n')
        print("BERT_Emo Fixed layers: {} / {}: \n{}".format(
            len(fixed_layers), len(self.bert.state_dict()), fixed_layers))
        print('\n', '*'*15, '\n')

        # reserve for "words" -> "tokens"
        self.maxlen = min(int(args.bert_input_max_sequence_length * 1.5), 512)
        self.doc_maxlen = self.maxlen - 2

        # other layers
        self.fc = nn.Linear(args.bert_hidden_dim + args.bert_emotion_dim,
                            args.output_dim_of_pattern_based_model)

    def forward(self, idxs, dataset, tokens_features, maps=None):
        # nodes_tokens:
        #   each item, just like: [[2544, 1300], [2571, 45, 300], ...]
        # maps:
        #   each item, just like: torch.tensor([0.02, 0.15, ...])

        # t = time.time()
        nodes_tokens = [dataset.graphs[idx.item()]['graph'] for idx in idxs]
        nodes_tokens = [[n[-1] for n in nodes[:MAX_TOKENS_OF_A_POST]]
                        for nodes in nodes_tokens]
        # print('Loading tokens from graph.json, it took {:.2f}s'.format(
        #     time.time()-t))

        t = time.time()
        if maps is not None:
            tokened_maps = []
            for i, nodes in enumerate(nodes_tokens):
                m = maps[i]
                assert len(nodes) == len(m)

                # [[0.01, 0.01], [0.05, 0.05, 0.05], ...]
                tokened_m = [[m[nidx]/len(node) for _ in node]
                             for nidx, node in enumerate(nodes)]
                # [0.01, 0.01, 0.05, 0.05, 0.05, ...]
                tokened_m = [a for b in tokened_m for a in b]

                tokened_maps.append(torch.tensor(
                    tokened_m, device=self.args.device))

            # print('Transfer word-level map to token-level map, it took {:.2f}s'.format(
            #     time.time()-t))
            # t = time.time()
        else:
            tokened_maps = None

        # Each item, transfer to: [2544, 1300, 2571, 45, 300, ...]
        nodes_tokens = [[a for b in nodes for a in b]
                        for nodes in nodes_tokens]

        # print('nodes_tokens: ', nodes_tokens[0])
        # print('maps: ', maps)
        # print('tokened_maps: ', tokened_maps)

        t = time.time()
        out = self.forward_BERT(idxs, dataset, nodes_tokens, maps=tokened_maps)
        # print('In forward_BERT(), it took {:.2f}'.format(time.time() - t))
        return out

    def forward_BERT(self, idxs, dataset, tokens, maps=None):
        inputs = [self._encode(t) for t in tokens]
        # (batch_size, max_length)
        input_ids = torch.tensor(
            [i[0] for i in inputs], dtype=torch.long, device=self.args.device)
        # (batch_size, max_length, 1)
        masks = torch.stack([i[1] for i in inputs])
        # print('masks: ', masks.shape, masks)

        # (batch_size, max_length, 768)
        seq_output, _ = self.bert(input_ids)

        if maps is None:
            # semantic_output = torch.mean(seq_output, dim=1)
            semantic_output = torch.sum(masks*seq_output, dim=1)
        else:
            # (batch_size, max_sequence_length, 1)
            maps_attention = self._padding([m[:, None] for m in maps])
            semantic_output = torch.sum(maps_attention * seq_output, dim=1)

        # emotion features
        if self.args.bert_emotion_dim > 0:
            # (batch_size, emo_dim)
            emotion_output = dataset.BERT_Emo_features[idxs]
            emotion_output = torch.tensor(
                emotion_output, dtype=torch.float, device=self.args.device)
            # (batch_size, 768 + emo_dim)
            output = torch.cat([semantic_output, emotion_output], dim=1)
        else:
            # (batch_size, 768)
            output = semantic_output

        # out = self.fc(output)
        out = F.gelu(self.fc(output))
        return out

    def _encode(self, doc):
        doc = doc[:self.doc_maxlen]

        padding_length = self.maxlen - (len(doc) + 2)
        input_ids = [101] + doc + [102] + [103] * padding_length

        mask = torch.zeros(self.maxlen, 1, dtype=torch.float,
                           device=self.args.device)
        mask[:-padding_length] = 1 / (len(doc) + 2)

        return input_ids, mask

    def _padding(self, t):
        # t:
        #   type is list, the size is batch_size. Each elem is a (num_tokens, dim) tensor.
        # Return:
        #   a (batch_size, max_sequence_length, dim) tensor

        dim = t[0].shape[-1]
        padded_t = torch.zeros(
            (len(t), self.maxlen, dim), device=self.args.device)
        for i, x in enumerate(t):
            sz = min(len(x), self.maxlen)
            padded_t[i, :sz] = x[:sz]

        return padded_t


class EANN_Text(nn.Module):
    ''' Learning Hierarchical Discourse-level Structures for Fake News Detection. NAACL 2019.'''
    """
        From https://github.com/yaqingwang/EANN-KDD18/blob/master/src/EANN_text.py
    """

    def __init__(self, args):
        super(EANN_Text, self).__init__()
        self.args = args

        self.input_dim = args.eann_input_dim  # 768 if using BERT embedding
        self.hidden_dim = args.eann_hidden_dim  # 64
        self.event_num = args.eann_event_num  # depends on #clusters
        self.max_sequence_length = args.eann_input_max_sequence_length

        self.use_textcnn = args.eann_use_textcnn  # otherwise use bilstm

        # TextCNN
        channel_in = 1
        filter_num = 20
        window_sizes = [1, 2, 3, 4]
        self.convs = nn.ModuleList(
            [nn.Conv2d(channel_in, filter_num, (K, self.input_dim)) for K in window_sizes])
        self.fc_cnn = nn.Linear(
            filter_num * len(window_sizes), self.hidden_dim)

        self.event_discriminator = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(True),
            nn.Linear(self.hidden_dim, self.event_num),
            nn.Softmax(dim=1)
        )

        self.fake_news_detector = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(True),
            nn.Linear(self.hidden_dim, args.output_dim_of_pattern_based_model),
            # nn.Softmax(dim=1)
        )

    def forward_EANN(self, text, mask, maps=None):
        # text -> [batch_size, text_len, embedding_size]
        # mask -> [batch_size, text_len, 1] (If using pref map, the mask will be in [0,1] and the padding tokens are 0, else the entries are 0/1)
        # Textual Feature Extraction

        if maps is None:
            text = text * mask
        else:
            # maps: (batch_size, text_len, 1)
            text = text * maps

        # [batch_size, 1, text_len, embedding_size]
        text = text.unsqueeze(1)
        # [batch_size, filter_num, feature_len] * len(window_sizes)
        text = [F.relu(conv(text)).squeeze(3) for conv in self.convs]
        # [batch_size, filter_num] * len(window_sizes)
        text = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in text]
        # [batch_size, filter_num * len(window_sizes)]
        text = torch.cat(text, 1)
        text = F.relu(self.fc_cnn(text))  # [batch_size, hidden_dim]

        # Fake News Detection: (batch_size, output_dim)
        detector_output = self.fake_news_detector(text)

        # Event Discrimination: (batch_size, 300)
        reverse_text_feature = grad_reverse(text)
        discriminator_output = self.event_discriminator(reverse_text_feature)

        return detector_output, discriminator_output

    def forward(self, idxs, dataset, tokens_features, maps=None):
        # tokens_features:
        #   type is list, the size is batch_size. Each elem is a (num_tokens, input_dim) tensor.
        # maps:
        #   type is list (or None), the size is batch_size. Each elem is a (num_tokens) tensor.

        tokens_features, masks = self._padding(tokens_features)
        if maps is not None:
            maps, _ = self._padding([m[:, None] for m in maps])

        return self.forward_EANN(text=tokens_features, mask=masks, maps=maps)

    def _padding(self, t):
        # t:
        #   type is list, the size is batch_size. Each elem is a (num_tokens, dim) tensor.
        # Return:
        #   a (batch_size, max_sequence_length, dim) tensor

        dim = t[0].shape[-1]
        padded_t = torch.zeros(
            (len(t), self.max_sequence_length, dim), device=self.args.device)
        mask = torch.zeros(
            (len(t), self.max_sequence_length, 1), device=self.args.device)
        for i, x in enumerate(t):
            sz = min(len(x), self.max_sequence_length)
            padded_t[i, :sz] = x[:sz]
            mask[i, :sz] = 1 / (sz + ZERO)

        return padded_t, mask


class GRL(Function):
    '''Gradient Reversal Layer'''
    """
        Refer to https://blog.csdn.net/t20134297/article/details/107870906
    """
    @staticmethod
    def forward(ctx, x, lamd):
        ctx.lamd = lamd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.lamd
        return grad_output, None


def grad_reverse(x, lamd=1):
    return GRL.apply(x, lamd)
