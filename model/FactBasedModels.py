import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import ZERO
from config import MAX_RELEVANT_ARTICLES, MAX_TOKENS_OF_A_DOC


class BaseDetector(nn.Module):
    """
    The Base Class for the fact-based detectors
    """

    def __init__(self, args):
        super(BaseDetector, self).__init__()
        self.args = args

    def init_fact_based_detector_input(self, idxs, dataset, tokens_features):
        # ========= Get Post Features & Maps =========

        # queries_features: (batch_size, max_nodes, 768)
        # queries_mask: (batch_size, max_nodes, 1)
        queries_features, queries_mask = tokens_features

        # ========= Get Top Articles Features =========

        # (batch_size, MAX_RELEVANT_ARTICLES)
        top_articles_idxs = dataset.top_articles_idxs[idxs]

        # (batch_size, MAX_RELEVANT_ARTICLES, MAX_TOKENS_OF_A_DOC, 768)
        articles_features = torch.zeros(
            top_articles_idxs.shape[0], top_articles_idxs.shape[1], MAX_TOKENS_OF_A_DOC, queries_features.shape[-1])
        # (batch_size, MAX_RELEVANT_ARTICLES, MAX_TOKENS_OF_A_DOC)
        articles_mask = torch.zeros_like(articles_features[:, :, :, 0])

        for i, articles_idxs in enumerate(top_articles_idxs):
            for j, article_idx in enumerate(articles_idxs):
                # (num_tokens, 768)
                feat = dataset.articles_features[article_idx]
                sz = min(len(feat), MAX_TOKENS_OF_A_DOC)

                articles_features[i, j, :sz, :] = feat[:sz]
                articles_mask[i, j, :sz] = 1 / (sz + ZERO)

        # Use all relevant articles: (batch_size, MAX_RELEVANT_ARTICLES)
        articles_num_mask = torch.ones_like(
            articles_mask[:, :, 0]) / articles_mask.shape[1]

        articles_features = articles_features.to(self.args.device)
        articles_mask = articles_mask.to(self.args.device)
        articles_num_mask = articles_num_mask.to(self.args.device)

        return queries_features, queries_mask, articles_features, articles_mask, articles_num_mask


class MAC(nn.Module):
    '''Hierarchical Multi-head Attentive Network for Evidence-aware Fake News Detection. EACL 2021.'''
    """
        Refer to https://github.com/nguyenvo09/EACL2021/blob/9d04d8954c1ded2110daac23117de11221f08cc6/Models/FCWithEvidences/hierachical_multihead_attention.py
    """

    def __init__(self, args):
        super(MAC, self).__init__()
        self.args = args

        self.max_sequence_length = args.mac_input_max_sequence_length
        self.input_dim = args.mac_input_dim  # 768
        self.hidden_size = args.mac_hidden_dim  # 300
        self.dropout_doc = args.mac_dropout_doc  # 0.2
        self.dropout_query = args.mac_dropout_query  # 0.2
        self.num_heads_1 = args.mac_nhead_1
        self.num_heads_2 = args.mac_nhead_2

        self.num_layers = 1

        self.doc_bilstm = nn.LSTM(self.input_dim, self.hidden_size, self.num_layers,
                                  bidirectional=True, batch_first=True, dropout=self.dropout_doc)
        self.query_bilstm = nn.LSTM(self.input_dim, self.hidden_size, self.num_layers,
                                    bidirectional=True, batch_first=True, dropout=self.dropout_query)

        self.W1 = nn.Linear(4 * self.hidden_size, 2 *
                            self.hidden_size, bias=False)
        self.W2 = nn.Linear(2 * self.hidden_size, self.num_heads_1, bias=False)
        self.W3 = nn.Linear((self.num_heads_1 + 1) * 2 *
                            self.hidden_size, 2 * self.hidden_size, bias=False)
        self.W4 = nn.Linear(2 * self.hidden_size, self.num_heads_2, bias=False)
        last_output_dim = 2 * self.hidden_size * \
            (1 + self.num_heads_1 * self.num_heads_2)
        self.W5 = nn.Linear(
            last_output_dim, args.output_dim_of_fact_based_model, bias=True)
        # self.W6 = nn.Linear(self.hidden_size, 1, bias=True)

        self.doc_features_dropout = nn.Dropout(0.5)

    def forward(self, idxs, dataset, tokens_features, maps=None):
        # tokens_features:
        #   type is list, the size is batch_size. Each elem is a (num_tokens, input_dim) tensor.
        # maps:
        #   type is list (or None), the size is batch_size. Each elem is a (num_tokens) tensor.

        # === Get Top Articles Features ===

        # (batch_size, MAX_RELEVANT_ARTICLES)
        top_articles_idxs = dataset.top_articles_idxs[idxs]
        # print('top_articles_idxs: ', top_articles_idxs)

        # (batch_size, MAX_RELEVANT_ARTICLES, MAX_TOKENS_OF_A_DOC, 768)
        articles_features = torch.zeros(
            top_articles_idxs.shape[0], top_articles_idxs.shape[1], self.args.mac_max_doc_length, self.input_dim)
        # (batch_size, MAX_RELEVANT_ARTICLES, MAX_TOKENS_OF_A_DOC)
        articles_mask = torch.zeros_like(articles_features[:, :, :, 0])

        for i, articles_idxs in enumerate(top_articles_idxs):
            for j, article_idx in enumerate(articles_idxs):
                # (num_tokens, 768)
                feat = dataset.articles_features[article_idx]
                sz = min(len(feat), self.args.mac_max_doc_length)

                articles_features[i, j, :sz, :] = feat[:sz]
                articles_mask[i, j, :sz] = 1

        # Use all relevant articles: (batch_size, MAX_RELEVANT_ARTICLES)
        articles_num_mask = torch.ones_like(articles_mask[:, :, 0])

        # === Get Post Features & Maps ===
        queries_features, queries_mask = self._padding(tokens_features)
        if maps is not None:
            maps, _ = self._padding([m[:, None] for m in maps])

        articles_features = articles_features.to(self.args.device)
        articles_mask = articles_mask.to(self.args.device)
        articles_num_mask = articles_num_mask.to(self.args.device)

        return self.forward_MAC(queries_features, queries_mask, articles_features, articles_mask, articles_num_mask, maps=maps)

    def _padding(self, t):
        # t:
        #   type is list, the size is batch_size. Each elem is a (num_tokens, dim) tensor.
        # Return:
        #   padded_t: a (batch_size, max_sequence_length, dim) tensor
        #   mask: a (batch_size, max_sequence_legnth) tensor

        dim = t[0].shape[-1]
        padded_t = torch.zeros(
            (len(t), self.max_sequence_length, dim), device=self.args.device)
        mask = torch.zeros_like(padded_t[:, :, 0])

        for i, x in enumerate(t):
            sz = min(len(x), self.max_sequence_length)
            padded_t[i, :sz] = x[:sz]
            mask[i, :sz] = 1
        return padded_t, mask

    def forward_MAC(self, query_feature, query_mask, doc_features, doc_mask, doc_num_mask, maps=None):
        # query_feature -> [batch_size, post_len(padded), embedding_size]
        # query_mask -> [batch_size, post_len(padded)]
        # doc_features -> [batch_size, #doc, doc_len(padded), embedding_size]
        # doc_mask -> [batch_size, #doc, doc_len(padded)]
        # doc_num_mask -> [batch_size, #doc]

        # [batch_size, post_len, hidden_size * 2]
        query_hiddens, _ = self.query_bilstm(query_feature)
        query_hiddens = query_hiddens.masked_fill(
            (query_mask[:, :, None] == 0), 0)

        if maps is None:
            # [batch_size, hidden_size * 2] Eq.(1)
            query_repr = torch.sum(query_hiddens, dim=1) / \
                torch.sum(query_mask+ZERO, dim=1, keepdim=True)
        else:
            query_repr = torch.sum(query_hiddens * maps, dim=1)

        # TimeDistributed(BiLSTM)
        df_sizes = doc_features.size()
        # [batch_size * #doc, doc_len, emb_dim]
        doc_hiddens = doc_features.view(-1, df_sizes[-2], df_sizes[-1])
        # [batch_size * #doc, doc_len, hidden_size * 2]
        doc_hiddens = self.doc_bilstm(doc_hiddens)[0]
        # [batch_size, #doc, doc_len, hidden_size * 2]
        doc_hiddens = doc_hiddens.view(
            df_sizes[0], df_sizes[1], df_sizes[2], doc_hiddens.size()[-1])

        doc_hiddens = self.doc_features_dropout(doc_hiddens)

        # Multi-head Word Attention Layer
        C1 = query_repr.unsqueeze(1).unsqueeze(1).repeat(
            1, doc_hiddens.shape[1], doc_hiddens.shape[2], 1)  # [batch_size, #doc, doc_len, hidden_size * 2]
        # [batch_size, #doc, doc_len, hidden_size*4]
        A1 = torch.cat((doc_hiddens, C1), dim=-1)
        # [batch_size, #doc, doc_len, head_num_1]
        A1 = self.W2(torch.tanh(self.W1(A1)))

        # exclude the padding words in each doc
        A1 = F.softmax(A1, dim=-1)  # [batch_size, #doc, doc_len, head_num_1]
        A1 = A1.masked_fill((doc_mask[:, :, :, None] == 0), 0)

        # [batch_size * #doc, doc_len, head_num_1]
        A1_tmp = A1.reshape(-1, A1.shape[-2], A1.shape[-1])
        # [batch_size * #doc, doc_len, hidden_size * 2]
        doc_hiddens_tmp = doc_hiddens.reshape(-1,
                                              doc_hiddens.shape[-2], doc_hiddens.shape[-1])
        # [batch_size*#doc, head_num_1, doc_len] * [batch_size*#doc, doc_len, hidden_size * 2]
        D = torch.bmm(A1_tmp.permute(0, 2, 1), doc_hiddens_tmp)
        # [batch_size, #doc, head_num_1 * hidden_size * 2]
        D = D.view(A1.shape[0], A1.shape[1], -1)

        # Multi-head Document Attention Layer
        # [batch_size, #doc, hidden_size * 2]
        C2 = query_repr.unsqueeze(1).repeat(1, D.shape[1], 1)
        # [batch_size, #doc, (head_num_1 + 1) * hidden_size * 2]
        A2 = torch.cat((D, C2), dim=-1)
        A2 = self.W4(torch.tanh(self.W3(A2)))  # [batch_size, #doc, head_num_2]

        # [batch_size, #doc, 1]
        A2 = F.softmax(A2, dim=-1)  # [batch_size, #doc, head_num_2]

        # [batch_size, #doc, head_num_2] * [batch_size, #doc, head_num_1 * hidden_size * 2]
        D = torch.bmm(A2.permute(0, 2, 1), D)
        # [batch_size, head_num_2 * head_num_1 * hidden_size * 2] Eq.(9)
        D = D.view(D.shape[0], -1)

        # Output Layer
        # [batch_size, (head_num_2 * head_num_1 * 2 + 2) * hidden_size]
        final_features = torch.cat((query_repr, D), dim=-1)
        # [batch_size, output_dim]
        final_features = self.W5(final_features)

        final_features = F.gelu(final_features)
        return final_features


class EVIN(nn.Module):
    '''Evidence Inference Networks for Interpretable Claim Verification. AAAI 2021.'''

    def __init__(self, args):
        super(EVIN, self).__init__()
        self.args = args

        self.max_sequence_length = args.evin_input_max_sequence_length
        self.dropout_att = args.evin_dropout_att  # 0.5
        # self.dropout_mlp = args.evin_dropout_mlp  # 0.6
        self.G1 = GateAffineAbsorpModule(args.evin_hidden_dim * 2)
        self.G2 = GateAffineAbsorpModule(args.evin_hidden_dim * 2)
        self.input_dim = args.evin_input_dim  # 768 if using BERT embedding
        self.output_dim = args.evin_hidden_dim

        self.num_heads = args.evin_nhead
        self.mhatt = MultiHeadAttention(
            self.output_dim * 2, self.num_heads, dropout=self.dropout_att)

        self.num_layers = 1

        self.doc_all_bilstm = nn.LSTM(self.input_dim, self.output_dim, self.num_layers,
                                      bidirectional=True, batch_first=True)
        self.doc_single_bilstm = nn.LSTM(self.input_dim, self.output_dim, self.num_layers,
                                         bidirectional=True, batch_first=True)
        self.query_bilstm = nn.LSTM(self.input_dim, self.output_dim, self.num_layers,
                                    bidirectional=True, batch_first=True)

        # the paper does not provide output_dim for biLSTM in Section 3.4. So I keep the param unchanged.
        self.ecl_query_bilstm = nn.LSTM(self.output_dim * 2, self.output_dim, self.num_layers,
                                        bidirectional=True, batch_first=True)
        self.ecl_doc_bilstm = nn.LSTM(self.output_dim * 2, self.output_dim, self.num_layers,
                                      bidirectional=True, batch_first=True)

        self.mlp1 = nn.Linear(4 * self.output_dim, 4 *
                              self.output_dim, bias=False)
        self.fc = nn.Linear(4 * self.output_dim,
                            self.args.output_dim_of_fact_based_model, bias=True)

        # self.doc_features_dropout = nn.Dropout(0.5)

    def forward(self, idxs, dataset, tokens_features, maps=None):
        # tokens_features:
        #   type is list, the size is batch_size. Each elem is a (num_tokens, input_dim) tensor.
        # maps:
        #   type is list (or None), the size is batch_size. Each elem is a (num_tokens) tensor.

        # === Get Top Articles Features ===

        # (batch_size, MAX_RELEVANT_ARTICLES)
        top_articles_idxs = dataset.top_articles_idxs[idxs]
        # print('top_articles_idxs: ', top_articles_idxs)

        # (batch_size, MAX_RELEVANT_ARTICLES, MAX_TOKENS_OF_A_DOC, 768)
        articles_features = torch.zeros(
            top_articles_idxs.shape[0], top_articles_idxs.shape[1], self.args.evin_max_doc_length, self.input_dim)
        # (batch_size, MAX_RELEVANT_ARTICLES, MAX_TOKENS_OF_A_DOC)
        articles_mask = torch.zeros_like(articles_features[:, :, :, 0])

        # (batch_size, MAX_TOKENS_OF_A_DOC, 768)
        articles_all_features = torch.zeros(
            top_articles_idxs.shape[0], self.args.evin_max_doc_length, self.input_dim)
        # (batch_size, MAX_TOKENS_OF_A_DOC)
        articles_all_mask = torch.zeros_like(articles_all_features[:, :, 0])

        for i, articles_idxs in enumerate(top_articles_idxs):
            curr = 0
            for j, article_idx in enumerate(articles_idxs):
                # (num_tokens, 768)
                feat = dataset.articles_features[article_idx]
                sz = min(len(feat), self.args.evin_max_doc_length)

                articles_features[i, j, :sz, :] = feat[:sz]
                articles_mask[i, j, :sz] = 1

                sz = min(len(feat), int(
                    self.args.evin_max_doc_length / top_articles_idxs.shape[1]))
                articles_all_features[i, curr:curr+sz, :] = feat[:sz]
                articles_all_mask[i, curr:curr+sz] = 1
                curr += sz

        # Use all relevant articles: (batch_size, MAX_RELEVANT_ARTICLES)
        articles_num_mask = torch.ones_like(articles_mask[:, :, 0])

        # === Get Post Features & Maps ===
        queries_features, queries_mask = self._padding(tokens_features)
        if maps is not None:
            maps, _ = self._padding([m[:, None] for m in maps])

        articles_features = articles_features.to(self.args.device)
        articles_mask = articles_mask.to(self.args.device)
        articles_num_mask = articles_num_mask.to(self.args.device)
        articles_all_features = articles_all_features.to(self.args.device)
        articles_all_mask = articles_all_mask.to(self.args.device)

        return self.forward_EVIN(queries_features, queries_mask, articles_features, articles_mask, articles_num_mask, articles_all_features, articles_all_mask, maps=maps)

    def _padding(self, t):
        # t:
        #   type is list, the size is batch_size. Each elem is a (num_tokens, dim) tensor.
        # Return:
        #   padded_t: a (batch_size, max_sequence_length, dim) tensor
        #   mask: a (batch_size, max_sequence_legnth) tensor

        dim = t[0].shape[-1]
        padded_t = torch.zeros(
            (len(t), self.max_sequence_length, dim), device=self.args.device)
        mask = torch.zeros_like(padded_t[:, :, 0])

        for i, x in enumerate(t):
            sz = min(len(x), self.max_sequence_length)
            padded_t[i, :sz] = x[:sz]
            mask[i, :sz] = 1

        return padded_t, mask

    def forward_EVIN(self, query_feature, query_mask, doc_features, doc_mask, doc_num_mask, doc_all_features, doc_all_mask, maps=None):
        # query_feature -> [batch_size, post_len(padded), embedding_size]
        # query_mask -> [batch_size, post_len(padded)]
        # doc_features -> [batch_size, #doc, doc_len(padded), embedding_size]
        # doc_mask -> [batch_size, #doc, doc_len(padded)]
        # doc_num_mask -> [batch_size, #doc]
        # doc_all_features -> [batch_size, docs_len(padded), embedding_size]
        # doc_all_mask -> [batch_size, docs_len(padded)]

        # Input Encoding Layer
        # [batch_size, post_len, hidden_size * 2]
        ec, _ = self.query_bilstm(query_feature)
        # [batch_size, docs_len, hidden_size * 2]
        erall, _ = self.doc_all_bilstm(doc_all_features)

        # TimeDistributed(BiLSTM)
        df_sizes = doc_features.size()
        # [batch_size * #doc, doc_len, emb_dim]
        er = doc_features.view(-1, df_sizes[-2], df_sizes[-1])
        # [batch_size * #doc, doc_len, hidden_size * 2]
        er = self.doc_single_bilstm(er)[0]
        # [batch_size, #doc, doc_len, hidden_size * 2]
        er = er.view(df_sizes[0], df_sizes[1], df_sizes[2], er.size()[-1])

        # er = self.doc_features_dropout(er)

        # Co-interactive Shared Layer
        # [batch_size, docs_len, hidden_size * 2]
        Hs = self.mhatt(erall, ec, ec, doc_all_mask, query_mask)
        Hsc = self.G1(ec, Hs)  # [batch_size, post_len, hidden_size * 2]
        Hsr = self.G2(erall, Hs)  # [batch_size, docs_len, hidden_size * 2]

        # Fine-grained Conflict Discovery Layer
        docs = er.size(1)
        Hsr_tmp = Hsr.repeat(docs, 1, 1)
        er_tmp = er.view(-1, er.size(-2), er.size(-1))
        doc_all_mask_tmp = doc_all_mask.repeat(docs, 1)
        doc_mask_tmp = doc_mask.view(-1, doc_mask.size(-1))
        # [batch_size * #doc, docs_len, hidden_size * 2]
        Hclf_tmp = self._single_head_att(
            Hsr_tmp, er_tmp, er_tmp, doc_all_mask_tmp, doc_mask_tmp)
        # [batch_size, #doc, docs_len, hidden_size * 2]
        Hclf = Hclf_tmp.view(-1, docs, Hclf_tmp.size(-2), Hclf_tmp.size(-1))

        # Evidence-aware Coherence Layer
        msc = self.ecl_query_bilstm(Hsc.float())[1][0]  # last hidden state
        msc = msc.permute(1, 0, 2).reshape(
            Hsc.shape[0], -1)  # [batch_size, hidden_size * 2]

        # [batch_size * #doc, num_layers * num_directions, hidden_size)
        mclf_tmp = self.ecl_doc_bilstm(Hclf_tmp.float())[1][0].permute(1, 0, 2)
        # [batch_size * #doc, hidden_size * 2]
        mclf_tmp = mclf_tmp.reshape(Hclf_tmp.size(0), -1)
        # [batch_size, #docs, hidden_size * 2]
        mclf = mclf_tmp.view(-1, docs, mclf_tmp.size(-1))

        B, num_docs, docs_len, hh = Hclf.shape
        u_cc = torch.bmm(Hclf.reshape(B, -1, hh), msc.unsqueeze(-1))
        u_cc = u_cc.view(B, num_docs, docs_len)  # [batch_size, #doc, docs_len]
        beta_cc = F.softmax(u_cc, dim=1)

        # [batch_size, #doc, hidden_size * 2]
        vclf = torch.sum(
            torch.mul(beta_cc.unsqueeze(-1).repeat(1, 1, 1, hh), Hclf), dim=2)
        nclf = vclf + mclf

        scc = self.mlp1(torch.cat((nclf, msc.unsqueeze(1).repeat(
            1, nclf.shape[1], 1)), dim=-1))  # [batch_size, #doc, hidden_size * 4]
        scc = torch.mean(scc, dim=1)

        out = self.fc(scc)
        return out

    def _single_head_att(self, Q, K, V, Q_mask, K_mask):
        # Q_mask -> [batch_size, Q_len]
        # K_mask -> [batch_size, K_len]
        Q = Q / (Q.shape[-1] ** -0.5)
        QK = torch.bmm(Q, K.transpose(-1, -2))

        # [batch_size, Q_len, K_len]
        mask = torch.bmm(Q_mask.unsqueeze(-1), K_mask.unsqueeze(-2))
        # QK = QK.masked_fill((mask == 0), - 1e8)
        QK = F.softmax(QK, dim=-1)
        QK = QK.masked_fill((mask == 0), 0)
        QK = QK / torch.sum(QK + ZERO, dim=-1, keepdim=True)

        QKV = torch.bmm(QK, V)  # [batch_size, Q_len, V_dim]

        return QKV


class GateAffineAbsorpModule(nn.Module):
    '''Evidence Inference Networks for Interpretable Claim Verification. AAAI 2021.'''

    def __init__(self, hidden_dim):
        super(GateAffineAbsorpModule, self).__init__()

        self.alpha = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )
        self.beta = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.Tanh()
        )

        self.beta2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.gamma = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, e, H):
        assert not torch.any(torch.isnan(e))
        assert not torch.any(torch.isnan(H))
        H = self.beta(H)
        assert not torch.any(torch.isnan(H))
        a = torch.sigmoid(self.beta2(H))
        b = self.alpha(e)
        c = self.gamma(H)
        assert not torch.any(torch.isnan(a))
        assert not torch.any(torch.isnan(b))
        assert not torch.any(torch.isnan(c))
        return a * b + c


class MultiHeadAttention(nn.Module):
    ''' Multi-head Attetion Module from https://github.com/dat821168/multi-head_self-attention/blob/master/selfattention.py'''

    def __init__(self, hidden_size, num_attention_heads, dropout=0.5):
        super().__init__()
        assert hidden_size % num_attention_heads == 0, "The hidden size is not a multiple of the number of attention heads"

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.use_dropout = False
        if dropout > 0:
            self.use_dropout = True
            self.dropout = nn.Dropout(dropout)

        self.dense = nn.Linear(hidden_size, hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[
            :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, Q, K, V, Q_mask, K_mask):
        # assert not torch.any(torch.isnan(Q))
        # assert not torch.any(torch.isnan(K))
        # assert not torch.any(torch.isnan(V))
        # assert not torch.any(torch.isnan(Q_mask))
        # assert not torch.any(torch.isnan(K_mask))

        # [Batch_size x Seq_length x Hidden_size]
        mixed_query_layer = self.query(Q)
        # [Batch_size x Seq_length x Hidden_size]
        mixed_key_layer = self.key(K)
        # [Batch_size x Seq_length x Hidden_size]
        mixed_value_layer = self.value(V)

        # assert not torch.any(torch.isnan(mixed_query_layer))
        # assert not torch.any(torch.isnan(mixed_key_layer))
        # assert not torch.any(torch.isnan(mixed_value_layer))

        query_layer = self.transpose_for_scores(
            mixed_query_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]
        # [Batch_size x Num_of_heads x Seq_length x Head_size]
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(
            mixed_value_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]

        # print('value_layer has NAN: ', torch.any(torch.isnan(value_layer)))

        # assert not torch.any(torch.isnan(query_layer))
        # assert not torch.any(torch.isnan(key_layer))
        # assert not torch.any(torch.isnan(value_layer))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1,
                                                                         -2))  # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        attention_scores = attention_scores / (
            self.attention_head_size ** -0.5)  # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        # [batch_size, Q_len, K_len]
        mask = torch.bmm(Q_mask.unsqueeze(-1), K_mask.unsqueeze(-2))
        # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        attention_probs = self.softmax(attention_scores)
        attention_probs = attention_probs.masked_fill(
            (mask[:, None, :, :] == 0), 0)
        attention_probs = attention_probs / \
            (torch.sum(attention_probs + ZERO, dim=-1, keepdim=True))

        # assert not torch.any(torch.isnan(attention_scores))
        # assert not torch.any(torch.isnan(mask))
        # assert not torch.any(torch.isnan(attention_probs))

        if self.use_dropout:
            attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs,
                                     value_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]

        context_layer = context_layer.permute(0, 2, 1,
                                              3).contiguous()  # [Batch_size x Seq_length x Num_of_heads x Head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size,)  # [Batch_size x Seq_length x Hidden_size]
        # [Batch_size x Seq_length x Hidden_size]
        context_layer = context_layer.view(*new_context_layer_shape)
        # output = self.dropout(self.fc(output))
        output = self.dense(context_layer)

        # assert not torch.any(torch.isnan(context_layer))
        # assert not torch.any(torch.isnan(output))

        return output


class DeClarE(BaseDetector):
    ''' DeClarE: Debunking Fake News and False Claims using Evidence-Aware Deep Learning. EMNLP 2018.'''

    def __init__(self, args):
        super(DeClarE, self).__init__(args)

        self.args = args

        self.max_sequence_length = args.declare_input_max_sequence_length

        self.input_dim = args.declare_input_dim  # 768 if using BERT embedding
        self.hidden_size = args.declare_hidden_dim
        self.num_layers = args.declare_bilstm_num_layer

        self.post_bilstm = nn.LSTM(self.input_dim, self.hidden_size, self.num_layers,
                              bidirectional=True, batch_first=True, dropout=args.declare_bilstm_dropout)
        self.articles_bilstm = nn.LSTM(self.input_dim, self.hidden_size, self.num_layers,
                              bidirectional=True, batch_first=True, dropout=args.declare_bilstm_dropout)

        self.Wa = nn.Linear(4 * self.hidden_size, 1, bias=True)
        self.Wc = nn.Linear(2 * self.hidden_size, 2 *
                            self.hidden_size, bias=True)

        self.fc = nn.Linear(self.hidden_size * 2,
                            args.output_dim_of_fact_based_model)

    def forward(self, idxs, dataset, tokens_features, maps=None):
        # queries_features: (batch_size, max_nodes, 768)
        # queries_mask: (batch_size, max_nodes)
        # maps: (batch_size, max_nodes) or None
        # articles_features: (batch_size, #doc, max_doc_len, 768)
        # articles_mask: (batch_size, #doc, max_doc_len)
        # articles_num_mask: (batch_size, #doc)

        queries_features, queries_mask, articles_features, articles_mask, articles_num_mask = self.init_fact_based_detector_input(
            idxs, dataset, tokens_features)

        # ========= Claim(Post) Specific Attention ========= 
        # (batch_size, max_nodes, hidden_size * 2)
        hq, _ = self.post_bilstm(queries_features)
        hq = hq.masked_fill((queries_mask[:, :, None] == 0), 0)

        if maps is None:
            # (batch_size, 1, hidden_size * 2)
            pbar = torch.sum(queries_mask[:, :, None] * hq, dim=1, keepdim=True)
        else:
            pbar = torch.sum(maps[:, :, None] * hq, dim=1, keepdim=True)

        df_sizes = articles_features.size()
        hd = articles_features.view(-1, df_sizes[-2], df_sizes[-1])
        hd, _ = self.articles_bilstm(hd)
        # (batch_size, #doc, doc_len, hidden_size * 2)
        hd = hd.view(df_sizes[0], df_sizes[1], df_sizes[2], hd.shape[-1])

        # (batch_size, #doc, doc_len, hidden_size * 4)
        hd_hat = torch.cat((hd, pbar.unsqueeze(1).repeat(
            1, hd.shape[1], hd.shape[2], 1)), dim=-1)

        # (batch_size, #doc, doc_len, 1)
        hd_prime = torch.tanh(self.Wa(hd_hat))
        hd_prime = hd_prime.squeeze(-1)
        hd_prime = hd_prime.masked_fill((articles_mask == 0), -np.inf)
        # (batch_size, #doc, doc_len)
        alphad = F.softmax(hd_prime, dim=2)

        # ========= Per-Article Credibility Score of Claim =========
        # (batch_size, #doc, doc_len, hidden_size * 2)
        g = alphad[:, :, :, None] * hd
        # (batch_size, #doc, hidden_size * 2)
        g = torch.sum(articles_mask[:, :, :, None] * g, dim=-2)
        # (batch_size, hidden_size * 2)
        g = torch.sum(articles_num_mask[:, :, None] * g, dim=-2)

        out = self.fc(g)
        return out
