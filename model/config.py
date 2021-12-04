from argparse import ArgumentParser, ArgumentTypeError
import os


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


INDEX_OF_LABEL = {'fake': 1, 'real': 0}
INDEX2LABEL = ['real', 'fake']

MAX_TOKENS_OF_A_POST = 100
MAX_TOKENS_OF_A_DOC = 200
MAX_RELEVANT_ARTICLES = 5


parser = ArgumentParser(description='Pref-FEND')

# ======================== Dataset ========================

parser.add_argument('--dataset', type=str, default='Weibo')
parser.add_argument('--category_num', type=int, default=2)
parser.add_argument('--save', type=str, default='./ckpts/debug',
                    help='folder to save the final model')

# ======================== PrefFEND Framework ========================

# --- Framework ---
parser.add_argument('--use_preference_map', type=str2bool, default=True)
parser.add_argument('--use_pattern_based_model', type=str2bool, default=True)
parser.add_argument('--use_fact_based_model', type=str2bool, default=True)

# --- HetDGCN ---
parser.add_argument('--num_gnn_layers', type=int, default=2)
parser.add_argument('--dim_node_features', type=int, default=768)
parser.add_argument('--updated_weights_for_A', type=float,
                    default=0.5, help='1 - alpha')

# --- Pattern- and Fact-based Models ---
parser.add_argument('--pattern_based_model', type=str,
                    default='', help='[BiLSTM, BERT_Emo, EANN_Text]')
parser.add_argument('--fact_based_model', type=str,
                    default='', help='[DeClarE, EVIN, MAC]')
parser.add_argument('--output_dim_of_pattern_based_model',
                    type=int, default=256)
parser.add_argument('--output_dim_of_fact_based_model', type=int, default=256)

# --- MLP ---
parser.add_argument('--num_mlp_layers', type=int, default=3)

# --- Loss ---
parser.add_argument('--weight_of_normal_loss',
                    type=float, default=2.0, help='beta1')
parser.add_argument('--weight_of_preference_loss',
                    type=float, default=1.0, help='beta2')
parser.add_argument('--weight_of_reversed_loss',
                    type=float, default=1.0, help='beta3')

# ======================== Pattern-based Models ========================

# --- BiLSTM ---
parser.add_argument('--bilstm_input_max_sequence_length',
                    type=int, default=MAX_TOKENS_OF_A_POST)
parser.add_argument('--bilstm_input_dim', type=int, default=768)
parser.add_argument('--bilstm_hidden_dim', type=int, default=128)
parser.add_argument('--bilstm_num_layer', type=int, default=1)
parser.add_argument('--bilstm_dropout', type=float, default=0)

# --- EANN_Text ---
parser.add_argument('--eann_input_max_sequence_length',
                    type=int, default=MAX_TOKENS_OF_A_POST)
parser.add_argument('--eann_input_dim', type=int, default=768)
parser.add_argument('--eann_hidden_dim', type=int, default=64)
parser.add_argument('--eann_event_num', type=int, default=300)
parser.add_argument('--eann_use_textcnn', type=bool, default=True)
parser.add_argument('--eann_weight_of_event_loss', type=float, default=1.0)


# --- BERT_Emo ---
parser.add_argument('--bert_pretrained_model', type=str)
parser.add_argument('--bert_input_max_sequence_length',
                    type=int, default=MAX_TOKENS_OF_A_POST)
parser.add_argument('--bert_training_embedding_layers',
                    type=str2bool, default=True)
parser.add_argument('--bert_training_inter_layers',
                    type=str2bool, default=True)
parser.add_argument('--bert_emotion_dim', type=int, default=0)
parser.add_argument('--bert_hidden_dim', type=int, default=768)

# ======================== Fact-based Models ========================

# --- DeClarE ---
parser.add_argument('--declare_input_max_sequence_length',
                    type=int, default=MAX_TOKENS_OF_A_POST)
parser.add_argument('--declare_input_dim', type=int, default=768)
parser.add_argument('--declare_hidden_dim', type=int, default=128)
parser.add_argument('--declare_max_doc_length', type=int,
                    default=MAX_TOKENS_OF_A_DOC)
parser.add_argument('--declare_bilstm_num_layer', type=float, default=1)
parser.add_argument('--declare_bilstm_dropout', type=float, default=0)

# --- EVIN ---
parser.add_argument('--evin_input_max_sequence_length',
                    type=int, default=MAX_TOKENS_OF_A_POST)
parser.add_argument('--evin_max_doc_length', type=int,
                    default=MAX_TOKENS_OF_A_DOC)
parser.add_argument('--evin_input_dim', type=int, default=768)
parser.add_argument('--evin_hidden_dim', type=int,
                    default=60, help='hidden_dim * 2 % nhead = 0')
parser.add_argument('--evin_dropout_att', type=float, default=0.5)
parser.add_argument('--evin_dropout_mlp', type=float, default=0.6)
parser.add_argument('--evin_nhead', type=int, default=6)

# --- MAC ---
parser.add_argument('--mac_input_max_sequence_length',
                    type=int, default=MAX_TOKENS_OF_A_POST)
parser.add_argument('--mac_max_doc_length', type=int,
                    default=MAX_TOKENS_OF_A_DOC)
parser.add_argument('--mac_input_dim', type=int, default=768)
parser.add_argument('--mac_hidden_dim', type=int, default=300)
parser.add_argument('--mac_dropout_doc', type=float, default=0)
parser.add_argument('--mac_dropout_query', type=float, default=0)
parser.add_argument('--mac_nhead_1', type=int, default=5)
parser.add_argument('--mac_nhead_2', type=int, default=2)

# ======================== Training ========================

parser.add_argument('--lr', type=float, default=5e-5,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=50,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='No. of the epoch to start training')
parser.add_argument('--resume', type=str, default='',
                    help='path to load trained model')
parser.add_argument('--evaluate', type=str2bool, default=False,
                    help='only use for evaluating')

parser.add_argument('--debug', type=str2bool, default=False)

# ======================== Devices ========================

parser.add_argument('--seed', type=int, default=9,
                    help='random seed')
parser.add_argument('--device', default='cpu')
parser.add_argument('--fp16', type=str2bool, default=True,
                    help='use fp16 for training')
parser.add_argument('--local_rank', type=int, default=-1)
