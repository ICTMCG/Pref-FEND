from argparse import ArgumentParser
import json
from tqdm import tqdm
import torch
import os
from transformers import BertTokenizer, BertModel


def get_bert_tokens(text):
    return tokenizer.encode(text, add_special_tokens=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--pretrained_model', type=str)
    args = parser.parse_args()

    dataset = args.dataset
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)

    save_dir = 'data'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, dataset)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, 'post')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # ====== Embedding Layer ======
    pretraned_model = BertModel.from_pretrained(args.pretrained_model)
    embedding_layer = list(list(pretraned_model.children())[0].children())[0]
    print('Bert: {},\n\n Embedding Layer:\n {}\n'.format(
        args.pretrained_model, embedding_layer))
    torch.save(embedding_layer,
               'data/{}/BertTokenEmbedding.pt'.format(args.dataset))

    # ====== Tokens ======
    datasets = []
    for t in ['train', 'val', 'test']:
        with open('../../dataset/{}/raw/post/{}.json'.format(dataset, t), 'r') as f:
            pieces = json.load(f)
            print(t, len(pieces))
            datasets.append(pieces)

    print('\n', '-'*10, '\n')
    for pieces in datasets:
        for p in tqdm(pieces):
            graph_words = [(t, w, get_bert_tokens(w)) for w, t in p['words']]
            graph_words = [(t, w, tokens)
                           for t, w, tokens in graph_words if len(tokens) > 0]
            p['graph'] = graph_words
            del p['words']

    for i, t in enumerate(['train', 'val', 'test']):
        with open(os.path.join(save_dir, 'graph_{}.json'.format(t)), 'w') as f:
            json.dump(datasets[i], f, indent=4, ensure_ascii=False)
