from argparse import ArgumentParser
import json
from tqdm import tqdm
import pickle
import os
from transformers import BertTokenizer


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
    save_dir = os.path.join(save_dir, 'articles')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # ====== Tokens ======
    with open('../../dataset/{}/raw/articles/articles.json'.format(dataset), 'r') as f:
        articles = json.load(f)
        print(len(articles))

    articles_tokens = [get_bert_tokens(a['text']) for a in tqdm(articles)]
    with open(os.path.join(save_dir, 'articles_tokens.pkl'), 'wb') as f:
        pickle.dump(articles_tokens, f)
