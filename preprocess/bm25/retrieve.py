from gensim.summarization import bm25
import numpy as np
import json
import os
from tqdm import tqdm
from argparse import ArgumentParser

TOP = 10

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()

    dataset = args.dataset

    save_dir = 'data'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, dataset)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Corpus
    with open('../../dataset/{}/raw/articles/articles.json'.format(dataset), 'r') as f:
        articles = json.load(f)
    corpus = [a['text'] for a in articles]
    print('Corpus: {}'.format(len(corpus)))

    print('\nLoading BM25 Model...')
    bm_model = bm25.BM25(corpus)
    print('Done.')

    for dataset_type in ['train', 'val', 'test']:
        with open('../../dataset/{}/raw/post/{}.json'.format(dataset, dataset_type), 'r') as f:
            pieces = json.load(f)
        queries = [p['content'] for p in pieces]
        print('\nDataset: {}, sz = {}'.format(dataset_type, len(queries)))

        bm25_scores = np.zeros((len(queries), len(corpus)))
        for i, query in enumerate(tqdm(queries)):
            scores = np.array(bm_model.get_scores(query))
            bm25_scores[i] = scores

        print('\nSaving the BM25 results...')
        np.save(os.path.join(save_dir, 'bm25_scores_{}_{}.npy'.format(
            dataset_type, bm25_scores.shape)), bm25_scores)
        print('Done.')

        print('\nRanking and Exporting...')
        ranked_arr = (-bm25_scores).argsort()
        arr = ranked_arr[:, :TOP]
        print('Top{}: {}'.format(TOP, arr.shape))
        np.save(os.path.join(
            save_dir, 'top{}_articles_{}.npy'.format(TOP, dataset_type)), arr)
        print('Done.')
