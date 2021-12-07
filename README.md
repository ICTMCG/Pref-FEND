# Pref-FEND

This is the official repository of the paper:

> **Integrating Pattern- and Fact-based Fake News Detection via Model Preference Learning.**
>
> Qiang Sheng\*, Xueyao Zhang\*, Juan Cao, and Lei Zhong.
>
> *Proceedings of the 30th ACM International Conference on Information and Knowledge Management (CIKM 2021)*
>
> [PDF](https://dl.acm.org/doi/10.1145/3459637.3482440) / [Poster](https://www.zhangxueyao.com/data/cikm2021-PrefFEND-poster.pdf) / [Code](https://github.com/ICTMCG/Pref-FEND) / [Chinese Blog](https://zhuanlan.zhihu.com/p/414464291)

## Datasets

The experimental datasets where can be seen in `dataset` folder, including the [Weibo Dataset](https://github.com/ICTMCG/Pref-FEND/tree/main/dataset/Weibo), and the [Twitter Dataset](https://github.com/ICTMCG/Pref-FEND/tree/main/dataset/Twitter). Note that you can download the datasets only after an ["Application to Use the Datasets for Pattern- and Fact-based Joint Fake News Detection"](https://forms.office.com/r/HF00qdb3Zk) has been submitted.

## Code

### Key Requirements

```
python==3.6.10
torch==1.6.0
torchvision==0.7.0
torch-geometric==1.7.0
torch-sparse==0.6.9
transformers==3.2.0
```

### Preparation 

#### Step1: Stylistic Tokens & Entities Recognition

Notice that Step1 is not necessary, because we have supplied the recognized results in dataset json files in `dataset` folder. 

If you would like to know about the details of the recoginition procedure, you can refer to the `preprocess/tokens_recognition` folder.

#### Step2: Tokenize

```
cd preprocess/tokenize
```

As the `run.sh` shows, you need to run:

```
python get_post_tokens.py --dataset [dataset] --pretrained_model [bert_pretrained_model]
```

#### Step3: Heterogeneous Graph Initialization

```
cd preprocess/graph_init
```

As the `run.sh` shows, you need to run:

```
python init_graph.py --dataset [dataset] --max_nodes [max_tokens_num]
```

#### Step4: Preparation of the Fact-based Models

Notice that Step3 is not necessary if you wouldn't use fact-based models as a componet for Pref-FEND.

##### Tokenize

```
cd preprocess/tokenize
```

As the `run.sh` shows, you need to run:

```
python get_articles_tokens.py --dataset [dataset] --pretrained_model [bert_pretrained_model]
```

##### Retrieve by BM25

```
cd preprocess/bm25
```

As the `run.sh` shows, you need to run:

```
python retrieve.py --dataset [dataset]
```

#### Step5: Preparation for some special fake news detectors

Notice that Step5 is not necessary if you wouldn't use `EANN-Text` or `BERT-Emo` as a componet for Pref-FEND.

##### EANN-Text

```
cd preprocess/EANN_Text
```

As the `run.sh` shows, you need to run:

```
python events_clustering.py --dataset [dataset] --events_num [clusters_num]
```

##### BERT-Emo

```
cd preprocess/BERT_Emo/code/preprocess
```

As the `run.sh` shows, you need to run:

```
python input_of_emotions.py --dataset [dataset]
```

### Training and Inferring

```
cd model
mkdir ckpts
```

Here we list all of our configurations and running script in  `run_weibo.sh` and `run_twitter.sh`. For example, if you would like to run `BERT-Emo` (Pattern-based) + `MAC` (Fact-based) with Pref-FEND on Weibo, you can run:

```
# BERT_Emo + MAC (Pref-FNED)
CUDA_VISIBLE_DEVICES=0 python main.py --dataset 'Weibo' \
--use_preference_map True --use_pattern_based_model True --use_fact_based_model True \
--pattern_based_model 'BERT_Emo' --fact_based_model 'MAC' \
--lr 5e-6 --batch_size 4 --epochs 50 \
--save 'ckpts/BERT_Emo+DeClarE_with_Pref-FEND'
```

then the results will be saved in `ckpts/BERT_Emo+DeClarE_with_Pref-FEND`.

## Citation

```
@inproceedings{Pref-FEND,
  author    = {Qiang Sheng and
               Xueyao Zhang and
               Juan Cao and
               Lei Zhong},
  title     = {Integrating Pattern- and Fact-based Fake News Detection via Model
               Preference Learning},
  booktitle = {{CIKM} '21: The 30th {ACM} International Conference on Information
               and Knowledge Management, Virtual Event, Queensland, Australia, November
               1 - 5, 2021},
  pages     = {1640--1650},
  year      = {2021},
  url       = {https://doi.org/10.1145/3459637.3482440},
  doi       = {10.1145/3459637.3482440}
}
```

