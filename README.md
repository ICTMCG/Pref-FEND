# Pref-FEND

**[Notes]** The repo may be incomplete and some of the code is a bit messy. We will improve in the near future. Readme will also include more details. Coming soon stay tuned :)

---

This is the official repository of the paper:

> **Integrating Pattern- and Fact-based Fake News Detection via Model Preference Learning.**
>
> Qiang Sheng\*, Xueyao Zhang\*, Juan Cao, and Lei Zhong.
>
> *Proceedings of the 30th ACM International Conference on Information and Knowledge Management (CIKM 2021)*
>
> [PDF](https://dl.acm.org/doi/10.1145/3459637.3482440) / [Poster](https://www.zhangxueyao.com/data/cikm2021-PrefFEND-poster.pdf) / [Code](https://github.com/ICTMCG/Pref-FEND) / [Chinese Blog](https://zhuanlan.zhihu.com/p/414464291)

## Datasets

The experimental datasets where can be seen in `dataset` folder, including the [Weibo Dataset](https://github.com/ICTMCG/Pref-FEND/tree/main/dataset/Weibo), and the [Twitter Dataset](https://github.com/ICTMCG/Pref-FEND/tree/main/dataset/Twitter).

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

#### Stylistic Tokens & Entities Recognition

#### Heterogeneous Graph Initialization

#### Preparation of the Fact-based Models

##### Tokenize

##### Retrieve by BM25

### Training and Inferring

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

