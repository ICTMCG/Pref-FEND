# Weibo Dataset

## Access

You will be shared the dataset by email after an ["Application to Use the Datasets for Pattern- and Fact-based Joint Fake News Detection"](https://forms.office.com/r/HF00qdb3Zk) has been submitted.

## Description

### Post

The posts are saved in `raw/post` folder, which contain `train.json`, `val.json`, and `test.json`. In every json file:

- the `content` identifies the content of the post.
- the `label` identifies the veracity of the post, whose value is `fake` or `real`.
- the `words` identifies the tokens and their type of the content:
  - the data type of  `words` is a List.
  - every item of `words` is a 2-Tuple, in which the first represents the token, and the last  represents the type of the token, ranging in `PATTERN`, `ENTITY`, and `OTHERS`.
  - the recognition procedure of the types for the tokens is described at [https://github.com/ICTMCG/Pref-FEND#stylistic-tokens--entities-recognition](https://github.com/ICTMCG/Pref-FEND#stylistic-tokens--entities-recognition).

### Relevant Articles

The relevant articles are saved in `raw/articles/articles.json`, in which:

- the `text` identifies the content of the article.
- the `_id` (if exists) identifies the unique id of the article.
- the `time` and `time_format` (if exists) identifies the publised time of the article.
- the `url` (if exists) identifies the raw URL of the article.