# Readme

This is an implementation for [Counterfactual Fairness: Unidentification, Bound and Algorithm](https://doi.org/10.24963/ijcai.2019/199) in IJCAI 2019.

## Development

1. Our implementation is based on Python 3.6 in Windows 10 (64-Bit).
2. The python distribution [Anaconda](https://www.anaconda.com) or [Miniconda](https://repo.continuum.io/miniconda/) is highly recommended. Since we utilize the environment management tool `conda`, Miniconda is minimal and sufficient.

## Reproduction

To reproduce this repository:

1. Recover the environment by `conda env create --file environment.yml --name YOUR_ENV_NAME`.
2. Run
   1. `python synthetic_detect.py` to get Table 1;
   2. `python synthetic_fair_classification.py` to get Tables 2 and 3;
   3. `python adult_fair_classification.py` to get Tables 4 and 5.

## BibTex

```
@inproceedings{DBLP:conf/ijcai/Wu0W19,
  author    = {Yongkai Wu and
               Lu Zhang and
               Xintao Wu},
  editor    = {Sarit Kraus},
  title     = {Counterfactual Fairness: Unidentification, Bound and Algorithm},
  booktitle = {Proceedings of the Twenty-Eighth International Joint Conference on
               Artificial Intelligence, {IJCAI} 2019, Macao, China, August 10-16,
               2019},
  pages     = {1438--1444},
  publisher = {ijcai.org},
  year      = {2019},
  url       = {https://doi.org/10.24963/ijcai.2019/199},
  doi       = {10.24963/ijcai.2019/199},
  timestamp = {Tue, 20 Aug 2019 16:18:18 +0200},
  biburl    = {https://dblp.org/rec/conf/ijcai/Wu0W19.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
