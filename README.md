# Abstractive Summarization of Opposed Political Perspectives on Controversial Issues
Code accompanying COMS 6998 project on an _Abstractive Summarization of Opposed Political Perspectives on Controversial Issues_

# Repository Hierarchy
All relevant code and scripts are held in the code directory. The hierarchy is broken down below:
- code/
  - data_collection/
    - Holds all jupyter notebooks for scraping the FlipSide (1, 2, & 3), Reddit (4), and Twitter (5) as well as consolidation into a single dataset for use in experiments.
    - twitter_utils/
      - scrape_twitter.sh: bash script for repeatedly calling the `pull_twitter` [repo](https://github.com/dhudsmith/pull_twitter) api for each url query
      - abscos_config.yaml: config file needed for `twitter_pull` 
  - experiments/
    - abstractive/
      - pretrained: CLI code for running experiments with abstractive baselines
    - extractive/
      - oracle: Code for evaluating the "extractive oracle" baseline (upper bound of ROUGE-2 scores and accompanying metrics)
      - cmos_baseline: Inference code for the CMOS baseline
      - lexrank_baseline: Inference code for the LexRank Baseline
    - togl_decoding/
      - TAM/: Reimplementation of the Topic-Aspect Model (TAM)
      - ToGL_Decoding/: Definition of togl_decoder class (togl_decoder.py) and CLI code for inference (togl_generate.py)
    - Full Evaluation.ipynb
      - Notebook for evaluating all models for all experiments

# Main Execution Paths

## 0) Installation and Environment Setup
Two external repos need to be cloned to enable data collection and the CoCoSum Baseline.

__Pull_Twitter__ `pull_twitter` is a small library I co-authored that simplifies queries to the Twitter API. Run the following line from the `code/data_collection/` directory:
```
git clone https://github.com/dhudsmith/pull_twitter.git
```

__CoCoSum__ The authors of [Comparative Opinion Summarization via Collaborative Decoding](https://arxiv.org/pdf/2110.07520/) released the implementation of their model, CoCoSum. Run the following line from the `code/experiments/abstractive/` directory:
```
git clone https://github.com/megagonlabs/cocosum.git
```

In the base directory, create a virtual environment, activate, and pip install requirements.txt:
```
python -m virtualenv abs_cos
source abs_cos/bin/activate
pip install -r requirements.txt
```

## 1) Data Generation and Preprocessing
The `code/data_collection/` folder holds all code for scraping TheFlipSide, Twitter, and Reddit in jupyter notebooks. The notebooks are numbered in the order they should be executed, with a note in `5 - Twitter Scraper.ipynb` stating when to run the scraping bash script. The first 4 can be executed automatically with `jupyter nbconvert --execute <Notebook 1-4>>`.

Run `5 - Twitter Scraper.ipynb` as well as the following command in the notebook:
```
  ./scrape_twitter.sh
```

The final notebook can be run with `jupyter nbconvert --execute "6 - Data Consolidation and Statistics.ipynb"`

The final notebook also preprocesses the data into separate csv's to be input to ToGL and baseline training/inference scripts. More details on data collection and preprocessing steps are included in a ReadMe in the data_collection folder.

## 2) Baseline Training

## 3) Training Experiments

## 4) Evaluation


