# Abstractive Summarization of Opposed Political Perspectives on Controversial Issues
Code accompanying COMS 6998 project on an _Abstractive Summarization of Opposed Political Perspectives on Controversial Issues_. [Skip to Execution Paths](https://github.com/NickDeas/Abstract-Contrastive-Opinion-Summarization#main-execution-paths)

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

Create 3 directories in the base directory of the full repo `Abstract-Contrastive-Opinion-Summarization`:
```
mkdir results
mkdir data
mkdir drivers
```

Then, install a chrome driver for your system and place it in the drivers directory for scraping TheFlipSide. Drivers can be installed from [https://chromedriver.chromium.org/downloads](https://chromedriver.chromium.org/downloads)

Finally, a small change to the `torchmetrics` library needed to be made for BertScores to be calculated correctly. Find the installation of torchmetrics, and go to `torchmetrics/bert.py`. In lines 188 and 196, change `truncation = False` to `truncation = True`.

## 1) Data Generation and Preprocessing
<img src="https://github.com/NickDeas/Abstract-Contrastive-Opinion-Summarization/blob/main/images/FlipSide%20Example.png?raw=True" width = 900>
The `code/data_collection/` folder holds all code for scraping TheFlipSide, Twitter, and Reddit in jupyter notebooks. The notebooks are numbered in the order they should be executed, with a note in `5 - Twitter Scraper.ipynb` stating when to run the scraping bash script. The first 4 can be executed automatically with `jupyter nbconvert --execute <Notebook 1-4>`, repolace `<>` with the name of each notebook.

Run `5 - Twitter Scraper.ipynb` as well as the following command in the notebook:
```
  ./scrape_twitter.sh
```

The final notebook can be run with `jupyter nbconvert --execute "6 - Data Consolidation and Statistics.ipynb"`

The final notebook also preprocesses the data into separate csv's to be input to ToGL and baseline training/inference scripts. More details on data collection and preprocessing steps are included in a ReadMe in the data_collection folder.

## 2) Baseline Training
In the experiments folder, baseline training and inference are split into `extractive/` and `abstractive`.
__Extractive__ Extractive baselines (Extractive oracle, CMOS, and LexRank) each have their own directory. Running the training and inference can be done by running the following code from the `extractive` directory:
```
jupyter nbconvert --execute "oracle/Maximum Extractive Baseline.ipynb"
jupyter nbconvert --execute "cmos_baseline/CMOS Baseline Evaluation.ipynb"
jupyter nbconvert --execute "lexrank_baseline/LexRank Baseline.ipynb"
```
Results will automatically be saved to the results directory.

__Abstractive__ Training and inference of abstractive baselines is packaged in a CLI to enable running many experiments. Within the `code/experiments/abstractive/pretrained` directory, both experiments without fine-tuning and with fine-tuning can be run.

The `main.py` script manages training and inference, with the first parameter designating which experiment to run. Running `python -m main zshot ...` runs the zero-shot experiment, and `python-m main fshot` runs the few shot experiment. For each model, `zshot` need only be run once, but `fshot` should be run for each kfold. CLI command details are below the execution paths.

__CoCoSum__ Follow the CoCoSum github page for instructions on running CoCoSum. A json file in the format expected by CoCoSum is created by the data_collection folder scripts and will be held in the `data/` directory.

## 3) Training Experiments
<img src="https://github.com/NickDeas/Abstract-Contrastive-Opinion-Summarization/blob/main/images/ToGL%20Decoding.png?raw=True" width=350>
Training and inference for ToGL-Decoding is completed in three steps. 
  
The first step fits the reimplemented Topic-Aspect Model to PoliSum. The notebook for training the TAM is located in `code/experiments/togl_decoding/TAM/TAM Training.ipynb`.
  
The second step runs the fitted TAM on PoliSum to pre-generate the ToGL distributions for evaluation. The notebook to generate the distributions is located in `code/experiments/togl_decoding/TAM/Aspect Prediction.ipynb`

The third step involves running inference with ToGL-Decoding and the TAM fit in the last step, which is also packaged in a CLI for easy experiments. In the `code/experiments/togl_decoding/ToGL_Decoding/` directory, the CLI is run with `togl_generate.py` to generate summaries. Parameters are explained in more detail below.CLI command details are below the execution paths.

## 4) Evaluation

Finally, all baselines and models are evaluated at the same time in a single jupyter notebook contained in `code/experiments/Full Evaluation.ipynb`. As other notebooks, the evaluation can be ran with `jupyter nbconvert --execute "Full Evaluation.ipynb`, but the file will need to be opened afterward to view results for each of the 3 (2 experiments and 1 supplementary) experiments.

# CLI Commands

## Zero Shot Abstractive Baseline
The following lists parameters for the command `python -m main zhot ...`
|Parameter (Short) | Description | Required (Default) |
|------------------|-------------|--------------------|
|--model (-m) | The name of the base model checkpoint | Yes |
|--test-fp (-tfp) | File path of the test csv file | Yes |
|--batch-size (-bs) | Batch size to use in evaluation | No (4) |
|-results-fp (-rf) | Directory to store logs and model generations | No ('./') |

## Few Shot/KFold Abstractive Baseline
The following lists parameters for the command `python -m main fhot ...`
|Parameter (Short) | Description | Required (Default) |
|------------------|-------------|--------------------|
|--model (-m) | The name of the base model checkpoint | Yes |
|--train-fp (-tr) | File path of the train csv file | Yes |
|--val-fp (-va) | File path of the validation csv file | No |
|--test-fp (-te) | File path of the test csv file | Yes |
|--batch-size (-bs) | Batch size to use in training and evaluation | No (4) |
|--chkpt-dir (-cd) | Directory to store model checkpoints in during training | No ('./') |
|-results-fp (-rf) | Directory to store logs and model generations | No ('./') |

## ToGL-Decoding
The following lists parameters for the command `python -m togl_generate ...`
|Parameter (Short) | Description | Required (Default) |
|------------------|-------------|--------------------|
|--intput (-i) | Input csv file path | Yes |
|--src-col (-sc) | Column name in input containing source texts | Yes |--togl-left (-tl) | File path to json dictionary of togl distributions for left summaries | Yes |
|--togl-right (-tr) | File path to json dictionary of togl distributions for right summaries | Yes |
|--output (-o) | Output csv path to store generated summaries | Yes |
|--model (-m) | Base pretrained model checkpoint to use in togl decoding | No ('facebook/bart-large-xsum') |
|--togl-func (-tf) | Function to use in combining output word distributions and ToGL distributions | No ('sum') |
|--togl-start (-ts) | Minimum index of token when togl distributions are incorporated | No (3) |
| --togl-weight (-tw) | Weighting of togl distributions in generation (should be less than 1) | No (0.1) |
| --num-beams (-nb) | Number of beams to use in beam search during generation | No (3) |
| --no-repeat-ngram (-nr) | No Repeat Ngram Size to constrain generations | No (3) |
| --min-length (-mi) | Minimum length of generations | No (16) |
| --max-length (-ma) | Maximum length of generations | No (128) |
|--device (-d) | Torch cuda device to use in generation | No ('cuda:0') |

