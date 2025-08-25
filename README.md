# text-norm

## Installation

### Clone the repo
```bash
git clone https://github.com/PantelisMakrygiannis/text-norm.git 
cd text-norm
```

### Create enviroment
```bash
conda create -n textnorm python=3.10 -y 
conda activate textnorm 
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

### Install requirements & SpaCy model
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```


## Usage

### Normalize a single input
```bash 
python agentic_text_normalization.py --input "<Unknown>/Wright, Justyce Kaseem"
```

### Run built-in demo examples
```bash
python agentic_text_normalization.py --demo
```

### Run evaluation on a sample dataset
```bash
python agentic_text_normalization.py --eval_csv sample_100.csv
```
