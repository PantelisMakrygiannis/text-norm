# text-norm

## Installation

### 1. Clone the repo/ Create enviroment/ Install requirements & SpaCy model
```bash
git clone https://github.com/PantelisMakrygiannis/text-norm.git
cd text-norm

conda create -n textnorm python=3.10 -y
conda activate textnorm
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Usage

### Normalize a single input
```bash
python agentic_text_normalization.py --input "Smith, John/Jane Doe"

### Run built-in demo examples
```bash
python agentic_text_normalization.py --demo

### Run evaluation on a sample dataset
```bash
python agentic_text_normalization.py --eval_csv sample_100.csv











