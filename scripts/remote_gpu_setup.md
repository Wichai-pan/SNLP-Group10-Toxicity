# Remote GPU setup (single CUDA GPU)

This project is designed to run on a single GPU (≈8GB). Steps below keep the environment isolated in the project directory.

## 1) Create a project-local venv

```bash
python3 -m venv .venv-xlmr
source .venv-xlmr/bin/activate
python -m pip install -U pip
```

## 2) Check CUDA/driver

```bash
nvidia-smi
```

## 3) Install PyTorch (CUDA build)

Install the CUDA-enabled wheel that matches your server (follow the official PyTorch “Get Started” selector), then install the remaining deps:

```bash
pip install -r requirements.txt
```

## 4) Train/eval + write test predictions

```bash
python run_xlmr.py --preset score --device cuda --hf_home artifacts/hf_home --predict_test \
  --out_tsv artifacts/pred_xlmr_test.tsv
```

