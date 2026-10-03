# Grokking modular addition

A small, self-contained reproduction of grokking in a one-layer transformer.

For the prime \(p=113\), the input is

\[
[a,\ b,\ =]
\]

and the target is

\[
c=(a+b)\bmod 113.
\]

There are \(113^2=12{,}769\) possible pairs. A fixed random 30% subset, 3,830 pairs, is used for training. The remaining 8,939 pairs form the test set.

The canonical setup follows Nanda et al.:

- One transformer block
- Model dimension 128
- Four attention heads of dimension 32
- ReLU MLP with hidden dimension 512
- No layer normalization
- Untied token embedding and unembedding
- Full-batch AdamW
- Learning rate \(10^{-3}\)
- Weight decay 1
- Adam betas \((0.9, 0.98)\)
- 40,000 epochs
- Model seed 999 and data seed 598

The model has 227,200 trainable parameters.

## Model size

| Model | Parameters | Relative to this model |
| --- | ---: | ---: |
| LeNet-5 for MNIST | about 60,000 | this model is 3.8 times larger |
| This modular-addition transformer | 227,200 | 1x |
| GPT-2 small checkpoint | 124,439,808 | about 548 times larger |
| GPT-2 XL | about 1.5 billion | about 6,600 times larger |

The experiment is tiny by modern transformer standards, but already larger than the classic LeNet-5 image classifier.

## Repository layout

```text
.
├── analysis.py       # Weight movement and Fourier projections
├── data.py           # Complete modular-addition table and fixed split
├── main.py           # Experiment setup and output generation
├── model.py          # One-layer transformer
├── plot.py           # Training and Fourier-clock plots
├── report.py         # Run summaries and readable statistics
├── train.py          # Full-batch training loop and measurements
├── requirements.txt
├── papers/
│   ├── power-2022-grokking.pdf
│   └── nanda-2023-progress-measures-for-grokking.pdf
└── runs/
    ├── cpu/
    └── gpu/
```

The project uses a flat module layout, so no `__init__.py` is needed.

## Run it

Create an environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

CPU:

```bash
python main.py --device cpu --cpu-threads 12 --output-dir runs/cpu
```

GPU:

```bash
python main.py --device cuda --output-dir runs/gpu
```

Install a CUDA-enabled PyTorch build for the GPU command. The correct package depends on the local CUDA and driver setup; see the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

The defaults reproduce the 40,000-epoch experiment. A shorter smoke test can be run with `--epochs 300`.

## Papers and origin

I first heard about grokking through [The AI That Learned to Understand Long After It Stopped Trying](https://towardsdatascience.com/the-ai-that-learned-to-understand-long-after-it-stopped-trying) by Utkarsh Mangal, published in *Towards Data Science* on 28 September 2026.

That article points to the two papers included in [`papers/`](papers/):

- Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, and Vedant Misra. *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets*. arXiv:2201.02177, 2022. [PDF](papers/power-2022-grokking.pdf) | [arXiv](https://arxiv.org/abs/2201.02177)
- Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, and Jacob Steinhardt. *Progress Measures for Grokking via Mechanistic Interpretability*. ICLR 2023. arXiv:2301.05217. [PDF](papers/nanda-2023-progress-measures-for-grokking.pdf) | [arXiv](https://arxiv.org/abs/2301.05217)

The implementation follows the mainline modular-addition experiment from Nanda et al.
