# Comparing LoRA Ranks for Language Model Fine-Tuning

This project investigates how the rank of a Low-Rank Adaptation (LoRA) adapter affects language model fine-tuning performance, training efficiency, and the number of trainable parameters.

LoRA is a parameter-efficient fine-tuning technique that freezes the original model weights and learns a low-rank update to adapt the model to a specific task. The rank determines the capacity of this update and the number of trainable parameters.

The experiment is a small comparison of full fine-tuning vs. LoRA at different adapter ranks, using **DistilGPT-2** on **WikiText-2**. It measures how LoRA rank affects **validation perplexity**, **trainable parameter count**, and **training time**. It evaluates ranks 1, 2, 4, 8, 16, 32, 64, and 128, alongside an unfine-tuned baseline and a full fine-tuning reference.

The repository contains a single-run summary and a three-seed sweep. **The three-seed sweep is the primary source for comparing ranks**, since it gives an estimate of run-to-run variability.

## Table of Contents

- [Hypotheses](#hypotheses)
- [Key Findings](#key-findings)
- [Experimental Setup](#experimental-setup)
- [Results: Three-Seed Sweep](#results-three-seed-sweep)
- [Results by Seed](#results-by-seed)
- [Results: Single-Run Summary](#results-single-run-summary)
- [Data Consistency Notes](#data-consistency-notes)
- [Conclusion](#conclusion)
- [Limitations](#limitations)
- [Repository Contents](#repository-contents)
- [How to Run](#how-to-run)
- [Future Work](#future-work)
- [References](#references)

## Hypotheses

Increasing LoRA rank increases the adapter's capacity and its number of trainable parameters, but higher ranks may not consistently improve validation performance.

1. **LoRA improves performance:** LoRA fine-tuning reduces validation perplexity compared with the unfine-tuned baseline.
2. **Rank affects model quality:** different LoRA ranks produce different validation performance.
3. **Higher rank does not guarantee better results:** increasing rank eventually gives diminishing returns, and some lower ranks may match or beat higher ones.
4. **LoRA is parameter-efficient:** LoRA needs substantially fewer trainable parameters than full fine-tuning.

**Outcome (three-seed sweep):** hypotheses 1, 3, and 4 are supported. Hypothesis 2 is only weakly supported: LoRA ranks differ by less than their run-to-run variation, so a rank effect cannot be confirmed from this data.

## Key Findings

1. **Full fine-tuning achieves the best predictive performance.** Mean validation perplexity is 41.554, compared with 61.145 for the best LoRA configuration and 76.483 for the baseline.
2. **LoRA drastically reduces trainable parameters.** Rank 4 trains 73,728 parameters versus 81,912,576 for full fine-tuning, roughly **1,111× fewer**. This is a parameter-count comparison only; it does not imply an equivalent reduction in total memory usage or training time.
3. **Higher rank does not guarantee lower perplexity.** There is no monotonic improvement with rank. Rank 4 has the lowest mean perplexity, while rank 32 wins one of the three individual seeds.
4. **Training time is broadly similar across LoRA ranks.** Mean LoRA training time ranges from about 9.38 s to 9.82 s. The highest rank is slowest on average, but the differences are small and may depend on the execution environment. Full fine-tuning averages about 14.45 s.

> **Caveat:** Differences between LoRA ranks are small relative to run-to-run variation, and only three seeds were evaluated. These results do not establish that rank 4 is universally optimal.

## Experimental Setup

| Setting | Value |
|---|---|
| Base model | DistilGPT-2 |
| Dataset | `Salesforce/wikitext`, config `wikitext-2-raw-v1` |
| Training examples | 1,000 |
| Validation examples | 200 |
| Training steps (per trained configuration) | 100 |
| Per-device batch size | 4 |
| Max sequence length | 128 tokens |
| Learning rate | 0.0002 |
| LoRA target module | `c_attn` |
| LoRA alpha / dropout | 16 / 0.1 |
| LoRA ranks | 1, 2, 4, 8, 16, 32, 64, 128 |
| Seeds (multi-seed sweep) | 42, 43, 44 |

**Evaluation metrics**

- **Trainable parameters:** number of parameters updated during fine-tuning.
- **Training time:** time to complete training.
- **Validation perplexity:** how well the model predicts unseen text (lower is better).

**Configurations**

- **Baseline:** pretrained model evaluated without any parameter updates.
- **Full fine-tuning:** updates all of the model's trainable parameters.
- **LoRA:** freezes the pretrained weights and trains only the adapter parameters.

Trainable parameters scale linearly with rank: each unit of rank adds 18,432 parameters.

**What was done**

- Loaded the pretrained DistilGPT-2 model and the WikiText-2 dataset.
- Prepared training and validation subsets.
- Evaluated the pretrained baseline.
- Ran full fine-tuning as a reference.
- Trained LoRA adapters at eight ranks.
- Recorded trainable parameters, training time, validation loss (single-run summary), and validation perplexity.
- Ran the multi-seed sweep (seeds 42, 43, 44).
- Saved results to CSV files.

## Results: Three-Seed Sweep

Source: `multi_sweep/run002/results.csv`. Training time and perplexity are reported as **mean ± sample standard deviation** across seeds 42, 43, and 44.

| Configuration | Trainable parameters | Mean training time (s) | Mean validation perplexity |
|---|---:|---:|---:|
| Baseline | 0 | 0.000 ± 0.000 | 76.483 ± 0.000 |
| Full fine-tuning | 81,912,576 | 14.450 ± 0.109 | **41.554 ± 0.232** |
| LoRA rank 1 | 18,432 | 9.400 ± 0.183 | 61.821 ± 0.246 |
| LoRA rank 2 | 36,864 | 9.426 ± 0.059 | 61.645 ± 0.777 |
| LoRA rank 4 | 73,728 | 9.409 ± 0.169 | **61.145 ± 0.455** |
| LoRA rank 8 | 147,456 | 9.377 ± 0.092 | 61.232 ± 0.515 |
| LoRA rank 16 | 294,912 | 9.533 ± 0.124 | 61.366 ± 0.459 |
| LoRA rank 32 | 589,824 | 9.638 ± 0.102 | 61.323 ± 0.365 |
| LoRA rank 64 | 1,179,648 | 9.613 ± 0.029 | 61.377 ± 0.315 |
| LoRA rank 128 | 2,359,296 | 9.817 ± 0.061 | 61.312 ± 0.393 |

**Interpretation.** Among LoRA configurations, rank 4 has the lowest mean validation perplexity. However, the gaps between ranks are small compared with their run-to-run variation, and only three seeds were evaluated.

The baseline's zero standard deviation reflects that the pretrained model is evaluated without training. It does **not** represent variability across independently trained models.

## Results by Seed

Individual validation perplexities used to compute the aggregates above.

| Configuration | Seed 42 | Seed 43 | Seed 44 |
|---|---:|---:|---:|
| Baseline | 76.4833 | 76.4833 | 76.4833 |
| Full fine-tuning | 41.7262 | 41.6461 | 41.2908 |
| LoRA rank 1 | 61.7824 | 61.5967 | 62.0843 |
| LoRA rank 2 | 62.5369 | 61.2820 | 61.1160 |
| LoRA rank 4 | 61.1949 | 61.5729 | 60.6665 |
| LoRA rank 8 | 61.8245 | 60.9812 | 60.8896 |
| LoRA rank 16 | 61.8859 | 61.1946 | 61.0162 |
| LoRA rank 32 | 61.6891 | 60.9598 | 61.3200 |
| LoRA rank 64 | 61.7390 | 61.1713 | 61.2203 |
| LoRA rank 128 | 61.7665 | 61.0725 | 61.0985 |

Rank 4 gives the best LoRA perplexity for seeds 42 and 44; rank 32 is best for seed 43. This variation is why results should be aggregated across seeds rather than choosing a rank from a single run.

## Results: Single-Run Summary

Source: `summary.csv`. This is a **separate set of measurements** and should not be combined with the multi-seed sweep as if they were the same runs.

| Configuration | Trainable parameters | Training time (s) | Evaluation loss | Perplexity |
|---|---:|---:|---:|---:|
| Baseline (no FT) | 0 | 0.00 | 4.337072 | 76.483297 |
| Full fine-tune | 81,912,576 | 14.57 | 3.731128 | 41.726168 |
| LoRA rank 1 | 18,432 | 9.45 | 4.123618 | 61.782375 |
| LoRA rank 2 | 36,864 | 9.36 | 4.135757 | 62.536912 |
| LoRA rank 4 | 73,728 | 9.50 | 4.114063 | 61.194864 |
| LoRA rank 8 | 147,456 | 9.45 | 4.124300 | 61.824517 |
| LoRA rank 16 | 294,912 | 9.49 | 4.125293 | 61.885925 |
| LoRA rank 32 | 589,824 | 9.54 | 4.122107 | 61.689057 |
| LoRA rank 64 | 1,179,648 | 9.55 | 4.122916 | 61.738995 |
| LoRA rank 128 | 2,359,296 | 9.75 | 4.123362 | 61.766498 |

These values are kept to document the single-run experiment. Prefer the multi-seed results for conclusions about performance across random seeds.

## Data Consistency Notes

> **Repository status:** the public repository currently contains `lora_rank_comp.ipynb`, `lora_rank_results.csv`, and `README.md`. The files `summary.csv` and `multi_sweep/run002/results.csv` referenced in this document are **not yet in the repository**, and `lora_rank_results.csv` is a third, separate set of single-run values that does not match `summary.csv` (e.g. full fine-tuning perplexity of about 41.66 vs. 41.726). Commit the missing files or update this document to reference the files that are actually published.

`summary.csv` and `multi_sweep/run002/results.csv` are not identical in every field:

- Full fine-tuning training time is **14.57 s** in `summary.csv` versus **14.464 s** for seed 42 in the sweep.
- LoRA training times also differ between the two files, so the single-run summary should be treated as an independent measurement.
- Rank-64 perplexity differs slightly: **61.738995** in the summary versus **61.738966** for seed 42 in the sweep.
- The summary includes evaluation loss; the multi-seed CSV contains validation perplexity only (no loss column).

These differences are preserved as recorded rather than overwritten. When regenerating results, make sure each CSV is produced from the same run as its associated metadata.

## Conclusion

LoRA can substantially improve language model performance (validation perplexity 76.5 → about 61) while training only a tiny fraction of the parameters. Raising the adapter rank does not reliably improve results: rank 4 has the lowest mean perplexity using just 73,728 trainable parameters, but it is statistically hard to separate from the other ranks. This suggests a small adapter offers a good balance of quality and parameter efficiency for this setup.

Full fine-tuning reaches clearly better perplexity (41.6) than any tested LoRA rank, at the cost of updating about 81.9 million parameters.

## Limitations

- Only three seeds in the multi-seed sweep.
- Relatively small training and validation subsets.
- Only one base model, dataset, learning rate, and LoRA target module.
- Training time depends on hardware, software stack, and runtime conditions.
- Small differences between LoRA ranks may not be reproducible.
- Peak memory usage, inference latency, and throughput are not included in the CSV results.

Treat these results as a small comparative experiment, not a definitive benchmark for choosing LoRA rank.

## Repository Contents

| File | Description |
|---|---|
| [`lora_rank_comp.ipynb`](https://github.com/GluStu/finetune_rank_comparison/blob/main/lora_rank_comp.ipynb) | Notebook with the experimental setup, training procedures, and evaluation |
| [`lora_rank_results.csv`](https://github.com/GluStu/finetune_rank_comparison/blob/main/lora_rank_results.csv) | Single-run results table currently in the repository |
| `summary.csv` | Single-run summary with evaluation loss (documented above; not yet in the repo) |
| `multi_sweep/run002/results.csv` | Three-seed sweep results (documented above; not yet in the repo) |

## How to Run

### Requirements

Python with an environment compatible with PyTorch, Hugging Face Transformers, Datasets, PEFT, Accelerate, Pandas, Matplotlib, and Jupyter. The notebook also specifies `torchao==0.18.0`; use package versions compatible with your hardware and Python version.

A GPU is recommended (runtime depends on hardware). An internet connection is required for the initial model and dataset downloads.

### 1. Clone the repository

```bash
git clone https://github.com/GluStu/finetune_rank_comparison.git
cd finetune_rank_comparison
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
# macOS / Linux
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install torch transformers datasets peft accelerate pandas matplotlib jupyter
python -m pip install "torchao==0.18.0"
```

If the PyTorch/TorchAO combination is incompatible with your Python version or hardware, install a compatible PyTorch build and its matching dependencies.

### 4. Run the notebook

```bash
jupyter notebook lora_rank_comp.ipynb
```

Run the cells in order. The notebook downloads the model and dataset, evaluates the baseline, performs full fine-tuning, and trains the LoRA configurations.

### 5. Inspect the results

The repository's current results include:

| File | Contents |
|---|---|
| `lora_rank_results.csv` | Single-run results currently in the repository |
| `summary.csv` | Single-run results, including evaluation loss |
| `multi_sweep/run002/results.csv` | Measurements for seeds 42, 43, and 44 |

Exact output paths depend on the notebook's saving configuration. Check the relevant notebook cells if files appear elsewhere.

## Future Work

- Run more random seeds and report uncertainty estimates.
- Investigate whether the best rank changes with dataset size or task complexity.
- Test different learning rates and LoRA scaling factors.
- Add peak memory usage, throughput, and inference latency.
- Compare ranks under matched compute budgets.
- Test additional datasets, models, and LoRA target modules.
- Automate CSV generation and validation to prevent discrepancies between summaries.
- Record exact hardware, package versions, and configuration for every run.

## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [WikiText dataset](https://huggingface.co/datasets/Salesforce/wikitext)
- [DistilGPT-2](https://huggingface.co/distilbert/distilgpt2)
