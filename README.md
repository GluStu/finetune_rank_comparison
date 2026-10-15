Comparing LoRA Ranks for Language Model Fine-Tuning
Overview

This project compares full fine-tuning with Low-Rank Adaptation (LoRA) at different adapter ranks for DistilGPT-2 on WikiText-2.

The experiment investigates how LoRA rank affects validation perplexity, trainable parameter count, and training time. It evaluates ranks 1, 2, 4, 8, 16, 32, 64, and 128, alongside an unfine-tuned baseline and a full fine-tuning configuration.

The repository contains a single-run summary and a multi-seed sweep. The three-seed sweep is the primary source for comparing ranks because it provides an estimate of variability across runs.

What has been done
Loaded the pretrained DistilGPT-2 language model and WikiText-2 dataset.
Prepared training and validation subsets.
Evaluated the pretrained model without fine-tuning.
Ran full fine-tuning as a reference configuration.
Trained LoRA adapters at eight different ranks.
Recorded trainable parameter counts, training time, validation loss in the single-run summary, and validation perplexity.
Ran the multi-seed sweep with seeds 42, 43, and 44.
Saved experiment results to CSV files for further analysis.
Experimental setup

The experiment uses DistilGPT-2 and the Salesforce/wikitext dataset with the wikitext-2-raw-v1 configuration.

The notebook's documented configuration includes 1,000 training examples, 200 validation examples, 100 training steps per trained configuration, a per-device batch size of 4, a maximum sequence length of 128 tokens, and a learning rate of 0.0002.

The LoRA configurations target the c_attn module, use alpha 16 and dropout 0.1, and vary the adapter rank from 1 to 128.

The multi-seed sweep uses random seeds 42, 43, and 44. The baseline does not update model parameters. Full fine-tuning updates the model's trainable parameters, while LoRA freezes the pretrained weights and trains the adapter parameters.

Results: Three-seed sweep

The table below summarizes multi_sweep/run002/results.csv. Perplexity and training time are reported as the mean and sample standard deviation across seeds 42, 43, and 44.

Configuration	Trainable parameters	Mean training time (s)	Mean validation perplexity
Baseline	0	0.000 ± 0.000	76.483 ± 0.000
Full fine-tuning	81,912,576	14.450 ± 0.109	41.554 ± 0.232
LoRA rank 1	18,432	9.400 ± 0.183	61.821 ± 0.246
LoRA rank 2	36,864	9.426 ± 0.059	61.645 ± 0.777
LoRA rank 4	73,728	9.409 ± 0.169	61.145 ± 0.455
LoRA rank 8	147,456	9.377 ± 0.092	61.232 ± 0.515
LoRA rank 16	294,912	9.533 ± 0.124	61.366 ± 0.459
LoRA rank 32	589,824	9.638 ± 0.102	61.323 ± 0.365
LoRA rank 64	1,179,648	9.613 ± 0.029	61.377 ± 0.315
LoRA rank 128	2,359,296	9.817 ± 0.061	61.312 ± 0.393

Interpretation: Rank 4 has the lowest mean validation perplexity among the tested LoRA configurations. However, the differences between ranks are relatively small compared with their run-to-run variation, and only three seeds were evaluated. These results do not establish that rank 4 is universally optimal.

The baseline perplexity is identical across the three seeds because the pretrained baseline is evaluated without training. Its zero standard deviation does not represent variability across independently trained models.

Results by seed

The following table provides the individual validation perplexities used to calculate the aggregate results.

Configuration	Seed 42	Seed 43	Seed 44
Baseline	76.4833	76.4833	76.4833
Full fine-tuning	41.7262	41.6461	41.2908
LoRA rank 1	61.7824	61.5967	62.0843
LoRA rank 2	62.5369	61.2820	61.1160
LoRA rank 4	61.1949	61.5729	60.6665
LoRA rank 8	61.8245	60.9812	60.8896
LoRA rank 16	61.8859	61.1946	61.0162
LoRA rank 32	61.6891	60.9598	61.3200
LoRA rank 64	61.7390	61.1713	61.2203
LoRA rank 128	61.7665	61.0725	61.0985

Rank 4 produces the best LoRA perplexity for seeds 42 and 44. Rank 32 produces the best LoRA perplexity for seed 43. This variation reinforces the need to report aggregated results rather than selecting a rank based on one run.

Results: Single-run summary

summary.csv contains a separate set of reported measurements. These results should not be combined with the multi-seed sweep as if they were identical runs.

Configuration	Trainable parameters	Training time (s)	Evaluation loss	Perplexity
Baseline (no FT)	0	0.00	4.337072	76.483297
Full fine-tune	81,912,576	14.57	3.731128	41.726168
LoRA rank 1	18,432	9.45	4.123618	61.782375
LoRA rank 2	36,864	9.36	4.135757	62.536912
LoRA rank 4	73,728	9.50	4.114063	61.194864
LoRA rank 8	147,456	9.45	4.124300	61.824517
LoRA rank 16	294,912	9.49	4.125293	61.885925
LoRA rank 32	589,824	9.54	4.122107	61.689057
LoRA rank 64	1,179,648	9.55	4.122916	61.738995
LoRA rank 128	2,359,296	9.75	4.123362	61.766498

These values are retained to document the single-run experiment. The multi-seed results above should be preferred for conclusions about performance across random seeds.

Key findings
1. Full fine-tuning achieves the best predictive performance

The multi-seed sweep reports a mean validation perplexity of 41.554 for full fine-tuning, compared with 61.145 for the best mean LoRA configuration and 76.483 for the baseline.

2. LoRA substantially reduces trainable parameters

Rank 4 trains 73,728 parameters, compared with 81,912,576 for full fine-tuning. That is approximately 1,111 times fewer trainable parameters.

This is a parameter-count comparison. It does not imply an equivalent reduction in total memory usage or training time.

3. Increasing rank does not guarantee lower perplexity

The results do not show a monotonic improvement as rank increases. Rank 4 has the lowest mean perplexity in this sweep, while rank 32 wins one of the three individual seeds.

4. Training time is broadly similar across LoRA ranks

Mean LoRA training time ranges from approximately 9.377 to 9.817 seconds in this sweep. The highest rank takes the longest on average, but the differences are small and may depend on the execution environment.

Full fine-tuning takes approximately 14.450 seconds on average in the reported sweep.

Data consistency and reporting notes

The single-run summary and multi_sweep/run002/results.csv are not identical in every field.

The full fine-tuning training time is 14.57 seconds in summary.csv versus 14.464 seconds for seed 42 in the multi-seed sweep.
LoRA training times also differ between the two files, indicating that the single-run summary should be treated as a separate measurement.
The rank-64 perplexity differs slightly between the files: 61.738995 in the summary and 61.738966 for seed 42 in the sweep.
The summary includes evaluation loss; the multi-seed CSV contains validation perplexity but does not include a loss column.

These differences should be preserved as recorded rather than silently overwriting one file with values from the other. When regenerating the results, ensure each CSV is produced from the same run as its associated metadata.

Limitations
The multi-seed sweep includes only three seeds.
The experiment uses a relatively small training and validation subset.
Only one base model, dataset, learning rate, and LoRA target module are evaluated.
Training time depends on the hardware, software stack, and runtime conditions.
The small observed differences between LoRA ranks may not be reproducible.
Peak memory usage, inference latency, and throughput are not included in the provided CSV results.

The results are best interpreted as a small comparative experiment rather than a definitive benchmark for choosing LoRA rank.

How to run
Requirements

Use Python and an environment compatible with PyTorch, Hugging Face Transformers, Datasets, PEFT, Accelerate, Pandas, Matplotlib, and Jupyter. The notebook also specifies torchao==0.18.0; use compatible package versions for your hardware and Python environment.

1. Clone the repository
git clone https://github.com/GluStu/finetune_rank_comparison.git
cd finetune_rank_comparison

2. Create a virtual environment
python -m venv .venv


Activate it on macOS or Linux:

source .venv/bin/activate


On Windows PowerShell:

.venv\Scripts\Activate.ps1

3. Install dependencies
python -m pip install --upgrade pip
python -m pip install torch transformers datasets peft accelerate pandas matplotlib jupyter
python -m pip install "torchao==0.18.0"


If the PyTorch/TorchAO combination is incompatible with your Python version or hardware, install a compatible PyTorch build and its matching dependencies.

4. Run the notebook
jupyter notebook lora_rank_comp.ipynb


Run the notebook cells in order. The experiment downloads the required model and dataset, evaluates the baseline, performs full fine-tuning, and trains the LoRA configurations.

A GPU is recommended, although runtime depends on the available hardware. An internet connection is required for the initial model and dataset downloads.

5. Inspect the results

Review the CSV files produced by the experiment. The repository's current results include:

summary.csv — single-run results, including evaluation loss.
multi_sweep/run002/results.csv — measurements for seeds 42, 43, and 44.

The exact output paths depend on the notebook's saving configuration. Check the relevant notebook cells if the generated files appear elsewhere.

Future work
Run more random seeds and report uncertainty estimates.
Add peak memory usage, throughput, and inference latency.
Compare ranks under matched compute budgets.
Test additional datasets, models, and LoRA target modules.
Automate CSV generation and validation to prevent discrepancies between summaries.
Record the exact hardware, package versions, and configuration for every run.
References
LoRA: Low-Rank Adaptation of Large Language Models
WikiText dataset
DistilGPT-2
