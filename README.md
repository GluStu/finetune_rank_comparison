# Comparing LoRA Ranks for Language Model Fine-Tuning

## Overview

This project investigates how the rank of a Low-Rank Adaptation (LoRA) adapter affects language model fine-tuning performance, training efficiency, and the number of trainable parameters.

LoRA is a parameter-efficient fine-tuning technique that freezes the original model weights and learns a low-rank update to adapt the model to a specific task. The rank determines the capacity of this update and influences the number of trainable parameters.

The experiment compares multiple LoRA ranks against an unfine-tuned baseline and full fine-tuning, using DistilGPT-2 on the WikiText-2 dataset.

## Hypotheses

Increasing the LoRA rank increases the adapter's representational capacity and the number of trainable parameters. However, higher ranks may not consistently improve validation performance.

1. **LoRA improves performance:** Fine-tuning with LoRA will reduce validation perplexity compared with the unfine-tuned baseline.
2. **Rank affects model quality:** Different LoRA ranks will produce different validation performance.
3. **Higher rank does not guarantee better results:** Increasing rank will eventually produce diminishing returns, and some lower ranks may perform as well as or better than higher ranks.
4. **LoRA is parameter-efficient:** LoRA will require substantially fewer trainable parameters than full fine-tuning.

## Experimental Setup

| Item | Value |
|---|---|
| Model | DistilGPT-2 |
| Dataset | WikiText-2 |
| Training subset | 1,000 examples |
| Training steps | 100 per LoRA configuration |

### Configurations

- Unfine-tuned DistilGPT-2 baseline
- Full fine-tuning
- LoRA fine-tuning with ranks of 1, 2, 4, 8, 16, 32, 64, and 128

The original model weights remain frozen during LoRA fine-tuning while the adapter parameters are trained. Full fine-tuning updates the model's ordinary trainable weights.

### Evaluation Metrics

- **Trainable parameters:** the number of parameters updated during fine-tuning.
- **Training time:** the time required to complete training.
- **Validation perplexity:** a measure of how well the model predicts unseen text. Lower is better.

### Procedure

1. Load the pretrained DistilGPT-2 model and prepare the WikiText-2 dataset.
2. Evaluate the unfine-tuned model to establish a baseline.
3. Fine-tune the model using full fine-tuning.
4. Train separate LoRA configurations for each rank.
5. Record trainable parameter counts, training time, and validation perplexity.
6. Compare the results to assess the relationship between LoRA rank, model quality, and computational efficiency.

## Results

| Configuration | Trainable Parameters | Validation Perplexity |
|---|---:|---:|
| Unfine-tuned baseline | 0 | 76.48 |
| Full fine-tuning | 81.9 million | **41.66** |
| LoRA, rank 1 | 18,432 | 61.75 |
| LoRA, rank 2 | 36,864 | 62.50 |
| LoRA, rank 4 | 73,728 | **61.24** |
| LoRA, rank 8 | 147,456 | 61.90 |
| LoRA, rank 16 | 294,912 | 62.05 |
| LoRA, rank 32 | 589,824 | 61.85 |
| LoRA, rank 64 | 1.18 million | 61.93 |
| LoRA, rank 128 | 2.36 million | 61.92 |

Bold marks the best result overall (full fine-tuning) and the best among LoRA configurations (rank 4).

## Key Observations

- **LoRA improves predictive performance.** Rank 4 reduces validation perplexity from 76.48 to 61.24 relative to the unfine-tuned baseline.
- **Increasing rank does not consistently improve performance.** Rank 4 achieves a lower validation perplexity than all of the higher-rank configurations.
- **LoRA substantially reduces the number of trainable parameters.** Rank 4 trains 73,728 parameters, compared with approximately 81.9 million for full fine-tuning.
- **Trainable parameters scale linearly with rank.** Each unit of rank adds 18,432 trainable parameters.
- **Full fine-tuning achieves the lowest validation perplexity in this experiment.** The improvement comes with a substantially larger number of trainable parameters.

## Conclusion

The results suggest that LoRA can improve language model performance while substantially reducing the number of trainable parameters. However, increasing the adapter rank does not necessarily produce better validation performance.

Among the reported LoRA configurations, rank 4 achieves the lowest validation perplexity while using only 73,728 trainable parameters. This suggests that a relatively small adapter can provide a useful balance between model quality and parameter efficiency for this experimental setup.

Full fine-tuning achieves better validation perplexity than the tested LoRA configurations, but requires updating considerably more parameters.

### Limitations

These findings are preliminary. The experiment uses a small training subset and a limited number of training steps, and each configuration was run once, so the results should not be generalized to all models, datasets, or fine-tuning tasks. The LoRA perplexities fall in a narrow range (roughly 61.2 to 62.5), so small differences between ranks may be within run-to-run noise. Repeated runs, broader datasets, and hyperparameter tuning would help establish whether the observed differences are consistent.

## Future Work

- Repeat each configuration with multiple random seeds.
- Compare training and inference memory consumption.
- Measure training time systematically across all ranks.
- Test different learning rates and LoRA scaling factors.
- Evaluate the configurations on larger datasets and additional models.
- Investigate whether rank selection changes with dataset size or task complexity.

## Repository Contents

- [`lora_rank_comp.ipynb`](lora_rank_comp.ipynb): notebook containing the experimental setup, training procedures, and evaluation results.

## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [WikiText Language Modeling Dataset](https://huggingface.co/datasets/Salesforce/wikitext)
- [DistilGPT-2 on Hugging Face](https://huggingface.co/distilbert/distilgpt2)
