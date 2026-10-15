import argparse
import gc
import math
import time
import sys
from pathlib import Path

import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Allow imports from the repository root.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.run_001.data import MODEL_NAME, load_data


RANKS = [1, 2, 4, 8, 16, 32, 64, 128]
DEFAULT_SEEDS = [42, 43, 44]
RESULT_COLUMNS = [
    "config",
    "rank",
    "seed",
    "trainable_params",
    "train_time_s",
    "val_ppl",
]


def evaluate_ppl(trainer):
    """Calculate validation perplexity from the evaluation loss."""
    loss = trainer.evaluate()["eval_loss"]
    return math.exp(loss) if loss < 80 else float("inf")


def make_trainer(
    model,
    train_data,
    valid_data,
    collator,
    output_dir,
    seed,
    max_steps,
    batch_size,
    learning_rate,
):
    args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        lr_scheduler_type="linear",
        warmup_steps=min(10, max_steps),
        eval_strategy="no",
        save_strategy="no",
        logging_strategy="no",
        report_to="none",
        seed=seed,
        data_seed=seed,
    )

    return Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        data_collator=collator,
    )


def run_one(
    config,
    rank,
    seed,
    train_data,
    valid_data,
    tokenizer,
    collator,
    output_root,
    max_steps,
    batch_size,
    lora_lr,
    full_lr,
):
    set_seed(seed)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.config.pad_token_id = tokenizer.pad_token_id

    if config == "lora":
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=rank,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=["c_attn"],
            ),
        )
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

    elif config == "full_ft":
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

    else:
        # Baseline: no parameters are updated.
        trainable_params = 0
        for param in model.parameters():
            param.requires_grad = False

    trainer = make_trainer(
        model=model,
        train_data=train_data,
        valid_data=valid_data,
        collator=collator,
        output_dir=output_root / f"{config}_rank_{rank}_seed_{seed}",
        seed=seed,
        max_steps=max_steps,
        batch_size=batch_size,
        learning_rate=full_lr if config == "full_ft" else lora_lr,
    )

    if config == "baseline":
        train_time_s = 0.0
        val_ppl = evaluate_ppl(trainer)
    else:
        start = time.perf_counter()
        trainer.train()
        train_time_s = time.perf_counter() - start
        val_ppl = evaluate_ppl(trainer)

    row = {
        "config": config,
        "rank": rank,
        "seed": seed,
        "trainable_params": trainable_params,
        "train_time_s": round(train_time_s, 3),
        "val_ppl": val_ppl,
    }

    del trainer, model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return row


def main():
    parser = argparse.ArgumentParser(
        description="Run the run_002 multi-seed fine-tuning sweep."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--ranks", type=int, nargs="+", default=RANKS)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lora-lr", type=float, default=2e-4)
    parser.add_argument("--full-lr", type=float, default=5e-5)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/run_002/results.csv",
    )
    args = parser.parse_args()

    if any(rank not in RANKS for rank in args.ranks):
        parser.error(f"Ranks must be selected from {RANKS}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_root = args.output.parent / "checkpoints"
    output_root.mkdir(parents=True, exist_ok=True)

    if args.output.exists():
        results = pd.read_csv(args.output)
    else:
        results = pd.DataFrame(columns=RESULT_COLUMNS)

    completed = set()
    if not results.empty:
        completed = {
            (str(row.config), int(row.seed), int(row["rank"]))
            for _, row in results.iterrows()
        }

    for seed in args.seeds:
        set_seed(seed)
        tokenizer, train_data, valid_data, collator = load_data(seed=seed)

        configs = [
            ("baseline", 0),
            ("full_ft", 0),
        ] + [("lora", rank) for rank in args.ranks]

        for config, rank in configs:
            key = (config, seed, rank)

            if key in completed:
                print(f"Skipping completed run: {key}")
                continue

            print(f"\nStarting config={config}, rank={rank}, seed={seed}")

            row = run_one(
                config=config,
                rank=rank,
                seed=seed,
                train_data=train_data,
                valid_data=valid_data,
                tokenizer=tokenizer,
                collator=collator,
                output_root=output_root,
                max_steps=args.max_steps,
                batch_size=args.batch_size,
                lora_lr=args.lora_lr,
                full_lr=args.full_lr,
            )

            results = pd.concat(
                [results, pd.DataFrame([row])],
                ignore_index=True,
            )
            results = results[RESULT_COLUMNS]
            results.to_csv(args.output, index=False)
            completed.add(key)

            print(f"Saved {args.output}: {row}")

        del tokenizer, train_data, valid_data, collator
        gc.collect()

    print(f"\nSweep finished. Results saved to {args.output}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()