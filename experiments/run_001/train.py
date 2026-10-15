import argparse
import gc
import time

import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)

from data import MODEL_NAME, SEED, load_data
from evaluate import evaluate_model


def count_trainable_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def train_experiment(
    model,
    train_data,
    valid_data,
    collator,
    output_dir,
    max_steps,
    batch_size,
    learning_rate,
    seed,
):
    args = TrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        lr_scheduler_type="linear",
        warmup_steps=10,
        eval_strategy="no",
        save_strategy="no",
        logging_strategy="no",
        report_to="none",
        seed=seed,
        data_seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        data_collator=collator,
    )

    start = time.perf_counter()
    trainer.train()
    train_seconds = time.perf_counter() - start

    metrics = evaluate_model(trainer)

    result = {
        "trainable_params": count_trainable_parameters(model),
        "train_seconds": round(train_seconds, 2),
        **metrics,
    }

    return trainer, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["full", "lora", "all", "baseline"],
        default="all",
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128],
    )
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip-full-ft", action="store_true")
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--save-adapters", action="store_true")
    cli = parser.parse_args()

    set_seed(cli.seed)

    tokenizer, train_data, valid_data, collator = load_data(
        seed=cli.seed
    )

    results = []

    # Baseline: evaluate the pretrained model without training.
    if cli.mode in ("baseline", "all"):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        model.config.pad_token_id = tokenizer.pad_token_id

        args = TrainingArguments(
            output_dir=f"{cli.output_dir}/baseline",
            per_device_eval_batch_size=cli.batch_size,
            report_to="none",
            seed=cli.seed,
        )

        trainer = Trainer(
            model=model,
            args=args,
            eval_dataset=valid_data,
            data_collator=collator,
        )

        results.append({
            "rank": "baseline (no FT)",
            "trainable_params": 0,
            "train_seconds": 0.0,
            **evaluate_model(trainer),
        })

        del trainer, model
        gc.collect()

    # Full fine-tuning.
    run_full = (
        cli.mode == "full"
        or (
            cli.mode == "all"
            and not cli.skip_full_ft
        )
    )

    if run_full:
        set_seed(cli.seed)

        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        model.config.pad_token_id = tokenizer.pad_token_id

        trainer, result = train_experiment(
            model=model,
            train_data=train_data,
            valid_data=valid_data,
            collator=collator,
            output_dir=f"{cli.output_dir}/full_ft",
            max_steps=cli.max_steps,
            batch_size=cli.batch_size,
            learning_rate=5e-5,
            seed=cli.seed,
        )

        results.append({"rank": "full fine-tune", **result})

        del trainer, model
        gc.collect()

    # LoRA rank comparison.
    if cli.mode in ("lora", "all"):
        for rank in cli.ranks:
            set_seed(cli.seed)

            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
            model.config.pad_token_id = tokenizer.pad_token_id

            config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=rank,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=["c_attn"],
            )

            model = get_peft_model(model, config)
            model.print_trainable_parameters()

            trainer, result = train_experiment(
                model=model,
                train_data=train_data,
                valid_data=valid_data,
                collator=collator,
                output_dir=f"{cli.output_dir}/rank_{rank}",
                max_steps=cli.max_steps,
                batch_size=cli.batch_size,
                learning_rate=cli.learning_rate,
                seed=cli.seed,
            )

            results.append({"rank": rank, **result})

            if cli.save_adapters:
                model.save_pretrained(f"./adapters/rank_{rank}")

            del trainer, model
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    df = pd.DataFrame(results)

    print("\nExperiment results:")
    print(df.to_string(index=False))

    df.to_csv(f"{cli.output_dir}/summary.csv", index=False)


if __name__ == "__main__":
    main()