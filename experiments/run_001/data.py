from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

MODEL_NAME = "distilgpt2"
SEED = 42
MAX_LENGTH = 128
TRAIN_SIZE = 1000
VALIDATION_SIZE = 200


def load_data(
    model_name=MODEL_NAME,
    seed=SEED,
    max_length=MAX_LENGTH,
):
    """Load, select, and tokenize WikiText-2."""

    raw = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    train_raw = (
        raw["train"]
        .filter(lambda x: len(x["text"].strip()) > 0)
        .shuffle(seed=seed)
        .select(range(TRAIN_SIZE))
    )

    valid_raw = (
        raw["validation"]
        .filter(lambda x: len(x["text"].strip()) > 0)
        .select(range(VALIDATION_SIZE))
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    train_data = train_raw.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
    )

    valid_data = valid_raw.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
    )

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    return tokenizer, train_data, valid_data, collator