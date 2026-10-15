import math


def evaluate_model(trainer):
    """Return validation loss and perplexity."""

    metrics = trainer.evaluate()
    eval_loss = metrics["eval_loss"]

    perplexity = (
        math.exp(eval_loss)
        if eval_loss < 80
        else float("inf")
    )

    return {
        "eval_loss": eval_loss,
        "perplexity": perplexity,
    }