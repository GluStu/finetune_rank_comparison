from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = ROOT / "experiments/run_002/results.csv"
DEFAULT_OUTPUT = ROOT / "results/figures"


def load_results(csv_path: Path) -> pd.DataFrame:
    """Load results and validate required columns."""
    df = pd.read_csv(csv_path)

    required = {
        "config",
        "rank",
        "seed",
        "trainable_params",
        "train_time_s",
        "val_ppl",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

    df["rank"] = pd.to_numeric(df["rank"], errors="raise")
    df["val_ppl"] = pd.to_numeric(df["val_ppl"], errors="raise")
    df["trainable_params"] = pd.to_numeric(
        df["trainable_params"], errors="raise"
    )

    return df


def plot_perplexity(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot mean validation perplexity +/- standard deviation by rank."""
    fig, ax = plt.subplots(figsize=(9, 6))

    lora = df[df["config"] == "lora"].copy()
    summary = (
        lora.groupby("rank")["val_ppl"]
        .agg(["mean", "std"])
        .sort_index()
    )

    ax.errorbar(
        summary.index,
        summary["mean"],
        yerr=summary["std"].fillna(0),
        fmt="o-",
        capsize=5,
        linewidth=2,
        markersize=6,
        label="LoRA (mean ± std)",
    )

    # Reference lines use the mean across available seeds.
    reference_styles = {
        "baseline": ("Baseline", "--"),
        "full_ft": ("Full fine-tuning", ":"),
    }

    for config, (label, linestyle) in reference_styles.items():
        values = df.loc[df["config"] == config, "val_ppl"].dropna()

        if not values.empty:
            mean_ppl = values.mean()
            ax.axhline(
                mean_ppl,
                linestyle=linestyle,
                linewidth=2,
                label=f"{label} (mean={mean_ppl:.2f})",
            )

    ax.set_xscale("log", base=2)
    ax.set_xticks(summary.index)
    ax.set_xticklabels([str(rank) for rank in summary.index])
    ax.set_xlabel("LoRA rank")
    ax.set_ylabel("Validation perplexity (lower is better)")
    ax.set_title("Validation Perplexity vs. LoRA Rank")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    fig.savefig(output_dir / "perplexity_vs_rank.png", dpi=300)
    fig.savefig(output_dir / "perplexity_vs_rank.pdf")
    plt.close(fig)


def plot_parameters(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot trainable parameter count against LoRA rank."""
    fig, ax = plt.subplots(figsize=(9, 6))

    lora = df[df["config"] == "lora"].copy()

    # Parameter counts should be identical across seeds for a given rank.
    summary = (
        lora.groupby("rank")["trainable_params"]
        .agg(["mean", "std"])
        .sort_index()
    )

    ax.plot(
        summary.index,
        summary["mean"],
        marker="o",
        linewidth=2,
        markersize=6,
        label="LoRA",
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks(summary.index)
    ax.set_xticklabels([str(rank) for rank in summary.index])
    ax.set_xlabel("LoRA rank")
    ax.set_ylabel("Trainable parameters")
    ax.set_title("Trainable Parameters vs. LoRA Rank")
    ax.grid(True, which="both", alpha=0.25)

    # Add full fine-tuning as a parameter-count reference.
    full_ft = df.loc[
        df["config"] == "full_ft", "trainable_params"
    ].dropna()

    if not full_ft.empty:
        ax.axhline(
            full_ft.mean(),
            linestyle=":",
            linewidth=2,
            label=f"Full fine-tuning ({full_ft.mean():,.0f})",
        )

    ax.legend()
    fig.tight_layout()

    fig.savefig(output_dir / "trainable_params_vs_rank.png", dpi=300)
    fig.savefig(output_dir / "trainable_params_vs_rank.pdf")
    plt.close(fig)


def main() -> None:
    output_dir = DEFAULT_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(DEFAULT_CSV)

    plot_perplexity(df, output_dir)
    plot_parameters(df, output_dir)

    print(f"Figures saved to {output_dir}")
    print(f"Results loaded from {DEFAULT_CSV}")


if __name__ == "__main__":
    main()