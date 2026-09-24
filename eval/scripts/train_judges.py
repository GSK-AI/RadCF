"""Entry point for training judge classifiers used in CIG evaluation."""

import argparse
from pathlib import Path

from eval.config import EvalConfig
from eval.models.judges import train_gender_judge, train_view_judge, train_disease_judge_by_view, train_age_judge

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(EvalConfig.EVAL_DATASETS.keys()),
        help="Evaluation dataset to train judges on (train split will be used)",
    )
    parser.add_argument(
        "--train",
        choices=["gender", "view", "disease", "age", "all"],
        default="all",
        help="Which classifier(s) to train",
    )
    parser.add_argument(
        "--view", choices=["PA", "AP"], help="Train only PA or AP disease classifier"
    )
    args = parser.parse_args()

    config = EvalConfig()
    print(f"Training judges on {args.dataset} dataset")
    print(f"Available datasets: {', '.join(EvalConfig.EVAL_DATASETS.keys())}\n")

    if args.train in ["gender", "all"]:
        train_gender_judge(args.dataset)

    if args.train in ["view", "all"]:
        train_view_judge(args.dataset)

    if args.train in ["disease", "all"]:
        train_disease_judge_by_view(args.dataset, view=args.view)

    if args.train in ["age", "all"]:
        train_age_judge(args.dataset)
