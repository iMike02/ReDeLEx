from argparse import ArgumentParser

from relbench.datasets import get_dataset
from relbench.tasks import get_task


def main() -> None:
    parser = ArgumentParser(
        description="Pre-download the RelBench Amazon dataset and optional task cache."
    )
    parser.add_argument("--dataset", default="rel-amazon")
    parser.add_argument(
        "--task",
        default="item-ltv",
        help="Task to pre-download. Pass an empty string to skip task download.",
    )
    args = parser.parse_args()

    print(f"Downloading dataset cache for {args.dataset}...")
    dataset = get_dataset(args.dataset, download=True)
    dataset.get_db()
    print("Dataset cache ready.")

    task_name = args.task.strip()
    if task_name:
        print(f"Downloading task cache for {args.dataset}/{task_name}...")
        task = get_task(args.dataset, task_name, download=True)
        task.get_table("train")
        print("Task cache ready.")


if __name__ == "__main__":
    main()
