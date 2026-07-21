"""Push the FamBench dataset to the Hugging Face Hub.

Run this yourself with your own token -- FamBench never handles credentials for you:

    huggingface-cli login          # once, stores your token locally
    python scripts/push_to_hub.py --repo <your-namespace>/fambench

This uploads the two config splits (`redundancy`, `temporal`) and the dataset card.
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. yourname/fambench")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", exist_ok=True, private=args.private)

    # Upload the parquet configs under a directory each so the loader exposes them as
    # named configs (load_dataset(repo, "redundancy") / (repo, "temporal")).
    for name in ("redundancy", "temporal"):
        api.upload_file(
            path_or_fileobj=str(ROOT / "data" / f"{name}.parquet"),
            path_in_repo=f"{name}/test-00000-of-00001.parquet",
            repo_id=args.repo,
            repo_type="dataset",
        )
    api.upload_file(
        path_or_fileobj=str(ROOT / "data" / "README.md"),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="dataset",
    )
    print(f"done -> https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
