"""Preprocess MMLU (cais/mmlu, 'all') into the parquet format the self-play
(sp_env.MMLUEnv) agent loop expects.

The env builds its own prompts from the `question` / `subject` / `choices` /
`answer` columns; the `prompt` column exists only to satisfy RLHFDataset's
tokenization + overlong filtering.

Usage:
    python3 examples/data_preprocess/mmlu.py --local_save_dir ~/data/mmlu
"""

import argparse
import os

import datasets

CHOICE_LABELS = ["A", "B", "C", "D"]


def build_prompt(question: str, subject: str, choices) -> list:
    subject = subject.replace("_", " ").title()
    choices_text = "\n".join(f"{l}: {t}" for l, t in zip(CHOICE_LABELS, choices))
    return [
        {
            "role": "user",
            "content": (
                f"Domain: {subject}\n"
                f"Question: {question}\n"
                f"Options:\n{choices_text}"
            ),
        }
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir", default="~/data/mmlu")
    parser.add_argument("--config", default="all")
    parser.add_argument("--splits", default="test,validation")
    args = parser.parse_args()

    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)

    for split in args.splits.split(","):
        split = split.strip()
        ds = datasets.load_dataset("cais/mmlu", args.config, split=split)

        def process(example, idx):
            question = example["question"]
            subject = example["subject"]
            choices = list(example["choices"])
            answer = int(example["answer"])
            return {
                "data_source": "cais/mmlu",
                "prompt": build_prompt(question, subject, choices),
                "ability": "mmlu",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": CHOICE_LABELS[answer],
                },
                "extra_info": {"split": split, "index": idx},
                # consumed by verl/envs/sp_env.py:transform_dataproto_to_sample
                "question": question,
                "subject": subject,
                "choices": choices,
                "answer": answer,
                "uid": f"mmlu_{split}_{idx}",
            }

        ds = ds.map(process, with_indices=True, remove_columns=ds.column_names)
        out = os.path.join(save_dir, f"{split}.parquet")
        ds.to_parquet(out)
        print(f"wrote {out}  ({len(ds)} rows)")


if __name__ == "__main__":
    main()
