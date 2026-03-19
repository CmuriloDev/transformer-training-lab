from datasets import load_dataset


def load_data():
    dataset = load_dataset("bentrevett/multi30k", split="train[:200]")

    pairs = []

    for item in dataset:
        src = item["en"]
        tgt = item["de"]
        pairs.append((src, tgt))

    return pairs