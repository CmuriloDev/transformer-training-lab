from dataset import load_data
from tokenizer_utils import tokenize_pairs, tokenizer
from model import SimpleTransformer
from train import train
from inference import generate
import torch


def main():

    print("TRANSFORMER TRAINING LAB")

    pairs = load_data()

    src, tgt = tokenize_pairs(pairs)

    vocab_size = tokenizer.vocab_size

    model = SimpleTransformer(vocab_size)

    train(model, src, tgt, epochs=10)

    test_src = src[0].unsqueeze(0)

    output = generate(model, test_src, tokenizer)

    print("Generated:", output)


if __name__ == "__main__":
    main()