import torch


def generate(model, src, tokenizer, max_len=10):

    model.eval()

    generated = [tokenizer.cls_token_id]

    for _ in range(max_len):

        tgt = torch.tensor([generated])

        output = model(src, tgt)

        next_token = output[:, -1, :].argmax(dim=-1).item()

        generated.append(next_token)

        if next_token == tokenizer.sep_token_id:
            break

    return generated