import torch
import torch.nn as nn
import torch.optim as optim


def train(model, src, tgt, epochs=10):

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):

        optimizer.zero_grad()

        # shift target
        input_tgt = tgt[:, :-1]
        target = tgt[:, 1:]

        output = model(src, input_tgt)

        output = output.reshape(-1, output.shape[-1])
        target = target.reshape(-1)

        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")