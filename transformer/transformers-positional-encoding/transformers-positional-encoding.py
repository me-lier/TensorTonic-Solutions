import numpy as np
import torch
def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here
    # pos_embedding_layer = torch.nn.Embedding(seq_length, d_model)
    # return pos_embedding_layer(torch.arange(seq_length))
    positions = np.arange(seq_length)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]

    angle_rates = 1 / np.power(
        10000,
        (2 * (dims// 2)) / d_model
    )

    angle_rads = positions * angle_rates

    pe = np.zeros((seq_length, d_model))

    pe[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pe[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return pe