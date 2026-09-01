import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Your code here
    attention_scores = Q @ K.transpose(1,2) #batching
    attention_scores = attention_scores / math.sqrt(K.shape[-1])
    attention_weights = torch.softmax(attention_scores, dim = -1)
    context_vector = attention_weights @ V

    return context_vector
    