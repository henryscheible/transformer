from torch import Tensor
from torch.nn.functional import softmax
import torch


def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    # q shape: batch_size, seq_length, d_k
    # k shape: batch_size, seq_length, d_k
    # v shape: batch_size, seq_length, d_v
    d_k = q.size()[-1]

    # Perform a batch matrix product on q and k.
    # Dimension transform: [batch_size, seq_length, num_features] *
    #                      [batch_size, num_features, seq_length] =
    #                      [batch_size, seq_length, seq_length]
    temp = q.bmm(q, k.transpose(1, 2))

    # Calculate the scale and run the softmax operation
    scale = d_k ** (1 / 2)
    post_softmax = softmax(temp / scale, dim=-1)
    return post_softmax.bmm(v)


def multi_head_attention(q: Tensor,
                         k: Tensor,
                         v: Tensor,
                         weights_q: Tensor,
                         weights_k: Tensor,
                         weights_v: Tensor,
                         weights_o: Tensor,
                         h=8) -> Tensor:
    # q shape: batch_size, seq_length, d_k
    # k shape: batch_size, seq_length, d_k
    # v shape: batch_size, seq_length, d_v
    # weights_q shape: h, d_model, d_k
    # weights_k shape: h, d_model, d_k
    # weights_v shape: h, d_model, d_v
    # weights_o shape: h * d_v, d_model
    # d_k=d_v=d_model/h=64
    head = list(h)
    for i in range(h):
        head[i] = scaled_dot_product_attention(
            q.bmm(weights_q),
            k.bmm(weights_k),
            v.bmm(weights_v)
        )
    return torch.cat(head, dim=1).bmm(weights_o)
