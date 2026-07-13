from collections.abc import Callable
from typing import Optional
import einops
import math
import numpy.typing as npt
import os
import torch
import torch.nn as nn
import typing

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.Tensor | None = None, dtype: torch.Tensor | None = None) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.weights, 0, std, -3 * std, 3 * std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weights.T

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.Tensor | None = None, dtype: torch.Tensor | None = None) -> None:
        super().__init__()
        self.embeddings = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.embeddings, 0, 1, -3, 3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        (batch, sequence_length) -> (batch, sequence_length, d_models)
        """
        return self.embeddings[token_ids]
    

"""
In the original transformer paper, the model uses a residual connection around each of the two sublayers, followed by layer normalization. This is known as post-norm.

Pre-norm actually improves training stablization if u apply it to the input of each sublayer if u apply it to the input of each sublayer if u apply it to the input of each sublayer if u apply it to the input of each sublayer, and u do another layer norm after the final transformer block.

The output of each transformer block sub-layer is then added to the sub-layer input via the residual connection.

"""
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, sequence_length, d_model) -> (batch_size, sequence_length, d_model)
        in_dtype = x.dtype
        x = x.to(torch.float32)

        result = x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.gamma

        return result.to(in_dtype)
    
"""
⊙ = element wise product

The original transformer paper had two linear transformations with a ReLU activation between them
Relu(x) = max(0, x). Also, the dimensionality of the inner feed forward layer is 4x the input dimensionality (to selectively zero out)

Recall: activation function is a non-linear function applied elementwise to a tensor to have the model learn non-linear tendencies. Otherwise, stacking layers is just a big matrix multiplication, as a composition of linear functions is still linear, and you'd never capture non-linear patterns.

Modern models incorporate
1. another activation function
2. employ a gating mechanism

SwiGLU (activation function adopted in LLMS like Llama 3 and Qwen 2.5), and omit the bias term, following most modern LLMs since PaLM and LLaMA

SiLU or swish activation function. sigmoid squashes any real number into range (0,1): smooth s shape
SiLU(x) = x * sigmoid(x) = x / (1 + e^(-x))

Gated Linear Units (GLUs) originally defined like so: element wise product of a linear transformation passed through a sigmoid and another linear transformation
- They reduce the vanishing gradient problem for deep learning by providing a linear path for the gradients while non-linear capabilities
GLU(x, W1, W2) = (sigmoid(W1x) ⊙ W2x)

Swish + GLU we get SwiGLU
FFN(x) = SwiGLU(x, W1, W2, W3) = W2(SiLU(W1x) ⊙ W3x)

x ∈ d_model
W1, W3 ∈ d_ff x d_model
W2 ∈ d_model x d_ff

d_ff = 8/3 d_model

x = (batch, row, col) = (batch, seq, d_model)
W1 = (d_ff, d_model)
W2 = (d_model, d_ff)
W3 = (dff, d_model)

gate: W1x = x * W1^T = (seq, d_model) x (d_model, d_ff) = (seq, d_ff) # project each token up
activated_gate = SiLU(gate) # (seq, d_ff) , activation will keep shape, and is element wise. applied to every element of that tensor
hidden = activated_gate ⊙ content # (seq, d_ff), element wise multiplication
output = W2 * hidden = hidden * W2^% # (seq_, d_ff) * (d_ff, d_model) project back down

# d(hidden)/d(content) = activated gate

content = W3x = x * W3^T = (seq, d_model) x (d_model, d_ff) = (seq, d_ff) # project token up again
"""
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, activation: str = "swiglu", device=None, dtype=None):
        super().__init__()
        assert activation in ("swiglu", "silu")
        self.activation = activation
        self.d_ff = round(8/3 * d_model / 64) * 64 if d_ff is None else d_ff
        self.w1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)
        # Gate projection only exists for SwiGLU. Plain SiLU FFN is w2(silu(w1 x)).
        self.w3 = Linear(d_model, self.d_ff, device=device, dtype=dtype) if activation == "swiglu" else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (batch, seq_length, d_model) -> (batch, seq_length, d_model)
        w1x = self.w1(x)
        silu_w1x = w1x * torch.sigmoid(w1x)
        if self.activation == "swiglu":
            return self.w2(silu_w1x * self.w3(x))
        return self.w2(silu_w1x)

"""
Rotary Position Embeddings (RoPE)

query token is a row in Q after we multiply x by the query matrix. dimension of Q is n x d

for a given query token q^i = W_q x^i ∈ d at token position i, we apply a pairwise rotation matrix R^i
giving us q'^i = R^iq^i = R^i W_q x^i

Your query vector q is d-dimensional. Pair up adjacent dimensions:
q = [q₀, q₁, q₂, q₃, q₄, q₅, ...]
     └─┬─┘ └─┬─┘ └─┬─┘
    pair 1 pair 2 pair 3 ...   (d/2 pairs total)

Each pair (q_{2k-1}, q_{2k}) (k starts at 1) is treated as a 2D vector you can rotate. by angle θ_{i,k} = i / Θ^((2k-2)/d_k) for k in 1...d/2 and some constant Theta.
i = token position, k = which pair in q_i

the further the position (i), the more the rotation. The further the pair (k), the slower the rotation.

Rotating a 2d matrix:
  R = [ cos θ   -sin θ ]
      [ sin θ    cos θ ]  

  R = [ R₁  0   0   ... ]    ← rotates pair 1
      [ 0   R₂  0   ... ]    ← rotates pair 2                                                                                                                              
      [ 0   0   R₃  ... ]    ← rotates pair 3                                                                                                                              
      [ ...             ]
Don't need to build this matrix, we can just apply each 2d rotation directly to the pair.

  q'_{2k-1} = q_{2k-1} · cos(θ_{i,k}) - q_{2k} · sin(θ_{i,k})                                                                                                            
  q'_{2k} = q_{2k} · sin(θ_{i,k}) + q_{2k-1} · cos(θ_{i,k})

  We rotate both q^i with R^i and k^j with R^j

This layer has no learnable parameters

"""
class RelativePositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        assert d_k % 2 == 0, "d_k is not even"
        # we need cos and sin tables of size (max_seq_len, d_k/2)
        # (i, k) remember
        # we need to build cos(θ_{i,k}) which is cos(i / Θ^((2k-2)/d_k))
        # we need to build sin(θ_{i,k}) which is sin(i / Θ^((2k-2)/d_k))
        
        # lets first get i which is torch.arange(n) which will give us [0, 1, ...,  n-1] of shape (n,)
        # we can also use torch.arange(start, end, step) # with a step
        """
        is=torch.arange(max_seq_len) will give us [0...(n-1)] size (max_seq_len,)
        ks=torch.arange(d_k/2) + 1 will give us [1...(d_k/2)] size (d_k/2)

        let's make this table (i, k)

        numerator = is
        denominator = Θ^((2k-2)/d_k) = theta ** ((2 * ks - 2)/d_k)

        so we need to broadcast numerator and denominator

        numerator[:, None] = (max_seq_len, 1)
        denominator[None, :] = (1, d_k/2)

        or even better
        theta_i_k = numerator[:, None] / denominator[None, :]
        """
        super().__init__()
        i = torch.arange(max_seq_len, device=device)
        k = torch.arange(d_k // 2, device=device) + 1
        denominator = theta ** ((2*k-2)/d_k)
        theta_i_k = i[:, None] / denominator[None, :]
        cos_i_k = torch.cos(theta_i_k)
        sin_i_k = torch.sin(theta_i_k)
        self.register_buffer("cos_i_k", cos_i_k, persistent=False)
        self.register_buffer("sin_i_k", sin_i_k, persistent=False)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x = (..., seq_len, d_k) -> (..., seq_len, d_k)
        # token positions = (..., seq_len) specifying token positions of x
        """
        for each pair k
        q'_{2k-1} = q_{2k-1} · cos(θ_{i,k}) - q_{2k} · sin(θ_{i,k})
        q'_{2k}   = q_{2k-1} · sin(θ_{i,k}) + q_{2k} · cos(θ_{i,k})

        pytorch indexing:
        cos_i_k = (max_seq_len, d_k/2)

        we need to get cos_k for each (..., seq_len) in token_position

        self.cos_i_k[token_positions] (..., seq_len, d_k/2)
        self.sin_i_k[token_positions] (..., seq_len, d_k/2)

        x_pairs = x.reshape(*x.shape[:-1], -1, 2) (..., seq_len, d_k) -> (..., seq_len, d_k/2, 2) # pairwise
        x.reshape(*x.shape[:-1], -1, 2) = x.reshape(..., seq_len, -1, 2) # -1 becomes (d_k/2) which is inferred. the last dimension basically is (q'_2{k-1}, q'_{2k})


        stacking a, b with shape S results in S + (2,)
        torch.stack([a,b], dim=-1) # new dim at the end (-1 is last position)

        x_pairs[0] = q_{2k-1} (..., seq_len, d_k/2)
        x_pairs[1] = q_{2k}   (..., seq_len, d_k/2)

        x_prime_pair_0 = x_pairs[..., 0] * self.cos_i_k[token_positions] - x_pairs[..., 1] * self.sin_i_k[token_positions] (..., seq_len, d_k/2)
        x_prime_pair_1 = x_pairs[..., 0] * self.sin_i_k[token_positions] + x_pairs[..., 1] * self.cos_i_k[token_positions] (..., seq_len, d_k/2) 

        x_prime_pairs = torch.stack([x_prime_pair_0, x_prime_pair_1], dim=-1) # (..., seq_len, d_k/2, 2) # stacks them on new dimension which is last dim

        # undo reshape by flattening last two layers, which undoes reshaping
        # x_prime = x_prime_pairs.flatten(-2) == x_prime_pairs.reshape(*x_prime_pairs.shape[:-2], -1) == x.reshape(..., seq_len, (d_k/2, 2)) combined = (..., seq_len, d_k)
        """
        in_dtype = x.dtype
        x_pairs = x.reshape(*x.shape[:-1], -1, 2) # (..., seq_len, d_k/2, 2)
        x_prime_pair_0 = x_pairs[..., 0] * self.cos_i_k[token_positions] - x_pairs[..., 1] * self.sin_i_k[token_positions] # (..., seq_len, d_k/2)
        x_prime_pair_1 = x_pairs[..., 0] * self.sin_i_k[token_positions] + x_pairs[..., 1] * self.cos_i_k[token_positions] # (..., seq_len, d_k/2)
        x_prime_pairs = torch.stack([x_prime_pair_0, x_prime_pair_1], dim=-1) # (..., seq_len, d_k/2, 2)
        x_prime = x_prime_pairs.flatten(-2) # (..., seq_len, d_k)
        # cos/sin tables are kept in fp32 for numerical accuracy; cast back so SDPA sees matching dtypes for q/k/v.
        return x_prime.to(in_dtype)


def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    """
    softmax(v)_i = exp(v_i) / sum over j of exp(v_j)

    softmax on dimension: perform that operation and that gets collapsed (eg reduced, traversed, normalized)

    eg sum on dim 0 on shape (2, 3, 4) -> (3, 4) (dim 0 collapses), we do the operation ACROSS that axis.

tensor([[[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]],

        [[12, 13, 14, 15],
         [16, 17, 18, 19],
         [20, 21, 22, 23]]])

tensor([[12, 14, 16, 18],
        [20, 22, 24, 26],
        [28, 30, 32, 34]])

    so softmax will normalize ACROSS that axis such that it sums to 1a.

    each line is a 1d slice along axis k, and the operation runs on each line independently.

    arithmetic operations broadcast automatically
    eg
    (3, 4, 5) * (1, 4, 5), that dim of size 1 will broadcast to size 3. if dims are missing, they are treated as 1 after right aligning

    oh fuck we need to subtract the max (softmax is invariant to adding a constant to all values)

    numerator/denominator e^c cancel out if we add constant c to each element of v
    """
    exp_x = torch.exp(x - torch.max(x, dim=i, keepdim=True).values)
    return exp_x / torch.sum(exp_x, dim=i, keepdim=True)

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """
    Q: (..., n, dk)
    K: (..., m, dk)
    V: (..., m, dk)

    QKT = (n, m)

    mask = (n, m)
    """
    d_k = Q.shape[-1]
    scores = (Q @ K.mT) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    return softmax(scores, -1) @ V

"""
MultiHeadSelfAttention:

multiple attention heads concat together

Q: (n, dk)
K: (m, dk)
V: (m, dk)

MultiHead(W, K, V) = concat ( head1, ..., head h)
for head_i = Attention(Q_i, K_i, V_i)

MultiHeadSelfAttention(x) = W_o * MultiHead(Wq * x, Wk * x, Wv * x)

given h attention heads, learneable parameters are
Wq = (h * d_k, d_model)
Wk = (h * d_k, d_model)
Wv = (h * d_v, d_model)
Wo = (d_model, h * d_v)
"""
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int | None = None, theta: float | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        d_k = d_model // num_heads
        d_v = d_model // num_heads
        self.h = num_heads
        h = num_heads
        self.wq = Linear(d_model, h * d_k, device = device, dtype = dtype)
        self.wk   = Linear(d_model, h * d_k, device = device, dtype = dtype)
        self.wv = Linear(d_model, h * d_v, device = device, dtype = dtype)
        self.wo = Linear(h * d_v, d_model, device = device, dtype = dtype)
        self.rope = RelativePositionalEmbedding(theta, d_k, max_seq_len, device=device) if theta is not None else None
        self.register_buffer("mask", torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool, device=device)) if max_seq_len is not None else None, persistent=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x is (B, n, d_model) -> (B, n, d_model)
        """
        Q = self.wq(x) # (B, n, h * d_k)
        K = self.wk(x) # (B, n, h * d_k)
        V = self.wv(x) # (B, n, h * d_v)

        n = x.shape[-2]

        q_reshaped = einops.rearrange(Q, "b n (h dk) -> b h n dk", h = self.h)
        k_reshaped = einops.rearrange(K, "b n (h dk) -> b h n dk", h = self.h)
        v_reshaped = einops.rearrange(V, "b n (h dv) -> b h n dv", h = self.h)

        if self.rope is not None:
            positions = torch.arange(n, device=x.device)
            q_reshaped = self.rope(q_reshaped, positions)
            k_reshaped = self.rope(k_reshaped, positions)

        # --- original hand-rolled path (kept for reference) ---
        # mask = self.mask[:n, :n] if self.mask is not None else torch.tril(torch.ones(n, n, dtype=torch.bool, device=x.device))
        # attention_results = scaled_dot_product_attention(q_reshaped, k_reshaped, v_reshaped, mask) # (B, h, n, dv)
        # --- SDPA path (FlashAttention on Ampere+/Hopper/Blackwell) ---
        attention_results = torch.nn.functional.scaled_dot_product_attention(
            q_reshaped, k_reshaped, v_reshaped, is_causal=True
        ) # (B, h, n, dv)
        return self.wo(einops.rearrange(attention_results, "b h n dv -> b n (h dv)", h = self.h))

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float,
                 use_rmsnorm: bool = True, use_rope: bool = True, activation: str = "swiglu",
                 device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        norm = lambda: RMSNorm(d_model, device=device, dtype=dtype) if use_rmsnorm else nn.Identity()
        self.norm1 = norm()
        self.attention = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta if use_rope else None, device, dtype)
        self.norm2 = norm()
        self.ffn = PositionWiseFeedForward(d_model, d_ff, activation=activation, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class TransformerLm(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float,
                 use_rmsnorm: bool = True, use_rope: bool = True, activation: str = "swiglu",
                 device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta,
                             use_rmsnorm=use_rmsnorm, use_rope=use_rope, activation=activation,
                             device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype) if use_rmsnorm else nn.Identity()
        self.lm_head = Linear(d_model, vocab_size, device, dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, seq_len) -> (batchsize, seq_len, vocab_size)
        embedded_tokens = self.token_embeddings(x)
        for layer in self.layers:
            embedded_tokens = layer(embedded_tokens)
        # still at (batchsize, seq_len, d_model)
        embedded_tokens = self.final_norm(embedded_tokens)
        return self.lm_head(embedded_tokens) # returns (Batch, seq length, vocab_size)
    

"""
Accounting for flops

mat mul is 2mnp flops, given A (mxn) and B (nxp) matrices and product AB

this is because AB[i, j] computes dot of A[i, :] and B[:, j] which requires n additions and n multiplications. We do this per m, p, so multiply everything

Number of learnable parameters

let dk = dv = d_model/num_heads

embeddings: vocab_size * d_model
num_layers * (
    attention:
        norm1: d_model +
        wq: d_model * num_heads * d_k +
        wk: d_model * num_heads * d_k + 
        wv: d_model * num_heads * d_v +
        wo: num_heads * d_v * d_model +
        norm2: d_model +
    +
    ffn:
        w1: d_model * d_ff
        w2: d_f * d_model
        w3: d_model * d_ff
)
+
final_norm: d_model +
lm_head : d_model * vocab_size

combining everything we get a total learnable parameter
vocab_size * d_model + num_layers * (4 * (d_model * d_model) + 2 * d_model + 3 * d_model * d_ff) + d_model + d_model * vocab_size

num_parameters = d_model * (2 * vocab_size + num_layers * (2 + 4 * d_model + 3 * d_ff) + 1)

for
vocab_size: 50257
context_length: 1024
num_layers: 48
d_model: 1600
num_heads: 25
d_ff: 4288

thats 1640452800 parameters * 4 bytes = ~6.56 GB
"""

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits: output of transformer (batch_size, vocab_size) and targets (batch_size)

    return single number loss

    we need to compute the max along the last dim (vocab size) and subtract it
    """
    max_logits = torch.max(logits, dim=-1, keepdim=True).values
    numerically_stable_logits = logits - max_logits
    exp_logits = torch.exp(numerically_stable_logits) # (batch_size, vocab_size) still
    return (torch.log(torch.sum(exp_logits, dim=-1)) - numerically_stable_logits.gather(dim = -1, index=targets.unsqueeze(-1)).squeeze(-1)).mean()

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, betas, eps):
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps
        }
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            betas = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # get state associated with p
                if "t" not in state:
                    # initialize
                    state["t"] = 1
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                t = state["t"]
                grad = p.grad.data # gradient of loss with respect to p
                a_t = lr * math.sqrt(1-betas[1]**t) / (1-betas[0]**t)
                p.data -= lr * weight_decay * p.data
                state["m"].mul_(betas[0]).add_(grad, alpha=(1-betas[0]))
                state["v"].mul_(betas[1]).add_(grad**2, alpha=(1-betas[1]))
                p.data -= a_t * state["m"] / (torch.sqrt(state["v"]) + eps)
                state["t"] += 1
        return loss

def get_lr_cosine_schedule(
    t,
    max_learning_rate, # a_max
    min_learning_rate, # a_min
    warmup_iters, # T_w
    cosine_cycle_iters, # T_c
):
    if t < warmup_iters:
        return t / warmup_iters * max_learning_rate
    elif warmup_iters <= t <= cosine_cycle_iters:
        return min_learning_rate + 1/2*(1+math.cos(math.pi*(t-warmup_iters)/(cosine_cycle_iters-warmup_iters)))*(max_learning_rate-min_learning_rate)
    elif t > cosine_cycle_iters:
        return min_learning_rate

def gradient_clipping(
    parameters,
    max_l2_norm
):
    """
    During training we can sometimes hit training examples yieldingn large gradients, which can destabalize training.
    We enforce a limit on the l2 norm of the gradient after each backwards pass before taking an optimizer step.

    We scale g down be a factor of M / (||g||_2 + e) where e is a small number added for numerical stability
    """
    parameters = list(parameters)
    norm = 0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        norm += (parameter.grad.data ** 2).sum()
    
    total_norm = torch.sqrt(norm)
    if total_norm >= max_l2_norm:
        for parameter in parameters:
            if parameter.grad is None:
                continue
            parameter.grad.data *= max_l2_norm/(total_norm+1e-6)
    return total_norm

def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str)-> tuple[torch.Tensor, torch.Tensor]:
    data = torch.from_numpy(dataset)
    sample_starts = torch.randint(0, data.shape[0] - context_length, (batch_size, )) # (batch_size) of random ints from [0, n-m)
    offsets = torch.arange(context_length) # (context_length)

    idx = sample_starts[:, None] + offsets[None, :] # (batch_size, context_length) batch sizes get stretched along cols, and offsets get stretched along rows, then we add them.
    inputs = data[idx].long().to(device)
    targets = data[idx+1].long().to(device)
    return (inputs, targets)

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    # Unwrap torch.compile() so state_dict keys don't get the "_orig_mod." prefix.
    inner = getattr(model, "_orig_mod", model)
    torch.save({
            "model": inner.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration
        }, out)

def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    obj = torch.load(src)
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])
    return obj["iteration"]

