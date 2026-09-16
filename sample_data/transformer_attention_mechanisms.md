# Transformer Attention Mechanisms and KV-Cache Memory Dynamics

## 1. Scaled Dot-Product Self-Attention
At the core of modern neural language modeling lies the scaled dot-product attention function:
`Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V`
Where `Q` (Query), `K` (Key), and `V` (Value) represent linear projections of the input representations, and `d_k` is the projection dimensionality of the keys. The scaling factor `1 / sqrt(d_k)` prevents the dot product magnitudes from growing excessively large in high dimensions, which would otherwise push the softmax function into regions with vanishingly small gradients.

Multi-Head Attention (MHA) extends this operation by projecting queries, keys, and values into `h` distinct representation subspaces. This enables the model to simultaneously attend to information from disparate representation subspaces at different token positions, capturing both syntactic dependencies and long-range semantic relationships.

## 2. KV-Cache Memory Bandwidth Bottlenecks
During autoregressive token generation, previous tokens are cached to avoid redundant recalculation of keys and values for past positions. While this caching reduces computational complexity from quadratic `O(N^2)` to linear `O(N)` per generation step, it introduces an acute memory bandwidth bottleneck on hardware accelerators.

The memory footprint of the Key-Value (KV) cache scales linearly with sequence length, batch size, number of layers, and hidden dimensions:
`KV_Memory = 2 * 2 * b * s * l * h * d` bytes
For a 70-billion parameter model serving a context of 128,000 tokens with 16-bit precision, the KV cache alone demands hundreds of gigabytes of high-bandwidth memory (HBM), dwarfing the parameter weights themselves during multi-tenant inference.

## 3. Structural Mitigations: MQA, GQA, and FlashAttention
To circumvent the memory wall, architectural variations have replaced vanilla Multi-Head Attention:
1. Multi-Query Attention (MQA): A single key head and value head are shared across all query heads, slashing KV cache memory traffic by a factor equal to the number of heads (typically 8x to 64x).
2. Grouped-Query Attention (GQA): An optimal interpolation between MHA and MQA where query heads are partitioned into groups, each sharing a single key-value head pair (e.g., 8 KV heads for 64 query heads). GQA recovers full model quality while retaining nearly all memory throughput gains of MQA.
3. FlashAttention: An exact attention algorithm that reorganizes memory accesses to maximize SRAM utilization on GPUs via block tiling. By fusing softmax computation and avoiding materialized `N x N` attention matrices in high-latency GPU High Bandwidth Memory, FlashAttention achieves 2x to 4x wall-clock speedups while reducing memory footprint to linear `O(N)`.

## 4. Rotary Position Embeddings (RoPE)
Rotary Position Embeddings encode absolute position with a rotation matrix while naturally incorporating relative position dependency through inner product invariants:
`q_m^T * k_n = g(x_m, x_n, m - n)`
RoPE exhibits superior length extrapolation capabilities, allowing models trained on 4K or 8K sequences to scale reliably to contexts exceeding 128K tokens through frequency base modification.
