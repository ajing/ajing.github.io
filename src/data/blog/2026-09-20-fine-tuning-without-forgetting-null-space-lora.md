---
author: Jing Lu
pubDatetime: 2026-09-20T18:00:00-07:00
title: "Fine-Tuning Without Forgetting: A Null-Space LoRA Experiment"
featured: true
draft: false
tags:
  - AI
  - LLM
  - ML Engineering
  - Post Training
  - Continual Learning
description: "A small PyTorch experiment shows how a rank-one update can learn new mappings while keeping protected logits unchanged to floating-point precision."
---

Fine-tuning changes a model so that it can learn a new task. The same update can damage behavior that already works. This conflict is the stability-plasticity problem.

A small experiment makes this problem easy to see. It also gives a clean solution. Put the weight update in the null space of the protected activations. Then the update is zero on those activations by construction.

The key result is simple:

> Low rank limits the size of an update. It does not limit where the update acts.

A rank-one update can still damage old behavior. A rank-one update in the correct subspace can preserve selected logits to floating-point precision.

## The task

The model is a one-token neural network with 56 parameters.

| Part               |   Shape |
| ------------------ | ------: |
| Token embedding    | `4 × 2` |
| First linear layer | `8 × 2` |
| `tanh` activation  |     `8` |
| Output layer       | `4 × 8` |

The vocabulary contains four tokens: `a`, `b`, `c`, and `d`. Base training produces these mappings:

```text
a -> b
b -> a
c -> a
d -> a
```

The new data contains only two mappings:

```text
c -> d
d -> c
```

The final model must keep `a -> b` and `b -> a`. It must learn the new mappings without replay data. It also cannot route tokens through separate models or inference branches.

The strongest test compares all output logits for `a` and `b`. The test does not compare only the top prediction.

## Why ordinary LoRA is not enough

[LoRA](https://arxiv.org/abs/2106.09685) freezes a weight matrix $W$ and learns a low-rank update:

$$
W' = W + BA
$$

The rank controls the capacity of the update. It does not control its effect on a specific input.

Let $h(x)$ be the hidden state before the output layer. The logit change is:

$$
\Delta z(x) = BA h(x)
$$

This value is usually nonzero for old inputs. A small rank does not change that fact.

The experiment confirms this point. A rank-two LoRA update learns the new data, but it changes the protected logits by `110.8`. It also breaks the old mapping for `b`.

## Put the update in a safe subspace

Stack the protected hidden states as rows of a matrix:

$$
H_p =
\begin{bmatrix}
h(a)^T \\
h(b)^T
\end{bmatrix}
\in \mathbb{R}^{2 \times 8}
$$

Now find a basis $N$ for the null space of $H_p$:

$$
H_p N = 0
$$

For this model, $H_p$ has at most rank two. Its null space therefore has at least six dimensions.

Restrict the output update to that null space:

$$
\Delta W = B A N^T
$$

For a protected hidden state $h_p$, the logit change is:

$$
\Delta z_p = B A N^T h_p = 0
$$

This is an algebraic constraint. It does not depend on a loss penalty or a careful learning rate.

## A rank-one form needs only one safe direction

Choose one vector $v$ from the null space. Then learn one output vector $u$:

$$
\Delta W = u v^T
$$

The output change becomes:

$$
\Delta z(x) = u\left(v^T h(x)\right)
$$

For `a` and `b`, the scalar $v^T h(x)$ is zero. Training $u$ cannot change their logits.

This form adds four trainable output parameters. The vector $v$ stays fixed. After training, the update can merge into the output weight:

$$
W' = W + u v^T
$$

The deployed model keeps its original inference graph.

## The embeddings add plasticity

The new batch contains only `c` and `d`. The embedding rows for `a` and `b` receive zero gradient. We can therefore train the embedding table while the new task changes only the rows for `c` and `d`.

This argument requires an optimizer without weight decay on the embedding table. Decoupled weight decay can change a row even when its task gradient is zero. A strict implementation must exclude these embeddings from weight decay or mask the protected rows.

The combination has two useful parts:

- The movable `c` and `d` embeddings give the model more ways to separate the new inputs.
- The null-space update changes the output layer without changing the protected logits.

## Minimal PyTorch implementation

The model uses singular value decomposition, or SVD, to find the null space. SVD splits a matrix into orthogonal directions with measured strength.

```python
import copy

import torch
from torch import nn


class ProtectedRankOne(nn.Module):
    def __init__(self, base_model, protected_ids, vocab_size):
        super().__init__()
        self.model = copy.deepcopy(base_model)

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        # The new batch contains only c and d.
        self.model.token_embedding.requires_grad_(True)

        with torch.no_grad():
            protected_hidden = self.model.hidden(protected_ids)
            protected_hidden = protected_hidden.reshape(
                len(protected_ids), -1
            )

            _, _, vh = torch.linalg.svd(
                protected_hidden,
                full_matrices=True,
            )
            hidden_rank = torch.linalg.matrix_rank(
                protected_hidden
            ).item()
            null_basis = vh[hidden_rank:].T

            # Keep one null-space direction fixed.
            self.register_buffer("v", null_basis[:, 0])

        self.u = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, input_ids):
        hidden = self.model.hidden(input_ids)
        base_logits = self.model.fc2(hidden)

        # delta_W = u v^T
        safe_coordinate = hidden @ self.v
        delta_logits = safe_coordinate.unsqueeze(-1) * self.u
        return base_logits + delta_logits
```

Use `Adam` with `weight_decay=0.0` for this experiment. After training, add `u[:, None] * v[None, :]` to `fc2.weight`.

## Results

The prediction string below uses inputs in the order `[a, b, c, d]`. The target string is `badc`.

| Method                                      | New-task loss | Max `a/b` logit drift | Prediction | Trainable parameters |
| ------------------------------------------- | ------------: | --------------------: | ---------: | -------------------: |
| Full fine-tuning                            |      0.000208 |                 9.698 |     `bddc` |                   56 |
| Embedding only                              |       1.06455 |                     0 |     `badd` |                    8 |
| Ordinary FC2 LoRA, rank 2, plus embeddings  |      0.000103 |                 110.8 |     `bddc` |                   32 |
| Protected FC2 LoRA, rank 2                  |       0.01234 |             `9.2e-14` |     `badc` |                   20 |
| Protected FC2 LoRA, rank 1, plus embeddings |      0.000094 |             `4.3e-15` |     `badc` |                   18 |
| Closed-form protected edit, rank at most 2  |      0.000045 |             `8.5e-14` |     `badc` |            0 trained |

Full fine-tuning and ordinary LoRA fit the new data. Both methods damage the old behavior. Embedding-only training keeps the old logits, but it fails to learn both new mappings.

The protected methods learn all four mappings. Their measured drift stays near the error level of 64-bit floating-point arithmetic.

The rank-one LoRA experiment succeeded with all 30 tested initializations. Its worst new-task loss was `0.000106`. Its worst protected drift was `1.71e-14`.

The smaller fixed-direction form also worked. Each of the six null-space basis vectors was tested alone. All six runs learned the correct mappings. They added only four output-side parameters before the merge.

## Why the closed-form edit is not the default choice

The new task has two inputs. We can solve a direct linear system for an output-weight update. A pseudoinverse gives an exact protected edit with a chosen logit margin.

This solution gets the lowest loss in the table. It also needs a manual target margin. Its behavior depends on the condition of the small linear system. Gradient training is easier to extend when the new data contains many examples.

## Relation to continual learning research

This experiment is small, but its geometry matches recent work on continual learning.

[InfLoRA](https://openaccess.thecvf.com/content/CVPR2024/html/Liang_InfLoRA_Interference-Free_Low-Rank_Adaptation_for_Continual_Learning_CVPR_2024_paper.html) places low-rank updates in a subspace that reduces interference with earlier tasks.

[NESS](https://arxiv.org/abs/2602.21919) uses small singular-value directions from old activations. It fixes the subspace basis and trains one task matrix. The fixed vector $v$ in this experiment follows the same core idea.

The important object is not the LoRA rank by itself. The important object is the relation between the update space and the protected activation space.

## What changes in a large language model

The same idea can apply to a linear layer in a transformer:

1. Collect input activations for behavior that the model must preserve.
2. Use SVD to find directions with small singular values.
3. Fix a basis for those directions.
4. Train a low-rank update inside that basis.
5. Measure old-task drift and new-task quality after the merge.

Large models make each step harder. The protected activation matrix can reach full rank. An exact null space can then disappear. In that case, small singular-value directions give an approximate null space, not an exact guarantee.

Coverage also matters. A null-space constraint protects the activations in the reference set and their linear span. It does not protect every prompt that expresses the same skill. Context, token position, and earlier layers can change the activation before it reaches the protected layer.

## Limits of the guarantee

The zero-drift result needs four conditions:

- The protected hidden states stay fixed.
- The vector $v$ stays in their null space.
- The output update uses only $u v^T$.
- The embedding optimizer does not change protected rows.

The guarantee is local to one layer and one protected activation set. It is not a proof that a full language model keeps every old capability.

There is also a capacity trade-off. A larger protected activation span leaves a smaller null space. More stability removes directions that the new task can use.

## The main lesson

Parameter-efficient tuning does not prevent forgetting by itself. Low rank reduces parameter count, but it does not define which behavior stays fixed.

The update needs a geometric constraint. If old activations occupy a known subspace, place the new weight update in its orthogonal complement. This turns selected stability from a training preference into a property of the parameterization.

The toy model gives an exact result because its geometry is small. The same view remains useful at scale, even when the null space becomes approximate.

## References

- Edward J. Hu et al. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685), 2021.
- Yan-Shuo Liang and Wu-Jun Li. [InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning](https://openaccess.thecvf.com/content/CVPR2024/html/Liang_InfLoRA_Interference-Free_Low-Rank_Adaptation_for_Continual_Learning_CVPR_2024_paper.html), CVPR 2024.
- Cuong Anh Pham, Praneeth Vepakomma, and Samuel Horváth. [Learning in the Null Space: Small Singular Values for Continual Learning](https://arxiv.org/abs/2602.21919), 2026.
