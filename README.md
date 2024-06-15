# Adalayers

Frozen transformer models adaptive layers selection
for downstream tasks efficient solving.

---

## Best results

> **Note:** results are highly dependent on base model domain.
> Clearly, some task-specific model could have been used to
> achieve even higher scores. Though, such
> comparison would not be fair.

| **Dataset** | **Base model** | **Score** |
|:-----------:|:--------------:|---------:|
| **IMDB**        | RoBERTa-large  | 96.1% acc <br>*(SOTA level)* |
| **CoLA**        | RoBERTa-large  | 83.6% acc <br>*(SOTA level)* |
| **CoNLL**       | RoBERTa-large  | 89.4% f1  |

## Problem

Let's consider the case when you already have a functioning SOTA-level large
transformer model and you need to solve a different task on the same data.
Aka speech recognition + emotions recognition, text summarization + NER, etc.

One possible solution is to use second model.
However, deploying a second transformer model requires lots of resources.
Combining two tasks into a single model training is not always feasible and may
deteriorate main task metrics.

## Solution

Let's reuse transformer hidden states! It is well-known that different
layers of transformer model extract features of different level. Therefore,
if we effectively combine hidden features, we could achieve good results.

Moreover, in such case base model stays intact, and as we reuse its computations,
proposed algorithm is highly computationally efficient.

General idea presented on the image

<img width="712" alt="Screenshot 2024-06-15 at 15 29 37" src="https://github.com/alexdremov/adalayers/assets/25539425/abbddae1-ba58-46cc-ad9a-672d38dca68f">

