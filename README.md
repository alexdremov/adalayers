# Adalayers

Frozen transformer models adaptive layers selection
for downstream tasks efficient solving.

---

## Best results

> **Note:** results are highly dependent on base model domain.
> Clearly, some task-specific model could have been used to
> achieve even higher scores. Though, such
> comparison would not be fair.

| Dataset | Base model | Score |
|:-----------:|:--------------:|---------:|
| IMDB        | RoBERTa-large  | 96.1% acc <br>*(SOTA level)* |
| CoLA        | RoBERTa-large  | 83.6% acc <br>*(SOTA level)* |
| CoNLL       | RoBERTa-large  | 89.4% f1  |

## Problem

Let's consider the case when you already have a functioning SOTA-level large transformer model and you need to solve a different task on the same data.
Aka speech recognition + emotion recognition, text summarization + NER, etc.

One possible solution is to use a second model.
However, deploying a second transformer model requires lots of resources.
Combining two tasks into a single model training is not always feasible and may
deteriorate main task metrics.

## Solution

Let's reuse transformer hidden states! It is well-known that different
layers of the transformer model extract features of different levels. Therefore,
if we effectively combine hidden features, we could achieve good results.

Moreover, in such a case, the base model stays intact, and as we reuse its computations,
the proposed algorithm is highly computationally efficient.

The general idea is presented in the image

<img width="712" alt="algo" src="https://github.com/alexdremov/adalayers/assets/25539425/abbddae1-ba58-46cc-ad9a-672d38dca68f">

Code for F can be found in [`adalayers/models/ada_layers_base.py`](https://github.com/alexdremov/adalayers/blob/main/adalayers/models/ada_layers_base.py).
All models are implemented with Huggingface interfaces.

## Launch

Training is omegaconf+hydra configurable. Configs can be found in [`configs`](https://github.com/alexdremov/adalayers/tree/main/configs).

The environment is poetry-controlled. You can set it up by calling `poetry install`

You can launch simple training by calling `adalayers/training/run.py` like

```
python adalayers/training/run.py --config-name adalayers_imdb.yaml
```
