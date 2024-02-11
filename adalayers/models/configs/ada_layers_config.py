from typing import Optional

from transformers import PretrainedConfig


class AdaLayersForSequenceClassificationConfig(PretrainedConfig):
    model_type = "ada_layers_classifier"

    def __init__(
        self,
        base_model: str = "bert-base-uncased",
        project_dim: int = 128,
        layers_num: int = 13,
        layer_in_dim: int = 768,
        attention_heads_num: int = 16,
        num_classes: int = 2,
        attention_dropout_prob: Optional[float] = None,
        topk_distribution: Optional[int] = None,
        freeze_distribution: bool = False,
        alpha_distribution: float = 10.0,
        lambda_distribution_entropy: float = 0.5,
        **kwargs,
    ):
        self.base_model = base_model
        self.project_dim = project_dim
        self.layers_num = layers_num
        self.layer_in_dim = layer_in_dim
        self.attention_heads_num = attention_heads_num
        self.attention_dropout_prob = attention_dropout_prob
        self.topk_distribution = topk_distribution
        self.freeze_distribution = freeze_distribution
        self.num_classes = num_classes
        self.alpha_distribution = alpha_distribution
        self.lambda_distribution_entropy = lambda_distribution_entropy

        super().__init__(**kwargs)
