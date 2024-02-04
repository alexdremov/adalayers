from typing import Optional

from transformers import PretrainedConfig


class AdaLayersForSequenceClassificationConfig(PretrainedConfig):
    model_type = "ada_layers_classifier"

    def __init__(
            self,
            base_model: str,
            project_dim: int,
            layers_num: int,
            layer_in_dim: int,
            attention_heads_num: int,
            num_classes: int,
            attention_dropout_prob: Optional[float] = None,
            topk_distribution: Optional[int] = None,
            freeze_distribution: bool = False,
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

        super().__init__(**kwargs)
