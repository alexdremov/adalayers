from typing import Optional

from transformers import PretrainedConfig


class AdaLayersBaseConfig(PretrainedConfig):
    model_type = "ada_layers_base"

    def __init__(
        self,
        base_model: str = "bert-base-uncased",
        project_dim: int = 128,
        layers_num: int = 13,
        layer_in_dim: int = 768,
        attention_heads_num: int = 16,
        attention_dropout_prob: Optional[float] = None,
        topk_distribution: Optional[int] = None,
        freeze_distribution: bool = False,
        alpha_distribution: float = 10.0,
        lambda_distribution_entropy: float = 0.02,
        pick_one_layer_only: Optional[int] = None,
        freeze_base_model: bool = True,
        classes_weights: Optional[list[float]] = None,
        add_pos_embeddings: bool = False,
        generate_fake_decoder_input_ids=False,
        distribution_cutoff = 0.0,
        **kwargs,
    ):
        self.base_model = base_model
        self.project_dim = project_dim
        self.layers_num = layers_num
        self.layer_in_dim = layer_in_dim
        self.attention_heads_num = attention_heads_num
        self.attention_dropout_prob = attention_dropout_prob
        self.topk_distribution = topk_distribution
        self.freeze_distribution = freeze_distribution or pick_one_layer_only is not None
        self.alpha_distribution = alpha_distribution
        self.lambda_distribution_entropy = lambda_distribution_entropy
        self.pick_one_layer_only = pick_one_layer_only
        self.freeze_base_model = freeze_base_model
        self.classes_weights = list(classes_weights) if classes_weights is not None else None
        self.add_pos_embeddings = add_pos_embeddings
        self.generate_fake_decoder_input_ids = generate_fake_decoder_input_ids
        self.distribution_cutoff = distribution_cutoff

        super().__init__(**kwargs)


class AdaLayersForSequenceClassificationConfig(AdaLayersBaseConfig):
    model_type = "ada_layers_classifier"

    def __init__(
        self,
        num_classes: int = 2,
        **kwargs
    ):
        self.num_classes = num_classes
        super().__init__(**kwargs)



class AdaLayersForTokenClassificationConfig(AdaLayersBaseConfig):
    model_type = "ada_layers_token_classifier"

    def __init__(
        self,
        num_classes: int = 2,
        focal_loss_enabled: bool = False,
        focal_loss_gamma: float = 2.0,
        focal_loss_alpha: float = 0.25,
        attention_layers_num=1,
        dim_feedforward=1024,
        use_crf= False,
        **kwargs
    ):
        self.num_classes = num_classes
        self.focal_loss_enabled = focal_loss_enabled
        self.focal_loss_gamma = focal_loss_gamma
        self.focal_loss_alpha = focal_loss_alpha
        self.attention_layers_num = attention_layers_num
        self.dim_feedforward = dim_feedforward
        self.use_crf = use_crf
        super().__init__(**kwargs)
