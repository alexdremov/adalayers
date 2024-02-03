from transformers import PretrainedConfig


class AdaLayersForSequenceClassificationConfig(PretrainedConfig):
    model_type = "ada_layers_classifier"

    def __init__(
            self,
            base_model: str,
            **kwargs,
    ):
        self.base_model = base_model
        super().__init__(**kwargs)
