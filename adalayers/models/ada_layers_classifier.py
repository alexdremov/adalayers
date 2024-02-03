import transformers
from transformers import PreTrainedModel

from adalayers.models.configs import AdaLayersForSequenceClassificationConfig


class AdaLayersForSequenceClassification(PreTrainedModel):
    config_class = AdaLayersForSequenceClassificationConfig

    def __init__(self, config: AdaLayersForSequenceClassificationConfig):
        super().__init__(config)
        self.model = transformers.AutoModel.from_pretrained(config.base_model)
