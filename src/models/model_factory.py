from src.models.methylformer_bert import MethylformerBertConfig, MethylformerBertModel


def create_model_class(model_class_name):
    if model_class_name in MODEL_NAME_TO_MODEL_CLASS:
        return MODEL_NAME_TO_MODEL_CLASS[model_class_name]
    else:
        raise ValueError(f"Model {model_class_name} not found.")


def create_model_config_class(model_config_class_name):
    if model_config_class_name in MODEL_CONFIG_NAME_TO_MODEL_CONFIG_CLASS:
        return MODEL_CONFIG_NAME_TO_MODEL_CONFIG_CLASS[model_config_class_name]
    else:
        raise ValueError(f"Model config {model_config_class_name} not found.")


MODEL_NAME_TO_MODEL_CLASS = {
    "MethylformerBertModel": MethylformerBertModel,
}
MODEL_CONFIG_NAME_TO_MODEL_CONFIG_CLASS = {
    "MethylformerBertConfig": MethylformerBertConfig,
}
