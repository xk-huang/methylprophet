import torch
from convnextv2_1d.configuration_convnextv2_1d import ConvNextV21DConfig
from convnextv2_1d.modeling_convnextv2_1d import ConvNextV21DModel


def main():
    batch_size = 7
    configs = {
        "gene_expr": {"num_channels": 1, "num_seq": 17284},
        "onehot_sequence": {"num_channels": 5, "num_seq": 1200},
    }

    for key, config in configs.items():
        print(f"\n{key}")
        num_channels = config["num_channels"]
        num_seq = config["num_seq"]

        data = torch.randn(batch_size, num_channels, num_seq)

        config_dict = {
            "num_channels": num_channels,
            "patch_size": 4,
            "num_stages": 4,
            "hidden_sizes": [96, 192, 384, 768],
            "depths": [3, 3, 9, 3],
            "hidden_act": "gelu",
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-12,
            "drop_path_rate": 0.0,
            "image_size": 224,
            "out_features": None,
            "out_indices": None,
        }
        configuration = ConvNextV21DConfig(**config_dict)
        model = ConvNextV21DModel(configuration)

        print("Input shape:", data.shape)
        outputs = model(data)
        print("Output shapes:")
        for k, v in outputs.items():
            print(k, v.shape)


if __name__ == "__main__":
    main()
