import torch

from transformers import AutoTokenizer, DistilBertConfig, DistilBertModel


def main():
    device = "cuda"
    dtype = torch.bfloat16
    num_position_embeddings = 20000

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    config = DistilBertConfig(
        _attn_implementation="flash_attention_2",
        # _attn_implementation="eager",
    )
    model = DistilBertModel(config)

    model.resize_position_embeddings(num_position_embeddings)
    model.embeddings.position_ids = torch.arange(num_position_embeddings).expand((1, -1))

    model = model.to(device=device, dtype=dtype)

    inputs = tokenizer("Hello, my dog is cute" * 500, return_tensors="pt").to(device=device)
    outputs = model(**inputs)

    print(f"Number of parameters: {model.num_parameters() / 1e6:.2f}M")
    breakpoint()


if __name__ == "__main__":
    main()
