import pprint

from absl import app, flags, logging
from streaming import StreamingDataLoader

from src.data.dataset import create_methylformer_streaming_dataset


flags.DEFINE_string("local", None, "local path")
flags.DEFINE_bool("debug", False, "debug mode")
FLAGS = flags.FLAGS

config = {
    # streamingdataset args
    "local": "data/mds/241213-encode_wgbs-train_0_9_val_0_1-ind_tissue/val_10_shards",
    "batch_size": 2,
    # streamingdataset custom args
    "group_idx_name_mapping_path": "data/mds/241213-encode_wgbs-train_0_9_val_0_1-ind_tissue/val_10_shards/group_idx_name_mapping.json",
    # data_preprocessor args
    "gene_expr_df_path": "data/processed/241213-encode_wgbs-train_0_9_val_0_1-ind_tissue/gene_expr.filtered.parquet",
    "sample_idx_path": "data/processed/241213-encode_wgbs-train_0_9_val_0_1-ind_tissue/sample_tissue_count_with_idx.csv",
    "num_nbase": 2000,
    "gene_expr_quantization": True,
    "num_gene_expr_bins": 51,
    "dna_tokenizer_name": "zhihan1996/DNABERT-2-117M",
}


def main(_):
    if FLAGS.local is not None:
        config["local"] = FLAGS.local

    dataset = create_methylformer_streaming_dataset(**config)
    logging.info(f"len(dataset): {len(dataset)}, size: {dataset.size}")
    logging.info(f"dataset: {pprint.pformat(dataset)}")
    logging.info(f"get_item: {pprint.pformat(dataset.get_item(0))}")

    batch_size = config["batch_size"]
    dataloader = StreamingDataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)

    for i, batch in enumerate(dataloader):
        logging.info(f"batch: {pprint.pformat(batch)}")
        shape_dict = {k: v.shape for k, v in batch.items()}
        logging.info(f"shape_dict: {pprint.pformat(shape_dict)}")
        break

    if FLAGS.debug:
        # fmt: off
        import IPython; IPython.embed()
        # fmt: on


if __name__ == "__main__":
    app.run(main)
