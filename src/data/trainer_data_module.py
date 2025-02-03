from pprint import pformat

from absl import logging
from lightning import LightningDataModule
from streaming import StreamingDataLoader

from src.data.dataset import create_methylformer_streaming_dataset


class TrainerDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.train_dataset_config = config.train_dataset
        self.val_dataset_config = config.val_dataset
        logging.info(f"train_dataset_config: {pformat(self.train_dataset_config)}")
        logging.info(f"val_dataset_config: {pformat(self.val_dataset_config)}")

        self.train_dataloader_config = config.train_dataloader
        self.val_dataloader_config = config.val_dataloader
        logging.info(f"train_dataloader_config: {pformat(self.train_dataloader_config)}")
        logging.info(f"val_dataloader_config: {pformat(self.val_dataloader_config)}")

        # NOTE xk: For batch size finder
        # https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html#batch-size-finder
        self.batch_size = config.train_dataloader.batch_size

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_dataset = create_methylformer_streaming_dataset(**self.train_dataset_config)
        else:
            train_dataset = None
        val_dataset = create_methylformer_streaming_dataset(**self.val_dataset_config)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def train_dataloader(self):
        # NOTE xk: For batch size finder
        # https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html#batch-size-finder
        self.train_dataloader_config.batch_size = self.batch_size

        # NOTE xk: Do no use shuffle in DataLoader, use shuffle in StreamingDataLoader.
        # train_dataset is an IterableDataset, it does not support shuffling.
        # Shuffling is controlled by StreamingDataset.
        # https://github.com/mosaicml/streaming/issues/419#issuecomment-1710873932
        train_dataloader = StreamingDataLoader(
            dataset=self.train_dataset,
            **self.train_dataloader_config,
            collate_fn=self.train_dataset.collate_fn,
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = StreamingDataLoader(
            dataset=self.val_dataset,
            **self.val_dataloader_config,
            collate_fn=self.val_dataset.collate_fn,
        )
        return val_dataloader

    def test_dataloader(self):
        # Use val_dataloader for testing
        return self.val_dataloader()
