import datetime
import json
import math
import multiprocessing as mp
import os.path as osp
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, Optional, Sequence

import pandas as pd
import torch
import torch.nn as nn
from absl import logging
from lightning import LightningModule
from lightning.pytorch.strategies import SingleDeviceStrategy
from omegaconf import DictConfig

from src.eval import MethylEval
from transformers import get_scheduler


@dataclass
class SpeedMonitor:
    config: DictConfig
    start_times: Deque[float] = field(default_factory=lambda: deque([]))
    global_total_tokens: int = 0
    total_training_Gflops: float = 0
    device_interval_tokens: Deque[int] = field(default_factory=lambda: deque([]))
    device_interval_Gflops: Deque[float] = field(default_factory=lambda: deque([]))

    def batch_start(
        self,
        global_total_tokens: int,
        device_batch_num_tokens: int,
        num_fwd_flops: int,
        num_bck_flops: int,
        record: bool = True,
    ) -> None:
        self.global_total_tokens = global_total_tokens
        # num_fwd_flops and num_bck_flops from the OLMo model computes flops per token
        # converting to GFLOPs here prevents numerical issues while logging
        self.total_training_Gflops = (num_fwd_flops + num_bck_flops) * global_total_tokens / 1e9

        if record:
            if len(self.start_times) >= self.config.window_size:
                self.start_times.popleft()
                self.device_interval_tokens.popleft()
                self.device_interval_Gflops.popleft()
            self.start_times.append(time.monotonic())
            self.device_interval_tokens.append(device_batch_num_tokens)
            self.device_interval_Gflops.append((num_fwd_flops + num_bck_flops) * device_batch_num_tokens / 1e9)

    def reset(self) -> None:
        self.start_times.clear()
        self.device_interval_tokens.clear()
        self.device_interval_Gflops.clear()

    def check(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {"throughput/total_tokens": self.global_total_tokens}

        # plot flops related metrics
        metrics["throughput/total_training_Gflops"] = self.total_training_Gflops
        metrics["throughput/total_training_log_Gflops"] = math.log(self.total_training_Gflops)

        if self.start_times:
            interval_seconds = time.monotonic() - self.start_times[0]
            interval_batches = len(self.start_times)
            interval_tokens = sum(self.device_interval_tokens)
            interval_Gflops = sum(self.device_interval_Gflops)
            metrics["throughput/device/tokens_per_second"] = interval_tokens / interval_seconds
            metrics["throughput/device/batches_per_second"] = interval_batches / interval_seconds
            metrics["throughput/device/Gflops_per_second"] = interval_Gflops / interval_seconds
        return metrics


class TrainerModel(LightningModule):
    def __init__(self, config, model):
        super().__init__()
        self.config = config

        self.model = model
        if config.gradient_checkpointing:
            logging.info("Use gradient checkpointing.")
            # NOTE xk: RuntimeError: Expected to mark a variable ready only once.
            # https://github.com/huggingface/accelerate/issues/389
            # https://github.com/huggingface/transformers/issues/23018
            # https://discuss.huggingface.co/t/ddp-gradient-checkpoint-crashes/58432
            self.model.bert.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        self.use_bin_logits_cls_loss = config.get("use_bin_logits_cls_loss", False)
        self.loss_fn = None
        self.num_bins = None
        self.bin_min_val = None
        self.bin_max_val = None
        if not self.use_bin_logits_cls_loss:
            logging.info("Use MSE loss on direct regression value.")
            self.loss_fn = nn.MSELoss(reduction="none")
        else:
            logging.info("Use CrossEntropy loss on bin logits.")
            self.loss_fn = nn.CrossEntropyLoss(reduction="none")
            self.num_bins = config.num_bins
            self.bin_min_val = config.bin_min_val
            self.bin_max_val = config.bin_max_val
            logging.info(
                f"num_bins: {self.num_bins}, bin_min_val: {self.bin_min_val}, bin_max_val: {self.bin_max_val}"
            )

        self.val_outputs = defaultdict(list)
        self.test_outputs = defaultdict(list)

        self.methyl_eval = None

        self.speed_monitor = SpeedMonitor(config.speed_monitor)

        self.__num_fwd_flops = None
        self.__num_bck_flops = None

        self.global_train_examples_seen_this_epoch = 0
        self.global_train_tokens_seen = 0
        self.first_batch = True

        self.eval_save_batch_interval = config.get("eval_save_batch_interval", 100_000)
        self.eval_save_idx = 0

    def on_fit_start(self):
        self.speed_monitor = SpeedMonitor(self.config.speed_monitor)
        self.global_train_examples_seen_this_epoch = 0
        self.global_train_tokens_seen = 0
        self.first_batch = True

    def convert_data_type(self, data):
        if isinstance(data, torch.Tensor):
            if torch.is_floating_point(data):
                return data.float()
            return data
        elif isinstance(data, dict):
            return {key: self.convert_data_type(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.convert_data_type(value) for value in data]
        else:
            return data

    def forward(self, batch):
        batch = self.convert_data_type(batch)

        sample_idx = batch.pop("sample_idx", None)
        if sample_idx is None:
            raise ValueError("sample_idx should not be None.")

        cpg_idx = batch.pop("cpg_idx", None)
        if cpg_idx is None:
            raise ValueError("cpg_idx should not be None.")

        group_idx = batch.pop("group_idx", None)
        if group_idx is None:
            raise ValueError("group_idx should not be None")

        gt_me = batch.pop("methylation", None)

        # FIXME xk: drop __index_level_0__
        # ref: https://discuss.huggingface.co/t/keyerror-index-level-0-error-with-datasets-arrow-writer-py/18082
        # batch.pop("__index_level_0__", None)

        # DEBUG xk: print idx for each node
        # print(
        #     f"Node {self.global_rank} at step {self.trainer.global_step}: sample_idx: {sample_idx}, cpg_idx: {cpg_idx}"
        # )

        outputs = self.model(**batch)
        pred_me = outputs.output_value
        pred_me_bin_logits = outputs.output_bin_logits

        # NOTE xk: shape (batch_size, )
        if len(pred_me.shape) != 1:
            pred_me = pred_me.squeeze(-1)

        loss = None
        if gt_me is not None:
            if not self.use_bin_logits_cls_loss:
                # FIXME xk: detect nan
                # Ref: https://github.com/huggingface/transformers/issues/25065
                # if torch.isnan(pred_me).any():
                #     breakpoint()
                # self.trainer.strategy.barrier()
                # if torch.isnan(cpg_me).any():
                #     breakpoint()
                # self.trainer.strategy.barrier()
                loss = self.loss_fn(pred_me, gt_me)
            else:
                bin_min_val, bin_max_val = self.bin_min_val, self.bin_max_val
                # NOTE: convert the value to the bin index. ` / (self.num_bins - 1)` and `.round()` is important.
                # e.g., [0, 0.1, 0.2, ..., 1.0] -> [0, 1, 2, ..., 100]
                gt_me_bin_idx = (
                    ((gt_me - bin_min_val) / (bin_max_val - bin_min_val) * (self.num_bins - 1)).round().long()
                )
                loss = self.loss_fn(pred_me_bin_logits, gt_me_bin_idx)

        return {
            "pred_me": pred_me,
            "gt_me": gt_me,
            "loss": loss,
            # Used for eval
            "sample_idx": sample_idx,
            "cpg_idx": cpg_idx,
            "group_idx": group_idx,
        }

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        loss_per_batch = outputs["loss"].mean()
        name = "train/"
        if not self.use_bin_logits_cls_loss:
            name += "mse_loss_per_batch"
        else:
            name += "ce_loss_per_batch"
            self.log(
                "train/mse_loss_per_batch",
                torch.nn.functional.mse_loss(outputs["pred_me"], outputs["gt_me"]),
                prog_bar=True,
                rank_zero_only=True,
            )
        self.log(
            name,
            loss_per_batch,
            prog_bar=True,
            rank_zero_only=True,
        )
        self.log("train/global_step", self.trainer.global_step, prog_bar=True, rank_zero_only=True)
        return loss_per_batch

    def on_train_batch_start(self, batch, batch_idx):
        batch_size = batch["tokenized_sequence_input_ids"].shape[0]
        est_seq_len = self.config.estimated_sequence_length_for_flops

        world_size = self.trainer.world_size
        global_batch_size = batch_size * world_size

        self.global_train_examples_seen_this_epoch += global_batch_size
        self.global_train_tokens_seen += global_batch_size * est_seq_len

        self.speed_monitor.batch_start(
            global_total_tokens=self.global_train_tokens_seen,
            device_batch_num_tokens=batch_size * est_seq_len,
            num_fwd_flops=self.num_fwd_flops,
            num_bck_flops=self.num_bck_flops,
            record=not self.first_batch,
        )

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # NOTE: Desription of each hook https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#hooks
        if self.global_step % self.trainer.log_every_n_steps == 0:
            self.log_dict(self.speed_monitor.check(), rank_zero_only=True, prog_bar=True)
        self.first_batch = False

    def test_step(self, batch, batch_idx):
        test_outputs = self._eval_step(batch, batch_idx)
        for k, v in test_outputs.items():
            self.test_outputs[k].append(v)
        return test_outputs

    def validation_step(self, batch, batch_idx):
        validation_outputs = self._eval_step(batch, batch_idx)
        for k, v in validation_outputs.items():
            self.val_outputs[k].append(v)
        return validation_outputs

    def _eval_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        if not self.use_bin_logits_cls_loss:
            loss_per_point = outputs["loss"]
        else:
            loss_per_point = torch.nn.functional.mse_loss(outputs["pred_me"], outputs["gt_me"], reduction="none")
        # NOTE (xk): make sure all_gather can work, so we can flatten dim 0 and 1.
        loss_per_batch = loss_per_point.mean().unsqueeze(0)

        return {
            "mse_loss_per_point": loss_per_point,
            "mse_loss_per_batch": loss_per_batch,
            "pred_methyl": outputs["pred_me"],
            "gt_methyl": outputs["gt_me"],
            # Used for eval
            "cpg_idx": outputs["cpg_idx"],
            "sample_idx": outputs["sample_idx"],
            "group_idx": outputs["group_idx"],
        }

    EXCLUDED_KEYS = ["mse_loss_per_batch"]

    def _on_eval_epoch_end(self, phrase_name: str, group_idx_name_mapping: Dict[str, str]) -> None:
        logging.info(f"[Eval epoch end] phrase_name: {phrase_name}, group_idx_name_mapping: {group_idx_name_mapping}")

        if self.config.full_eval and self.trainer.is_global_zero:
            eval_dir = Path(self.config.eval_dir)
            eval_save_dir = eval_dir / f"eval_results-{phrase_name}.parquet"
            result_df = pd.read_parquet(eval_save_dir)
            result_df.drop_duplicates(subset=["cpg_idx", "sample_idx", "group_idx"], keep="first", inplace=True)

            for group_idx, group_name in group_idx_name_mapping.items():
                group_idx = int(group_idx)
                group_name = osp.basename(group_name)
                if group_name is not None:
                    name = f"eval-{phrase_name}/{group_name}"
                else:
                    name = phrase_name

                group_result_df = result_df[result_df["group_idx"] == group_idx]
                if len(group_result_df) == 0:
                    logging.warning(f"The group_idx {group_idx} has no data during eval.")
                    continue

                self._eval_results(group_result_df, name)

                if self.config.plot_eval_results and self.trainer.is_global_zero:
                    self._plot_eval_results(name)
        # NOTE xk: barrier to make sure all the logs are written in rank zero.
        self.trainer.strategy.barrier()

        # NOTE xk: reset the methyl_eval to None.
        self.methyl_eval = None

    def _gather_outputs(self, outputs):
        # NOTE(xk): We manually synchronize the evaluation step by directly `mean` reducing.
        # See https://lightning.ai/docs/pytorch/stable/accelerators/accelerator_prepare.html#synchronize-validation-and-test-logging.
        # NOTE(xk): to all_reduce, we need to make sure the data is tensor type. Gather all the outputs from all the devices, in shape of [world_size, ...]
        # Other types like `str` will be escaped.
        # NOTE(xk) To debug the dist code:
        # >>> if self.trainer.is_global_zero:
        # >>>     breakpoint()
        # >>> self.trainer.strategy.barrier()
        # NOTE(xk) Use `rank_zero_only` instead of `if self.trainer.is_global_zero:` for model checkpointing.
        # As some strategy needs every process to access the log to monitor the metrics.
        # See: https://github.com/Lightning-AI/pytorch-lightning/issues/15852
        # NOTE xk: In lightning, on_*_epoch_end is called only after all dataloaders are exhausted.

        # NOTE(xk): We only gather the outputs that are not starting with "_", which are scalars.
        gathered_outputs = {k: v for k, v in outputs.items() if not k.startswith("_")}
        batch_outputs = next(iter(gathered_outputs.values()))
        num_batch = len(batch_outputs)
        batch_shape_before_gather = batch_outputs[0].shape

        # NOTE(xk) Debugging dist code
        # if self.trainer.is_global_zero:
        #     breakpoint()
        # self.trainer.strategy.barrier()

        # NOTE(xk): Safe guard, make sure the output is tensor, not scalar. To make the flatten 0 and 1 works.
        # NOTE(xk): Dist dataloader: first pad the whole data, then every proc has the same number of samples, and the length of the last batch is the same, but it may not be equal to the given batch size.
        # The order of each process is: i, i+ batch_size, i + batch_size * 2, ...
        for k, v in gathered_outputs.items():
            if v[0].ndim == 0:
                raise ValueError(
                    f"The output {k} is scalar, but it should be tensor. Otherwise the flatten 0 and 1 does not works."
                )
        # NOTE(xk): All values should have 1-D shape before gathering.
        gathered_outputs = self.all_gather(gathered_outputs)
        # for k,v in gathered_outputs.items(): print(k,v[0].shape)
        # for k,v in gathered_outputs.items(): print(k,type(v))
        # for k,v in gathered_outputs.items(): print(k,v.shape)

        batch_shape_after_gather = next(iter(gathered_outputs.values()))[0].shape
        logging.info(
            f"[Eval epoch end] num_batch: {num_batch}, batch_shape_before_gather: {batch_shape_before_gather}, batch_shape_after_gather: {batch_shape_after_gather}"
        )

        # NOTE(xk) Debugging dist code
        # if self.trainer.is_global_zero:
        #     breakpoint()
        # self.trainer.strategy.barrier()

        strategy_type = type(self.trainer.strategy)
        if isinstance(self.trainer.strategy, SingleDeviceStrategy):
            logging.info(f"Using {strategy_type}, no need to flatten. Assuming (batch_size, ...)")
        else:
            logging.info(f"Using {strategy_type}, flatten the outputs. Assuming (world_size, batch_size, ...)")

        # NOTE(xk): In dist eval, if the last batch does not have the same size as the other batches, the trainer would cycle the last batch to make the same size.
        # Thus, we need to remove the duplicated outputs after all_gather.
        # XXX(xk): Unless the batch size is 1, the samples in the last batch maybe be inconsistent with other batches.
        concat_gathered_outputs = {}
        for k, v in gathered_outputs.items():
            if k in self.EXCLUDED_KEYS:
                continue

            if not isinstance(self.trainer.strategy, SingleDeviceStrategy):
                # Note xk: If the strategy is not SingleDeviceStrategy, the outputs are in the shape of [world_size, batch_size, ...]
                # FIXME xk: Not sure about methods apart from DDP.
                v = [i.flatten(0, 1) for i in v]  # (world_size * batch_size, ...)

            v = torch.cat(v, 0)  # NOTE: Create new tensor obj, the previous list is not touched.

            # NOTE xk: we do not need dataset_size, since it not accurate. We just deduplicate the results with pandas.
            # if dataset_size is not None:
            #     v = v[:dataset_size]
            #     logging.info(f"Flatten {k} shape: {v.shape}, vs. dataset_size: {dataset_size}")
            # else:
            #     logging.info(f"Non-flatten {k} shape: {v.shape} due to dataset_size is None.")
            if v.ndim != 1:
                raise ValueError(
                    f"The output {k} is not 1-dim after flatten. We need 1d tensor to process in the pandas data frame/seris."
                )

            if torch.is_floating_point(v):
                # NOTE(xk): if bf16, numpy raise error
                concat_gathered_outputs[k] = v.float().cpu()
            else:
                # if it is int or long, keep it.
                concat_gathered_outputs[k] = v.cpu()
        return concat_gathered_outputs

    def _eval_results(self, result_df, name):
        tic = time.time()
        logging.info("Perform full evaluation.")

        # NOTE xk: remove rows containing NaN, esp. in the gt_methyl.
        result_df = result_df.dropna()

        eval_save_dir = Path(self.config.eval_dir) / "eval"
        self.methyl_eval = MethylEval(result_df, eval_save_dir, backend="pandas")
        _log_dict = self.methyl_eval.eval_results_to_log_dict()

        self.log_dict(
            # {f"{name}/{k}": torch.mean(v) for k, v in concat_gathered_outputs.items()},
            {f"{name}/{k}": v for k, v in _log_dict.items()},
            prog_bar=False,
            rank_zero_only=True,
        )
        toc = time.time()
        logging.info(f"Full evaluation taking {datetime.timedelta(seconds=toc - tic)}.")

        eval_save_dir = Path(self.config.eval_dir) / "eval"
        eval_save_dir.mkdir(parents=True, exist_ok=True)
        # NOTE xk: the name is the prefix name logging
        # like: "eval-test/train_cpg-val_sample.parquet"
        _name = name.split("/")[-1]
        output_log_dict_path = eval_save_dir / f"log_dict-{_name}.json"
        # NOTE xk: convert the tensor to float for json serialization.
        _log_dict = {k: float(v) for k, v in _log_dict.items()}
        with open(output_log_dict_path, "w") as f:
            json.dump(_log_dict, f, indent=4)
        logging.info(f"Save log_dict to: {output_log_dict_path}")

    def _plot_eval_results(self, name):
        tic = time.time()
        logging.info("Plot evaluation.")

        if self.methyl_eval is None:
            raise ValueError("The methyl_eval is None. Please run the full evaluation first.")
        prefix_name = name.split("/")[-1]
        numpy_image_dict = self.methyl_eval.eval_results_to_plot(prefix_name)

        # https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_image
        pl_logger = self.logger.experiment
        pl_logger_type = self.config.pl_logger_type
        for image_name, grid_images in numpy_image_dict.items():
            try:
                image_name_ = f"{name}/{image_name}"
                if self.config.pl_logger_type == "wandb":
                    self.logger.log_image(
                        key=image_name_,
                        images=[grid_images],
                        caption=[image_name],
                    )
                elif self.config.pl_logger_type == "tensorboard":
                    pl_logger.add_image(
                        image_name_,
                        grid_images,
                        dataformats="HWC",
                        global_step=self.global_step,
                    )
                else:
                    raise ValueError(f"Unknown pl_logger_type: {pl_logger_type}. Check the `main` config.")
            except Exception as e:
                logging.warning(f"Failed to add image to {pl_logger_type}: {e}")

        toc = time.time()
        logging.info(f"Plot evaluation taking {datetime.timedelta(seconds=toc - tic)}.")

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx % self.eval_save_batch_interval != 0 or batch_idx == 0:
            return

        num_batch = None
        try:
            num_batch = len(next(iter(self.test_outputs.values())))
        except StopIteration:
            # NOTE xk: no batch in the outputs. Just skip
            logging.info("No batch in the test_outputs. Skip the evaluation save.")
            return

        logging.info(
            f"Test batch start saving, num_batch: {num_batch}, batch_idx: {batch_idx}, eval_save_idx: {self.eval_save_idx}"
        )
        dataset = self.trainer.test_dataloaders.dataset
        group_idx_name_mapping = dataset.group_idx_name_mapping
        phrase_name = "test"

        concat_gathered_outputs = self._gather_outputs(self.test_outputs)
        if self.trainer.is_global_zero:
            self._eval_save(group_idx_name_mapping, phrase_name, concat_gathered_outputs)
        self.trainer.strategy.barrier()

        self.test_outputs.clear()
        if len(self.test_outputs) != 0:
            raise ValueError("The test_outputs should be empty after the save eval.")

    def _eval_save(self, group_idx_name_mapping, phrase_name, concat_gathered_outputs):
        tic = time.time()
        logging.info("Save the evaluation results.")

        result_df = pd.DataFrame.from_dict(concat_gathered_outputs)

        logging.info("Start deduplicate the results.")
        result_df.drop_duplicates(subset=["cpg_idx", "sample_idx", "group_idx"], keep="first", inplace=True)
        logging.info("End deduplicate the results.")

        eval_dir = Path(self.config.eval_dir)
        eval_save_dir = eval_dir / f"eval_results-{phrase_name}.parquet"
        eval_save_dir.mkdir(parents=True, exist_ok=True)
        eval_save_file_path = eval_save_dir / f"{self.eval_save_idx:06d}.parquet"
        result_df.to_parquet(eval_save_file_path)
        # NOTE xk: eval_save_idx is the index of the next file.
        self.eval_save_idx += 1

        group_idx_name_mapping_path = eval_dir / f"group_idx_name_mapping-{phrase_name}.json"
        with open(group_idx_name_mapping_path, "w") as f:
            json.dump(group_idx_name_mapping, f, indent=4)

        # Save test dataloader state after the test ends.
        test_dataloader = self.trainer.test_dataloaders
        if test_dataloader is not None:
            test_dataloader_ckpt_path = eval_dir / "test_dataloader.json"
            with open(test_dataloader_ckpt_path, "w") as f:
                json.dump(test_dataloader.state_dict(), f, indent=4)
            eval_save_idx_path = eval_dir / f"eval_save_idx-{phrase_name}.json"
            with open(eval_save_idx_path, "w") as f:
                json.dump({"eval_save_idx": self.eval_save_idx}, f, indent=4)

        toc = time.time()
        logging.info(f"Save df parquet taking {datetime.timedelta(seconds=toc - tic)}: {eval_save_file_path}.")

    def on_test_epoch_end(self) -> None:
        # NOTE xk: In lightning, on_*_epoch_end is called only after all dataloaders are exhausted.
        # XXX(xk) To log different ckpt with their name showed in the result. See the call of `trainer.test` in `src/main.py`
        # NOTE xk: https://lightning.ai/docs/pytorch/stable/data/access.html

        num_batch = None
        try:
            num_batch = len(next(iter(self.test_outputs.values())))
        except StopIteration:
            # NOTE xk: no batch in the outputs, but we may compute eval then.
            pass

        logging.info(
            f"Test epoch end, save the rest of the results: num_batch: {num_batch}, eval_save_idx: {self.eval_save_idx}"
        )
        dataset = self.trainer.test_dataloaders.dataset
        group_idx_name_mapping = dataset.group_idx_name_mapping
        phrase_name = "test"

        if num_batch is not None:
            concat_gathered_outputs = self._gather_outputs(self.test_outputs)
            if self.trainer.is_global_zero:
                self._eval_save(group_idx_name_mapping, phrase_name, concat_gathered_outputs)
            self.trainer.strategy.barrier()

        # NOTE xk: reset the eval_save_idx to 0.
        self.eval_save_idx = 0
        self.test_outputs.clear()
        if len(self.test_outputs) != 0:
            raise ValueError("The test_outputs should be empty after the evaluation.")

        self._on_eval_epoch_end("test", group_idx_name_mapping)

    def on_validation_epoch_end(self) -> None:
        # NOTE xk: In lightning, on_*_epoch_end is called only after all dataloaders are exhausted.
        logging.info("Validation epoch end")

        dataset = self.trainer.val_dataloaders.dataset
        group_idx_name_mapping = dataset.group_idx_name_mapping
        phrase_name = "val"

        concat_gathered_outputs = self._gather_outputs(self.val_outputs)
        if self.trainer.is_global_zero:
            self._eval_save(group_idx_name_mapping, phrase_name, concat_gathered_outputs)
        self.trainer.strategy.barrier()

        self.eval_save_idx = 0
        self.val_outputs.clear()
        if len(self.val_outputs) != 0:
            raise ValueError("The val_outputs should be empty after the evaluation.")

        self._on_eval_epoch_end("val", group_idx_name_mapping)

    def on_validation_start(self):
        if isinstance(self.trainer.val_dataloaders, Sequence):
            raise ValueError(
                f"The val_dataloaders should be a single dataloader, not a list of dataloaders {self.trainer.val_dataloaders}."
            )

    def on_test_start(self):
        if isinstance(self.trainer.test_dataloaders, Sequence):
            raise ValueError(
                f"The test_dataloaders should be a single dataloader, not a list of dataloaders {self.trainer.test_dataloaders}."
            )

    def on_test_epoch_start(self):
        test_dataloader = self.trainer.test_dataloaders
        if test_dataloader is not None:
            phrase_name = "test"
            eval_dir = Path(self.config.eval_dir)
            test_dataloader_ckpt_path = eval_dir / "test_dataloader.json"
            if test_dataloader_ckpt_path.exists():
                with open(test_dataloader_ckpt_path, "r") as f:
                    test_dataloader.load_state_dict(json.load(f))
                logging.info(f"Test start. Resuming dataloader state from {test_dataloader_ckpt_path}.")
            eval_save_idx_path = eval_dir / f"eval_save_idx-{phrase_name}.json"
            if eval_save_idx_path.exists():
                with open(eval_save_idx_path, "r") as f:
                    self.eval_save_idx = json.load(f)["eval_save_idx"]
                logging.info(f"Test start. Resuming eval_save_idx from {eval_save_idx_path}.")

    def configure_optimizer_params(self):
        # NOTE xk: Copied from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L215
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.model.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = dict(self.model.named_parameters())
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": self.config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]

        # Log the number of parameters in each group
        num_decay_params = len(optim_groups[0]["params"])
        num_non_decay_params = len(optim_groups[1]["params"])
        logging.info(
            f"weight_decay: {self.config.weight_decay}, num_decay_params: {num_decay_params}, num_non_decay_params: {num_non_decay_params}"
        )

        return optim_groups

    def configure_optimizers(self):
        logging.info(f"Optimizer: AdamW, lr: {self.config.learning_rate}, betas: {self.config.betas}")
        optimizer = torch.optim.AdamW(
            self.configure_optimizer_params(),
            lr=self.config.learning_rate,
            betas=self.config.betas,
        )

        # NOTE: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers
        # if self.config.scheduler_type.startswith("cosine") and self.config.scheduler_num_training_steps is None:
        #     raise ValueError("If you use cosine scheduler, you need to provide the scheduler_num_training_steps.")
        scheduler_type = self.config.scheduler_type
        scheduler_specific_kwargs = None
        if scheduler_type == "cosine_with_min_lr":
            scheduler_specific_kwargs = {"min_lr_rate": self.config.min_lr_rate}
        lr_scheduler = get_scheduler(
            name=self.config.scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.config.scheduler_num_warmup_steps,
            num_training_steps=self.config.scheduler_num_training_steps,
            scheduler_specific_kwargs=scheduler_specific_kwargs,
        )
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [lr_scheduler_config]

    def on_validation_end(self):
        # Save the train dataloader state after the validation ends.
        local_rank = self.trainer.local_rank
        train_dataloader = self.trainer.train_dataloader
        if train_dataloader is None:
            return

        ckpt_dir = Path(self.config.ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        train_dataloader_ckpt_path = ckpt_dir / f"train_dataloader-local_rank_{local_rank}.json"
        with open(train_dataloader_ckpt_path, "w") as f:
            json.dump(train_dataloader.state_dict(), f, indent=4)

        logging.info(f"Validation end. Saving dataloader state to {train_dataloader_ckpt_path}.")

        speed_monitor_metrics = {
            "global_train_examples_seen_this_epoch": self.global_train_examples_seen_this_epoch,
            "global_train_tokens_seen": self.global_train_tokens_seen,
        }
        with open(ckpt_dir / f"speed_monitor_metrics-local_rank_{local_rank}.json", "w") as f:
            json.dump(speed_monitor_metrics, f, indent=4)

        # Reset speed monitor so that we don't count the time taken to save checkpoints.
        self.speed_monitor.reset()

    def on_train_start(self):
        local_rank = self.trainer.local_rank
        train_dataloader = self.trainer.train_dataloader
        if train_dataloader is None:
            raise ValueError("The train_dataloader should not be None.")

        train_dataloader_ckpt_path = Path(self.config.ckpt_dir) / f"train_dataloader-local_rank_{local_rank}.json"
        if train_dataloader_ckpt_path.exists():
            with open(train_dataloader_ckpt_path, "r") as f:
                train_dataloader.load_state_dict(json.load(f))

            logging.info(f"Train start. Resuming dataloader state from {train_dataloader_ckpt_path}.")

        speed_monitor_metrics_path = Path(self.config.ckpt_dir) / f"speed_monitor_metrics-local_rank_{local_rank}.json"
        if speed_monitor_metrics_path.exists():
            with open(speed_monitor_metrics_path, "r") as f:
                speed_monitor_metrics = json.load(f)

            self.global_train_examples_seen_this_epoch = speed_monitor_metrics["global_train_examples_seen_this_epoch"]
            self.global_train_tokens_seen = speed_monitor_metrics["global_train_tokens_seen"]

            logging.info(
                f"Train start. Resuming speed monitor metrics from {speed_monitor_metrics_path}: {speed_monitor_metrics}"
            )

    # NOTE xk: customize the FLOPS computation. include_prefix="bert". As MLP do not have `seq` dim.
    @property
    def num_fwd_flops(self):
        if self.__num_fwd_flops:
            return self.__num_fwd_flops

        # embedding table is just a lookup in the forward pass
        n_params = self.num_params(include_embedding=False, include_prefix="bert")
        # the number of parameters is approximately the number of multiply-accumulates (MAC) in the network
        # each MAC has 2 FLOPs - we multiply by 2 ie 2 * n_param
        # this gets us FLOPs / token
        params_flops_per_token = 2 * n_params

        # NOTE xk: customize according to config
        n_layers = self.model.config.bert_config_dict["num_hidden_layers"]
        d_model = self.model.config.bert_config_dict["hidden_size"]
        max_sequence_length = self.config.estimated_sequence_length_for_flops
        logging.warning(
            f"update your config.trainer_model.estimated_sequence_length_for_flops accordingly: now is {max_sequence_length}."
        )

        # there are 2 FLOPS per mac; there is A=Q*K^T and out=A*V ops (ie mult by 2)
        attn_flops_per_token = n_layers * 2 * 2 * (d_model * max_sequence_length)
        self.__num_fwd_flops = params_flops_per_token + attn_flops_per_token
        return self.__num_fwd_flops

    # NOTE xk: customize the FLOPS computation. include_prefix="bert". As MLP do not have `seq` dim.
    @property
    def num_bck_flops(self):
        if self.__num_bck_flops:
            return self.__num_bck_flops

        n_params = self.num_params(include_embedding="bert")
        params_flops_per_token = 4 * n_params

        # NOTE xk: customize according to config
        n_layers = self.model.config.bert_config_dict["num_hidden_layers"]
        d_model = self.model.config.bert_config_dict["hidden_size"]
        max_sequence_length = self.config.estimated_sequence_length_for_flops
        logging.warning(
            f"update your config.trainer_model.estimated_sequence_length_for_flops accordingly: now is {max_sequence_length}."
        )
        attn_flops_per_token = n_layers * 8 * (d_model * max_sequence_length)

        self.__num_bck_flops = params_flops_per_token + attn_flops_per_token
        return self.__num_bck_flops

    def num_params(self, include_embedding: bool = True, include_prefix="") -> int:
        """
        Get the total number of parameters.
        """
        params = {}
        for mn, m in self.model.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                if not fpn.startswith(include_prefix):
                    continue
                if isinstance(m, nn.Embedding) and not include_embedding:
                    logging.info(f"num_params: skip embedding {fpn}")
                if fpn.endswith("bias") or fpn.endswith("weight"):
                    params[fpn] = p
        return sum(p.numel() for _, p in params.items())
