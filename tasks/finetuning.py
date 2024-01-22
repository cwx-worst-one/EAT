# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import sys
import torch
import numpy as np
import os


from typing import Optional
from dataclasses import dataclass, field
from omegaconf import MISSING

from fairseq.tasks import register_task
from fairseq.logging import metrics
from sklearn import metrics as sklearn_metrics
from .pretraining_AS2M import MaeImagePretrainingTask,MaeImagePretrainingConfig

from ..data.add_class_target_dataset import AddClassTargetDataset


logger = logging.getLogger(__name__)


@dataclass
class MaeImageClassificationConfig(MaeImagePretrainingConfig):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    input_size: int = 224
    local_cache_path: Optional[str] = None

    rebuild_batches: bool = True
    label_descriptors: str = "label_descriptors.csv"
    labels: str = "lbl"


@register_task("mae_image_classification", dataclass=MaeImageClassificationConfig)
class MaeImageClassificationTask(MaeImagePretrainingTask):
    """ """

    cfg: MaeImageClassificationConfig
    
    def __init__(
        self,
        cfg: MaeImageClassificationConfig,
    ):
        super().__init__(cfg)

        self.state.add_factory("labels", self.load_labels)
        
    def load_labels(self):
        labels = {}
        path = os.path.join(self.cfg.data, self.cfg.label_descriptors)
        with open(path, "r") as ldf:
            for line in ldf:
                if line.strip() == "":
                    continue
                items = line.split(",")
                idx = items[0]
                lbl = items[1]
                assert lbl not in labels, lbl
                labels[lbl] = idx
        return labels

    @property
    def labels(self):
        return self.state.labels


    @classmethod
    def setup_task(cls, cfg: MaeImageClassificationConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (AudioPretrainingConfig): configuration of this task
        """

        return cls(cfg)

    def load_dataset(self, split: str, task_cfg: MaeImageClassificationConfig = None, **kwargs):
        super().load_dataset(split, task_cfg, **kwargs)
        
        data_path = self.cfg.data
        task_cfg = task_cfg or self.cfg
                
        label_path = os.path.join(data_path, f"{split}.{task_cfg.labels}")
        skipped_indices = getattr(self.datasets[split], "skipped_indices", set())
        # print(self.datasets[split].skipped_indices)
        labels = []
        with open(label_path, "r") as f:
            for i, line in enumerate(f):
                if i not in skipped_indices:
                    lbl_items = line.rstrip().split()
                    labels.append([self.state.labels[x] for x in lbl_items[1].split(",")])

        assert len(labels) == len(self.datasets[split]), (
            f"labels length ({len(labels)}) and dataset length "
            f"({len(self.datasets[split])}) do not match"
        )

        self.datasets[split] = AddClassTargetDataset(
            self.datasets[split],
            labels,
            multi_class=True,
            add_to_input=True,
            num_classes=len(self.labels),
        )

    def build_model(self, model_cfg: MaeImageClassificationConfig, from_checkpoint=False):
        model = super().build_model(model_cfg, from_checkpoint)

        actualized_cfg = getattr(model, "cfg", None)
        if actualized_cfg is not None:
            if hasattr(actualized_cfg, "pretrained_model_args"):
                model_cfg.pretrained_model_args = actualized_cfg.pretrained_model_args

        return model
    
    def calculate_stats(self, output, target):

        classes_num = target.shape[-1]
        stats = []
        
        if self.cfg.esc50_eval  or self.cfg.spcv2_eval or not self.cfg.audio_mae:
            
            # Accuracy, only used for single-label classification such as esc-50, not for multiple label one such as AudioSet
            accuracy = sklearn_metrics.accuracy_score(np.argmax(target, 1), np.argmax(output, 1)) 
            dict = {"accuracy": accuracy}
            stats.append(dict)
            
        # Class-wise statistics
        else:
            for k in range(classes_num):
                # Average precision
                avg_precision = sklearn_metrics.average_precision_score(
                    target[:, k], output[:, k], average=None
                )

                dict = {
                    "AP": avg_precision,
                }
                
                stats.append(dict)
        return stats
    
    
    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        return loss, sample_size, logging_output


    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        if "correct" in logging_outputs[0]:
            zero = torch.scalar_tensor(0.0)
            correct = sum(log.get("correct", zero) for log in logging_outputs)
            metrics.log_scalar_sum("_correct", correct)

            metrics.log_derived(
                "accuracy",
                lambda meters: 100 * meters["_correct"].sum / meters["sample_size"].sum,
            )
            
        elif "_predictions" in logging_outputs[0]:
            metrics.log_concat_tensor(
                "_predictions",
                torch.cat([l["_predictions"].cpu() for l in logging_outputs], dim=0),
            )
            metrics.log_concat_tensor(
                "_targets",
                torch.cat([l["_targets"].cpu() for l in logging_outputs], dim=0),
            )

            def compute_stats(meters):
                if meters["_predictions"].tensor.shape[0] < 100:
                    return 0
                stats = self.calculate_stats(
                    meters["_predictions"].tensor, meters["_targets"].tensor
                )
                return np.nanmean([stat["AP"] for stat in stats])

            metrics.log_derived("mAP", compute_stats)            


    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        return None

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return sys.maxsize, sys.maxsize
