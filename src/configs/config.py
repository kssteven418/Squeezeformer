# Copyright 2020 Huy Le Nguyen (@usimarit)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union
from ..utils import file_util


class DatasetConfig:
    def __init__(self, config: dict = None):
        if not config: config = {}
        self.stage = config.pop("stage", None)
        self.data_paths = file_util.preprocess_paths(config.pop("data_paths", None))
        self.tfrecords_dir = file_util.preprocess_paths(config.pop("tfrecords_dir", None), isdir=True)
        self.tfrecords_shards = config.pop("tfrecords_shards", 16)
        self.shuffle = config.pop("shuffle", False)
        self.cache = config.pop("cache", False)
        self.drop_remainder = config.pop("drop_remainder", True)
        self.buffer_size = config.pop("buffer_size", 10000)
        for k, v in config.items(): setattr(self, k, v)


class RunningConfig:
    def __init__(self, config: dict = None):
        if not config: config = {}
        self.batch_size = config.pop("batch_size", 1)
        self.accumulation_steps = config.pop("accumulation_steps", 1)
        self.num_epochs = config.pop("num_epochs", 20)
        for k, v in config.items(): setattr(self, k, v)


class LearningConfig:
    def __init__(self, config: dict = None):
        if not config: config = {}
        self.train_dataset_config = DatasetConfig(config.pop("train_dataset_config", {}))
        self.eval_dataset_config = DatasetConfig(config.pop("eval_dataset_config", {}))
        self.test_dataset_config = DatasetConfig(config.pop("test_dataset_config", {}))
        self.optimizer_config = config.pop("optimizer_config", {})
        self.running_config = RunningConfig(config.pop("running_config", {}))
        for k, v in config.items(): setattr(self, k, v)


class Config:
    """ User config class for training, testing or infering """

    def __init__(self, data: Union[str, dict]):
        config = data if isinstance(data, dict) else file_util.load_yaml(file_util.preprocess_paths(data))
        self.speech_config = config.pop("speech_config", {})
        self.decoder_config = config.pop("decoder_config", {})
        self.model_config = config.pop("model_config", {})
        self.learning_config = LearningConfig(config.pop("learning_config", {}))
        for k, v in config.items(): setattr(self, k, v)
