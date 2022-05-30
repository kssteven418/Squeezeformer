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

import os
import json
import abc
from typing import Union
import tqdm
import numpy as np
import tensorflow as tf

from ..featurizers.speech_featurizers import (
    load_and_convert_to_wav, 
    read_raw_audio, 
    tf_read_raw_audio, 
    TFSpeechFeaturizer
)
from ..featurizers.text_featurizers import TextFeaturizer
from ..utils import feature_util, file_util, math_util, data_util

logger = tf.get_logger()


BUFFER_SIZE = 10000
AUTOTUNE = tf.data.experimental.AUTOTUNE


class BaseDataset(metaclass=abc.ABCMeta):
    """ Based dataset for all models """

    def __init__(
        self,
        data_paths: list,
        cache: bool = False,
        shuffle: bool = False,
        buffer_size: int = BUFFER_SIZE,
        indefinite: bool = False,
        drop_remainder: bool = True,
        stage: str = "train",
        **kwargs,
    ):
        self.data_paths = data_paths or []
        if not isinstance(self.data_paths, list):
            raise ValueError('data_paths must be a list of string paths')
        self.cache = cache  # whether to cache transformed dataset to memory
        self.shuffle = shuffle  # whether to shuffle tf.data.Dataset
        if buffer_size <= 0 and shuffle:
            raise ValueError("buffer_size must be positive when shuffle is on")
        self.buffer_size = buffer_size  # shuffle buffer size
        self.stage = stage  # for defining tfrecords files
        self.drop_remainder = drop_remainder  # whether to drop remainder for multi gpu training
        self.indefinite = indefinite  # Whether to make dataset repeat indefinitely -> avoid the potential last partial batch
        self.total_steps = None  # for better training visualization

    @abc.abstractmethod
    def parse(self, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def create(self, batch_size):
        raise NotImplementedError()


class ASRDataset(BaseDataset):
    """ Dataset for ASR using Generator """

    def __init__(
        self,
        stage: str,
        speech_featurizer: TFSpeechFeaturizer,
        text_featurizer: TextFeaturizer,
        data_paths: list,
        cache: bool = False,
        shuffle: bool = False,
        indefinite: bool = False,
        drop_remainder: bool = True,
        buffer_size: int = BUFFER_SIZE,
        input_padding_length: int = 3300,
        label_padding_length: int = 530,
        **kwargs,
    ):
        super().__init__(
            data_paths=data_paths, 
            cache=cache, shuffle=shuffle, stage=stage, buffer_size=buffer_size,
            drop_remainder=drop_remainder, indefinite=indefinite
        )
        self.speech_featurizer = speech_featurizer
        self.text_featurizer = text_featurizer
        self.input_padding_length = input_padding_length
        self.label_padding_length = label_padding_length


    # -------------------------------- ENTRIES -------------------------------------

    def read_entries(self):
        if hasattr(self, "entries") and len(self.entries) > 0: return
        self.entries = []
        for file_path in self.data_paths:
            logger.info(f"Reading {file_path} ...")
            with tf.io.gfile.GFile(file_path, "r") as f:
                temp_lines = f.read().splitlines()
                # Skip the header of tsv file
                self.entries += temp_lines[1:]
        # The files is "\t" seperated
        self.entries = [line.split("\t", 2) for line in self.entries]
        for i, line in enumerate(self.entries):
            self.entries[i][-1] = " ".join([str(x) for x in self.text_featurizer.extract(line[-1]).numpy()])
        self.entries = np.array(self.entries)
        if self.shuffle: np.random.shuffle(self.entries)  # Mix transcripts.tsv
        self.total_steps = len(self.entries)

    # -------------------------------- LOAD AND PREPROCESS -------------------------------------

    def generator(self):
        for path, _, indices in self.entries:
            audio = load_and_convert_to_wav(path).numpy()
            yield bytes(path, "utf-8"), audio, bytes(indices, "utf-8")


    def tf_preprocess(self, path: tf.Tensor, audio: tf.Tensor, indices: tf.Tensor):
        with tf.device("/CPU:0"):
            signal = tf_read_raw_audio(audio, self.speech_featurizer.sample_rate)
            features = self.speech_featurizer.tf_extract(signal)
            input_length = tf.cast(tf.shape(features)[0], tf.int32)

            label = tf.strings.to_number(tf.strings.split(indices), out_type=tf.int32)
            label_length = tf.cast(tf.shape(label)[0], tf.int32)

            prediction = self.text_featurizer.prepand_blank(label)
            prediction_length = tf.cast(tf.shape(prediction)[0], tf.int32)

            return path, features, input_length, label, label_length, prediction, prediction_length

    def parse(self, path: tf.Tensor, audio: tf.Tensor, indices: tf.Tensor):
        """
        Returns:
            path, features, input_lengths, labels, label_lengths, pred_inp
        """
        data = self.tf_preprocess(path, audio, indices)
        _, features, input_length, label, label_length, prediction, prediction_length = data
        return (
            data_util.create_inputs(
                inputs=features,
                inputs_length=input_length,
                predictions=prediction,
                predictions_length=prediction_length
            ),
            data_util.create_labels(
                labels=label,
                labels_length=label_length
            )
        )


    def process(self, dataset, batch_size):
        dataset = dataset.map(self.parse, num_parallel_calls=AUTOTUNE)
        self.total_steps = math_util.get_num_batches(self.total_steps, batch_size, drop_remainders=self.drop_remainder)
        if self.cache:
            dataset = dataset.cache()

        if self.shuffle:
            dataset = dataset.shuffle(self.buffer_size, reshuffle_each_iteration=True)

        if self.indefinite and self.total_steps:
            dataset = dataset.repeat()

        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=(
                data_util.create_inputs(
                    inputs=tf.TensorShape([self.input_padding_length, 80, 1]),
                    inputs_length=tf.TensorShape([]),
                    predictions=tf.TensorShape([self.label_padding_length]),
                    predictions_length=tf.TensorShape([])
                ),
                data_util.create_labels(
                    labels=tf.TensorShape([self.label_padding_length]),
                    labels_length=tf.TensorShape([])
                ),
            ),
            padding_values=(
                data_util.create_inputs(
                    inputs=0.0,
                    inputs_length=0,
                    predictions=self.text_featurizer.blank,
                    predictions_length=0
                ),
                data_util.create_labels(
                    labels=self.text_featurizer.blank,
                    labels_length=0
                )
            ),
            drop_remainder=self.drop_remainder
        )

        # PREFETCH to improve speed of input length
        dataset = dataset.prefetch(AUTOTUNE)
        return dataset

    def create(self, batch_size: int):
        self.read_entries()
        if not self.total_steps or self.total_steps == 0: return print("Couldn't create")
        dataset = tf.data.Dataset.from_generator(
            self.generator,
            output_types=(tf.string, tf.string, tf.string),
            output_shapes=(tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([]))
        )
        return self.process(dataset, batch_size)


class ASRSliceDataset(ASRDataset):
    """ Dataset for ASR using Slice """

    @staticmethod
    def load(record: tf.Tensor):
        def fn(path: bytes): return load_and_convert_to_wav(path.decode("utf-8")).numpy()
        audio = tf.numpy_function(fn, inp=[record[0]], Tout=tf.string)
        return record[0], audio, record[2]

    def create(self, batch_size: int):
        self.read_entries()
        if not self.total_steps or self.total_steps == 0: return None
        dataset = tf.data.Dataset.from_tensor_slices(self.entries)
        dataset = dataset.map(self.load, num_parallel_calls=AUTOTUNE)
        return self.process(dataset, batch_size)

    def preprocess_dataset(self, tfrecord_path, shard_size=0, max_len=None):
        self.read_entries()
        if not self.total_steps or self.total_steps == 0: return None
        logger.info(f"Preprocess dataset")
        dataset = tf.data.Dataset.from_tensor_slices(self.entries)
        dataset = dataset.map(self.load, num_parallel_calls=AUTOTUNE)
        self.create_preprocessed_tfrecord(dataset, tfrecord_path, shard_size, max_len)
