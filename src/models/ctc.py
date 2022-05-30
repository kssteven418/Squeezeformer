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

from typing import Dict, Union
import numpy as np
import tensorflow as tf

from .base_model import BaseModel
from ..featurizers.speech_featurizers import TFSpeechFeaturizer
from ..featurizers.text_featurizers import TextFeaturizer
from ..utils import math_util, shape_util, data_util
from ..losses.ctc_loss import CtcLoss

logger = tf.get_logger()


class CtcModel(BaseModel):
    def __init__(
        self,
        encoder: tf.keras.Model,
        decoder: Union[tf.keras.Model, tf.keras.layers.Layer] = None,
        augmentation: tf.keras.Model = None,
        vocabulary_size: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        if decoder is None:
            assert vocabulary_size is not None, "vocabulary_size must be set"
            self.decoder = tf.keras.layers.Dense(units=vocabulary_size, name=f"{self.name}_logits")
        else:
            self.decoder = decoder
        self.augmentation = augmentation
        self.time_reduction_factor = 1
	
    def make(self, input_shape, batch_size=None):
        inputs = tf.keras.Input(input_shape, batch_size=batch_size, dtype=tf.float32)
        inputs_length = tf.keras.Input(shape=[], batch_size=batch_size, dtype=tf.int32)
        self(
            data_util.create_inputs(
                inputs=inputs,
                inputs_length=inputs_length
            ),
            training=False
        )

    def compile(self, optimizer, blank=0, run_eagerly=None, **kwargs):
        loss = CtcLoss(blank=blank)
        super().compile(loss=loss, optimizer=optimizer, run_eagerly=run_eagerly, **kwargs)

    def add_featurizers(
        self,
        speech_featurizer: TFSpeechFeaturizer,
        text_featurizer: TextFeaturizer,
    ):
        self.speech_featurizer = speech_featurizer
        self.text_featurizer = text_featurizer

    def call(self, inputs, training=False, **kwargs):
        x, x_length = inputs["inputs"], inputs["inputs_length"]
        if training and self.augmentation is not None:
            x = self.augmentation(x, x_length)
        logits = self.encoder(x, x_length, training=training, **kwargs)
        logits = self.decoder(logits, training=training, **kwargs)
        return data_util.create_logits(
            logits=logits,
            logits_length=math_util.get_reduced_length(x_length, self.time_reduction_factor)
        )

    # -------------------------------- GREEDY -------------------------------------
    @tf.function
    def recognize_from_logits(self, logits: tf.Tensor, lengths: tf.Tensor):
        probs = tf.nn.softmax(logits)
        # blank is in the first index of `probs`, where `ctc_greedy_decoder` supposes it to be in the last index.
        # threfore, we move the first column to the last column to be compatible with `ctc_greedy_decoder`
        probs = tf.concat([probs[:, :, 1:], tf.expand_dims(probs[:, :, 0], -1)], axis=-1)
        def _map(elems): return tf.numpy_function(self._perform_greedy, inp=[elems[0], elems[1]], Tout=tf.string)

        return tf.map_fn(_map, (probs, lengths), fn_output_signature=tf.TensorSpec([], dtype=tf.string))
    

    @tf.function
    def recognize(self, inputs: Dict[str, tf.Tensor]):
        logits = self(inputs, training=False)
        probs = tf.nn.softmax(logits["logits"])
        # send the first index (skip token) to the last index
        # for compatibility with the ctc_decoders library
        probs = tf.concat([probs[:, :, 1:], tf.expand_dims(probs[:, :, 0], -1)], axis=-1)
        lengths = logits["logits_length"]

        def map_fn(elem): return tf.numpy_function(self._perform_greedy, inp=[elem[0], elem[1]], Tout=tf.string)

        return tf.map_fn(map_fn, [probs, lengths], fn_output_signature=tf.TensorSpec([], dtype=tf.string))

    def _perform_greedy(self, probs: np.ndarray, length):
        from ctc_decoders import ctc_greedy_decoder
        decoded = ctc_greedy_decoder(probs[:length], vocabulary=self.text_featurizer.non_blank_tokens)
        return tf.convert_to_tensor(decoded, dtype=tf.string)

    # -------------------------------- BEAM SEARCH -------------------------------------

    @tf.function
    def recognize_beam(self, inputs: Dict[str, tf.Tensor], lm: bool = False):
        logits = self(inputs, training=False)
        probs = tf.nn.softmax(logits["logits"])

        def map_fn(prob): return tf.numpy_function(self._perform_beam_search, inp=[prob, lm], Tout=tf.string)

        return tf.map_fn(map_fn, probs, dtype=tf.string)

    def _perform_beam_search(self, probs: np.ndarray, lm: bool = False):
        from ctc_decoders import ctc_beam_search_decoder
        decoded = ctc_beam_search_decoder(
            probs_seq=probs,
            vocabulary=self.text_featurizer.non_blank_tokens,
            beam_size=self.text_featurizer.decoder_config.beam_width,
            ext_scoring_func=self.text_featurizer.scorer if lm else None
        )
        decoded = decoded[0][-1]

        return tf.convert_to_tensor(decoded, dtype=tf.string)
