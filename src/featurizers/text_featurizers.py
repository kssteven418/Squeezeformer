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
import abc
import codecs
import unicodedata
from multiprocessing import cpu_count
import sentencepiece as sp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tds

from ..utils import file_util

ENGLISH_CHARACTERS = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
                      "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]


class TextFeaturizer(metaclass=abc.ABCMeta):
    def __init__(self):
        self.scorer = None
        self.blank = None
        self.tokens2indices = {}
        self.tokens = []
        self.num_classes = None
        self.max_length = 0

    @property
    def shape(self) -> list:
        return [self.max_length if self.max_length > 0 else None]

    @property
    def prepand_shape(self) -> list:
        return [self.max_length + 1 if self.max_length > 0 else None]

    def update_length(self, length: int):
        self.max_length = max(self.max_length, length)

    def reset_length(self):
        self.max_length = 0

    def preprocess_text(self, text):
        text = unicodedata.normalize("NFC", text.lower())
        return text.strip("\n")  # remove trailing newline

    def add_scorer(self, scorer: any = None):
        """ Add scorer to this instance """
        self.scorer = scorer

    def normalize_indices(self, indices: tf.Tensor) -> tf.Tensor:
        """
        Remove -1 in indices by replacing them with blanks
        Args:
            indices (tf.Tensor): shape any

        Returns:
            tf.Tensor: normalized indices with shape same as indices
        """
        with tf.name_scope("normalize_indices"):
            minus_one = -1 * tf.ones_like(indices, dtype=tf.int32)
            blank_like = self.blank * tf.ones_like(indices, dtype=tf.int32)
            return tf.where(indices == minus_one, blank_like, indices)

    def prepand_blank(self, text: tf.Tensor) -> tf.Tensor:
        """ Prepand blank index for transducer models """
        return tf.concat([[self.blank], text], axis=0)

    @abc.abstractclassmethod
    def extract(self, text):
        raise NotImplementedError()

    @abc.abstractclassmethod
    def iextract(self, indices):
        raise NotImplementedError()

    @abc.abstractclassmethod
    def indices2upoints(self, indices):
        raise NotImplementedError()


class SentencePieceFeaturizer(TextFeaturizer):
    """
    Extract text feature based on sentence piece package.
    """
    UNK_TOKEN, UNK_TOKEN_ID = "<unk>", 1
    BOS_TOKEN, BOS_TOKEN_ID = "<s>", 2
    EOS_TOKEN, EOS_TOKEN_ID = "</s>", 3
    PAD_TOKEN, PAD_TOKEN_ID = "<pad>", 0  # unused, by default

    def __init__(self, decoder_config: dict, model=None):
        super(SentencePieceFeaturizer, self).__init__()
        self.vocabulary = decoder_config['vocabulary']
        self.model = self.__load_model() if model is None else model
        self.blank = 0  # treats blank as 0 (pad)
        # vocab size
        self.num_classes = self.model.get_piece_size()
        self.__init_vocabulary()

    def __load_model(self):
        filename_prefix = os.path.splitext(self.vocabulary)[0]
        processor = sp.SentencePieceProcessor()
        processor.load(filename_prefix + ".model")
        return processor

    def __init_vocabulary(self):
        self.tokens = []
        for idx in range(1, self.num_classes):
            self.tokens.append(self.model.decode_ids([idx]))
        self.non_blank_tokens = self.tokens.copy()
        self.tokens.insert(0, "")
        self.upoints = tf.strings.unicode_decode(self.tokens, "UTF-8")
        self.upoints = self.upoints.to_tensor()  # [num_classes, max_subword_length]

    @classmethod
    def load_from_file(cls, decoder_config: dict, filename: str = None):
        if filename is not None:
            filename_prefix = os.path.splitext(file_util.preprocess_paths(filename))[0]
        else:
            filename_prefix = decoder_config.get("output_path_prefix", None)
        processor = sp.SentencePieceProcessor()
        processor.load(filename_prefix + ".model")
        return cls(decoder_config, processor)

    def extract(self, text: str) -> tf.Tensor:
        """
        Convert string to a list of integers
        # encode: text => id
        sp.encode_as_pieces('This is a test') --> ['▁This', '▁is', '▁a', '▁t', 'est']
        sp.encode_as_ids('This is a test') --> [209, 31, 9, 375, 586]
        Args:
            text: string (sequence of characters)

        Returns:
            sequence of ints in tf.Tensor
        """
        text = self.preprocess_text(text)
        text = text.strip()  # remove trailing space
        indices = self.model.encode_as_ids(text)
        return tf.convert_to_tensor(indices, dtype=tf.int32)

    def iextract(self, indices: tf.Tensor) -> tf.Tensor:
        """
        Convert list of indices to string
        # decode: id => text
        sp.decode_pieces(['▁This', '▁is', '▁a', '▁t', 'est']) --> This is a test
        sp.decode_ids([209, 31, 9, 375, 586]) --> This is a test

        Args:
            indices: tf.Tensor with dim [B, None]

        Returns:
            transcripts: tf.Tensor of dtype tf.string with dim [B]
        """
        indices = self.normalize_indices(indices)
        with tf.device("/CPU:0"):  # string data is not supported on GPU
            def decode(x):
                if x[0] == self.blank: x = x[1:]
                return self.model.decode_ids(x.tolist())

            text = tf.map_fn(
                lambda x: tf.numpy_function(decode, inp=[x], Tout=tf.string),
                indices,
                fn_output_signature=tf.TensorSpec([], dtype=tf.string)
            )
        return text

    @tf.function(
        input_signature=[
            tf.TensorSpec([None], dtype=tf.int32)
        ]
    )
    def indices2upoints(self, indices: tf.Tensor) -> tf.Tensor:
        """
        Transform Predicted Indices to Unicode Code Points (for using tflite)
        Args:
            indices: tf.Tensor of Classes in shape [None]

        Returns:
            unicode code points transcript with dtype tf.int32 and shape [None]
        """
        with tf.name_scope("indices2upoints"):
            indices = self.normalize_indices(indices)
            upoints = tf.gather_nd(self.upoints, tf.expand_dims(indices, axis=-1))
            return tf.gather_nd(upoints, tf.where(tf.not_equal(upoints, 0)))
