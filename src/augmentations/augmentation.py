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

import tensorflow as tf

from ..utils import shape_util


class SpecAugmentation(tf.keras.Model):

    def __init__(
        self, 
        num_freq_masks=2,
        freq_mask_len=27,
        num_time_masks=5,
        time_mask_prop=0.05,
        name='specaug',
        **kwargs,
    ):
        super(SpecAugmentation, self).__init__(name=name, **kwargs)

        self.num_freq_masks = num_freq_masks
        self.freq_mask_len = freq_mask_len
        self.num_time_masks = num_time_masks
        self.time_mask_prop = time_mask_prop


    def time_mask(self, inputs, inputs_len):
        time_max = inputs_len
        B, T, F = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]
        t = tf.random.uniform(shape=tf.shape(time_max), minval=0, maxval=self.time_mask_prop)
        t = tf.cast(tf.cast(time_max, tf.dtypes.float32) * t, 'int32')
        t0 = tf.random.uniform(shape=tf.shape(time_max), minval=0, maxval=1)
        t0 = tf.cast(tf.cast(time_max - t, tf.dtypes.float32) * t0, 'int32')
        t = tf.repeat(tf.reshape(t, (-1, 1)), T, axis=1)
        t0 = tf.repeat(tf.reshape(t0, (-1, 1)), T, axis=1)

        indices = tf.repeat(tf.reshape(tf.range(T), (1, -1)), B, axis=0)

        left_mask = tf.cast(tf.math.greater_equal(indices, t0), 'float32')
        right_mask = tf.cast(tf.math.less(indices, t0 + t), 'float32')
        mask = 1.0 - left_mask * right_mask
        masked_inputs = inputs * tf.reshape(mask, (B, T, 1, 1))
        return masked_inputs


    def frequency_mask(self, inputs, inputs_len):
        B, T, F = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]
        f = tf.random.uniform(shape=tf.shape(inputs_len), minval=0, maxval=self.freq_mask_len, dtype='int32')
        f0 = tf.random.uniform(shape=tf.shape(inputs_len), minval=0, maxval=1)
        f0 = tf.cast(tf.cast(F - f, tf.dtypes.float32) * f0, 'int32')

        f = tf.repeat(tf.reshape(f, (-1, 1)), F, axis=1)
        f0 = tf.repeat(tf.reshape(f0, (-1, 1)), F, axis=1)

        indices = tf.repeat(tf.reshape(tf.range(F), (1, -1)), B, axis=0)
        left_mask = tf.cast(tf.math.greater_equal(indices, f0), 'float32')
        right_mask = tf.cast(tf.math.less(indices, f0 + f), 'float32')
        mask = 1.0 - left_mask * right_mask
        masked_inputs = inputs * tf.reshape(mask, (B, 1, F, 1))
        return masked_inputs


    @tf.function
    def call(self, inputs, inputs_len):
        masked_inputs = inputs
        for _ in range(self.num_time_masks):
            masked_inputs = self.time_mask(masked_inputs, inputs_len)
        for _ in range(self.num_freq_masks):
            masked_inputs = self.frequency_mask(masked_inputs, inputs_len)
        return masked_inputs
