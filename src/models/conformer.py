import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.framework import ops
from tensorflow.python.eager import def_function

from .ctc import CtcModel
from .conformer_encoder import ConformerEncoder
from ..augmentations.augmentation import SpecAugmentation
from ..utils import math_util
from ..utils.training_utils import (
    _minimum_control_deps,
    reduce_per_replica,
    write_scalar_summaries,
)

class ConformerCtc(CtcModel):
    def __init__(
        self,
        vocabulary_size: int,
        encoder_subsampling: dict,
        encoder_dmodel: int = 144,
        encoder_num_blocks: int = 16,
        encoder_head_size: int = 36,
        encoder_num_heads: int = 4,
        encoder_mha_type: str = "relmha",
        encoder_kernel_size: int = 32,
        encoder_fc_factor: float = 0.5,
        encoder_dropout: float = 0,
        encoder_time_reduce_idx : list = None,
        encoder_time_recover_idx : list = None,
        encoder_conv_use_glu: bool = False,
        encoder_ds_subsample: bool = False,
        encoder_no_post_ln: bool = False,
        encoder_adaptive_scale: bool = False,
        encoder_fixed_arch: list = None,
        augmentation_config=None,
        name: str = "conformer",
        **kwargs,
    ) -> object:
        assert encoder_dmodel == encoder_num_heads * encoder_head_size
        if not isinstance(encoder_fixed_arch[0], list):
            encoder_fixed_arch = [encoder_fixed_arch] * encoder_num_blocks
        super().__init__(
            encoder=ConformerEncoder(
                subsampling=encoder_subsampling,
                dmodel=encoder_dmodel,
                num_blocks=encoder_num_blocks,
                head_size=encoder_head_size,
                num_heads=encoder_num_heads,
                mha_type=encoder_mha_type,
                kernel_size=encoder_kernel_size,
                fc_factor=encoder_fc_factor,
                dropout=encoder_dropout,
                time_reduce_idx=encoder_time_reduce_idx,
                time_recover_idx=encoder_time_recover_idx,
                conv_use_glu=encoder_conv_use_glu,
                ds_subsample=encoder_ds_subsample,
                no_post_ln=encoder_no_post_ln,
                adaptive_scale=encoder_adaptive_scale,
                fixed_arch=encoder_fixed_arch,
                name=f"{name}_encoder",
            ),
            decoder=tf.keras.layers.Conv1D(
                filters=vocabulary_size, kernel_size=1,
                strides=1, padding="same",
                name=f"{name}_logits"
            ),
            augmentation = SpecAugmentation(
                num_freq_masks=augmentation_config['freq_masking']['num_masks'],
                freq_mask_len=augmentation_config['freq_masking']['mask_factor'],
                num_time_masks=augmentation_config['time_masking']['num_masks'],
                time_mask_prop=augmentation_config['time_masking']['p_upperbound'],
                name=f"{name}_specaug"
            ) if augmentation_config is not None else None,
            vocabulary_size=vocabulary_size,
            name=name,
            **kwargs
        )
        self.time_reduction_factor = self.encoder.conv_subsampling.time_reduction_factor
        self.dmodel = encoder_dmodel

    # The following functions override the original function
    # in order to gather the outputs from multiple TPU cores

    def make_train_function(self):
        if self.train_function is not None:
            return self.train_function

        def step_function(model, iterator):
            """Runs a single training step."""

            def run_step(data):
                outputs = model.train_step(data)
                # Ensure counter is updated only if `train_step` succeeds.
                with ops.control_dependencies(_minimum_control_deps(outputs)):
                    model._train_counter.assign_add(1)  # pylint: disable=protected-access
                return outputs

            data = next(iterator)
            outputs = model.distribute_strategy.run(run_step, args=(data,))
            outputs = reduce_per_replica(outputs, self.distribute_strategy)
            write_scalar_summaries(outputs, step=model._train_counter)  # pylint: disable=protected-access
            return outputs

        if self._steps_per_execution.numpy().item() == 1:

            def train_function(iterator):
                """Runs a training execution with one step."""
                return step_function(self, iterator)
        else:

            def train_function(iterator):
                """Runs a training execution with multiple steps."""
                for _ in math_ops.range(self._steps_per_execution):
                    outputs = step_function(self, iterator)
                return outputs

        if not self.run_eagerly:
            train_function = def_function.function(
                train_function, experimental_relax_shapes=True)

        self.train_function = train_function

        if self._cluster_coordinator:
            self.train_function = lambda iterator: self._cluster_coordinator.schedule(  # pylint: disable=g-long-lambda
                train_function, args=(iterator,))

        return self.train_function

    def make_test_function(self):
        if self.test_function is not None:
            return self.test_function

        def step_function(model, iterator):
            """Runs a single evaluation step."""

            def run_step(data):
                outputs = model.test_step(data)
                # Ensure counter is updated only if `test_step` succeeds.
                with ops.control_dependencies(_minimum_control_deps(outputs)):
                    model._test_counter.assign_add(1)  # pylint: disable=protected-access
                return outputs

            data = next(iterator)
            outputs = model.distribute_strategy.run(run_step, args=(data,))
            outputs = reduce_per_replica(outputs, self.distribute_strategy)
            return outputs

        if self._steps_per_execution.numpy().item() == 1:

            def test_function(iterator):
                """Runs an evaluation execution with one step."""
                return step_function(self, iterator)
        else:

            def test_function(iterator):
                """Runs an evaluation execution with multiple steps."""
                for _ in math_ops.range(self._steps_per_execution):
                    outputs = step_function(self, iterator)
                return outputs

        if not self.run_eagerly:
            test_function = def_function.function(test_function, experimental_relax_shapes=True)

        self.test_function = test_function

        if self._cluster_coordinator:
            self.test_function = lambda iterator: self._cluster_coordinator.schedule(  # pylint: disable=g-long-lambda
                test_function, args=(iterator,))

        return self.test_function


class ConformerCtcAccumulate(ConformerCtc):
    def __init__(self, n_gradients: int = 1, **kwargs) -> object:
        super().__init__(**kwargs)
        self.time_reduction_factor = self.encoder.conv_subsampling.time_reduction_factor

        self.n_gradients = tf.constant(n_gradients, dtype=tf.int32, name="conformer/num_accumulated_gradients")
        self.n_acum_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="conformer/accumulate_step")

    def make(self, input_shape, batch_size=None):
        super().make(input_shape, batch_size)
        self.gradient_accumulation = [
                tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False, name=f"{v.name}/cached_accumulated_gradient") for v in self.trainable_variables
        ]

    def train_step(self, batch):
        """
        Args:
            batch ([tf.Tensor]): a batch of training data

        Returns:
            Dict[tf.Tensor]: a dict of validation metrics with keys are the name of metric

        """
        self.n_acum_step.assign_add(1)

        inputs, y_true = batch
        loss, y_pred, gradients = self.gradient_step(inputs, y_true)

        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(gradients[i] / tf.cast(self.n_gradients, tf.float32))

        tf.cond(tf.equal(self.n_acum_step, self.n_gradients), self.apply_accu_gradients, lambda: None)

        self._metrics["loss"].update_state(loss)
        if 'WER' in self._metrics:
            self._metrics['WER'].update_state(y_true, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def apply_accu_gradients(self):
        # Apply accumulated gradients
        self.optimizer.apply_gradients(zip(self.gradient_accumulation, 
                                           self.trainable_variables))

        # Reset
        self.n_acum_step.assign(0)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(
                tf.zeros_like(self.trainable_variables[i],  dtype=tf.float32)
            )
