import wandb
import tensorflow as tf
import numpy as np
from numpy import linalg as la


from . import env_util

logger = env_util.setup_environment()

class StepLossMetric(tf.keras.metrics.Metric):
    def __init__(self, name='step_loss', **kwargs):
        super(StepLossMetric, self).__init__(name=name, **kwargs)
        self.loss = tf.zeros(())

    def update_state(self, loss):
        self.loss = loss

    def result(self):
        return self.loss

    def reset_states(self):
        self.loss = tf.zeros(())


class LoggingCallback(tf.keras.callbacks.Callback):
    def __init__(
        self, 
        optimizer, 
        model, 
    ):
        super(LoggingCallback, self).__init__()
        self.optimizer = optimizer
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        logger.info("saving checkpoint")
        iterations = self.optimizer.iterations
        lr = self.optimizer.learning_rate(iterations)
        logger.info(f"[LR Logger] Epoch: {epoch}, lr: {lr}")
        wandb.log({"epoch": epoch, "lr": lr, "iterations": iterations.numpy()})
