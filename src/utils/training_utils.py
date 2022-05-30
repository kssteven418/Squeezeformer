import tensorflow as tf

from tensorflow.python.keras import backend
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import ops
from tensorflow.python.eager import context
from tensorflow.python.util import nest
from tensorflow.python.ops import variables
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import values as ds_values

def _minimum_control_deps(outputs):
        """Returns the minimum control dependencies to ensure step succeeded."""
        if context.executing_eagerly():
                return []    # Control dependencies not needed.
        outputs = nest.flatten(outputs, expand_composites=True)
        for out in outputs:
                # Variables can't be control dependencies.
                if not isinstance(out, variables.Variable):
                        return [out]    # Return first Tensor or Op from outputs.
        return []    # No viable Tensor or Op to use for control deps.


def reduce_per_replica(values, strategy):
    """Reduce PerReplica objects.

    Args:
        values: Structure of `PerReplica` objects or `Tensor`s. `Tensor`s are
            returned as-is.
        strategy: `tf.distribute.Strategy` object.
        reduction: One of 'first', 'concat'.

    Returns:
        Structure of `Tensor`s.
    """

    def _reduce(v):
        """Reduce a single `PerReplica` object."""
        if _collective_all_reduce_multi_worker(strategy):
            return _multi_worker_concat(v, strategy)
        if not isinstance(v, ds_values.PerReplica):
            return v
        if _is_tpu_multi_host(strategy):
            return _tpu_multi_host_concat(v, strategy)
        else:
            return concat(strategy.unwrap(v))

    return nest.map_structure(_reduce, values)


def concat(tensors, axis=0):
    if len(tensors[0].shape) == 0:
        return tf.math.add_n(tensors)
    """Concats `tensor`s along `axis`."""
    if isinstance(tensors[0], sparse_tensor.SparseTensor):
        return sparse_ops.sparse_concat_v2(axis=axis, sp_inputs=tensors)
    return array_ops.concat(tensors, axis=axis)

def _collective_all_reduce_multi_worker(strategy):
    return (isinstance(strategy,
            collective_all_reduce_strategy.CollectiveAllReduceStrategy)
            ) and strategy.extended._in_multi_worker_mode()    # pylint: disable=protected-access

def _is_scalar(x):
    return isinstance(x, (ops.Tensor, variables.Variable)) and x.shape.rank == 0

def write_scalar_summaries(logs, step):
    for name, value in logs.items():
        if _is_scalar(value):
            summary_ops_v2.scalar('batch_' + name, value, step=step)

def _is_tpu_multi_host(strategy):
    return (backend.is_tpu_strategy(strategy) and
                    strategy.extended.num_hosts > 1)

def _tpu_multi_host_concat(v, strategy):
    """Correctly order TPU PerReplica objects."""
    replicas = strategy.unwrap(v)
    # When distributed datasets are created from Tensors / NumPy,
    # TPUStrategy.experimental_distribute_dataset shards data in
    # (Replica, Host) order, and TPUStrategy.unwrap returns it in
    # (Host, Replica) order.
    # TODO(b/150317897): Figure out long-term plan here.
    num_replicas_per_host = strategy.extended.num_replicas_per_host
    ordered_replicas = []
    for replica_id in range(num_replicas_per_host):
        ordered_replicas += replicas[replica_id::num_replicas_per_host]
    return concat(ordered_replicas)


