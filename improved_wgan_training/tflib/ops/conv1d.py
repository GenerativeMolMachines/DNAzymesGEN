import numpy as np
import tensorflow as tf
from tensorflow.compat import v1 as tfv1
import tflib as lib
_default_weightnorm = False

def enable_default_weightnorm():
    global _default_weightnorm
    _default_weightnorm = True

def Conv1D(name, input_dim, output_dim, filter_size, inputs, he_init=True, mask_type=None, stride=1, weightnorm=None, biases=True, gain=1.):
    """
    inputs: tensor of shape (batch size, num channels, width)
    mask_type: one of None, 'a', 'b'

    returns: tensor of shape (batch size, num channels, width)
    """
    with tfv1.name_scope(name) as scope:

        if mask_type is not None:
            mask_type, mask_n_channels = mask_type
            mask = np.ones(
                (filter_size, input_dim, output_dim), 
                dtype='float32'
            )
            center = filter_size // 2

            # Mask out future locations
            mask[center+1:, :, :] = 0.

            # Mask out future channels
            for i in range(mask_n_channels):
                for j in range(mask_n_channels):
                    if (mask_type=='a' and i >= j) or (mask_type=='b' and i > j):
                        mask[
                            center,
                            i::mask_n_channels,
                            j::mask_n_channels
                        ] = 0.

        def uniform(stdev, size):
            return np.random.uniform(
                low=-stdev * np.sqrt(3),
                high=stdev * np.sqrt(3),
                size=size
            ).astype('float32')

        fan_in = input_dim * filter_size
        fan_out = output_dim * filter_size / stride

        if mask_type is not None:
            fan_in /= 2.
            fan_out /= 2.

        if he_init:
            filters_stdev = np.sqrt(4./(fan_in+fan_out))
        else:
            filters_stdev = np.sqrt(2./(fan_in+fan_out))

        filter_values = uniform(
            filters_stdev,
            (filter_size, input_dim, output_dim)
        )
        filter_values *= gain

        filters = lib.param(name+'.Filters', filter_values)

        if weightnorm is None:
            weightnorm = _default_weightnorm
        if weightnorm:
            norm_values = np.sqrt(np.sum(np.square(filter_values), axis=(0,1)))
            target_norms = lib.param(
                name + '.g',
                norm_values
            )
            with tfv1.name_scope('weightnorm') as scope:
                norms = tf.sqrt(tf.reduce_sum(tf.square(filters), axis=[0,1]))
                filters = filters * (target_norms / norms)

        if mask_type is not None:
            with tfv1.name_scope('filter_mask'):
                filters = filters * mask

        # Преобразуем входные данные в формат NWC для conv1d
        inputs_nwc = tf.transpose(inputs, [0, 2, 1])  # NCHW -> NWC
        result = tf.nn.conv1d(
            input=inputs_nwc,
            filters=filters,
            stride=stride,
            padding='SAME',
            data_format='NWC'
        )
        # Возвращаем к формату NCHW
        result = tf.transpose(result, [0, 2, 1])

        if biases:
            _biases = lib.param(
                name+'.Biases',
                np.zeros([output_dim], dtype='float32')
            )
            result = result + tf.reshape(_biases, [1, output_dim, 1])

        return result