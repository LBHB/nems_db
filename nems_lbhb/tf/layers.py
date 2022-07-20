"""
modules/state.py

functions for applying state-related transformations
"""

import re
import logging
import numpy as np
import tensorflow as tf

import nems_lbhb.preprocessing as preproc
from nems0.modules import NemsModule
from nems0.registry import xmodule
from nems0.tf.layers import BaseLayer

log = logging.getLogger(__name__)


class sdexp_layer(BaseLayer):
    """sdexp stategain.
    unit= number of state channels
    n_inputs = number of pred (input) channels
    """

    _STATE_LAYER = True

    def __init__(self,
                 units=None,
                 n_inputs=1,
                 initializer=None,
                 seed=0,
                 state_type='both',
                 bounds=None,
                 *args,
                 **kwargs,
                 ):
        super(sdexp_layer, self).__init__(*args, **kwargs)

        self.state_type = state_type

        # try to infer the number of units if not specified
        if units is None and initializer is None:
            self.units = 1
        elif units is None:
            self.units = initializer['amplitude_g'].value.shape[1]
        else:
            self.units = units

        self.n_inputs = n_inputs

        self.initializer = {
                'amplitude_g': tf.random_normal_initializer(seed=seed),
                'amplitude_d': tf.random_normal_initializer(seed=seed + 1),
                'base_g': tf.random_normal_initializer(seed=seed + 2),
                'base_d': tf.random_normal_initializer(seed=seed + 3),
                'kappa_g': tf.random_normal_initializer(seed=seed + 4),
                'kappa_d': tf.random_normal_initializer(seed=seed + 5),
                'offset_g': tf.random_normal_initializer(seed=seed + 6),
                'offset_d': tf.random_normal_initializer(seed=seed + 7),
        }
        if initializer is not None:
            self.initializer.update(initializer)

    def build(self, input_shape):
        input_shape, state_shape = input_shape

        if self.state_type != 'dc_only':
            self.amplitude_g = self.add_weight(name='amplitude_g',
                                     shape=(self.n_inputs, self.units),
                                     dtype='float32',
                                     initializer=self.initializer['amplitude_g'],
                                     trainable=True,
                                     )
            self.base_g = self.add_weight(name='base_g',
                                          shape=(self.n_inputs, self.units),
                                          dtype='float32',
                                          initializer=self.initializer['base_g'],
                                          trainable=True,
                                          )
            self.kappa_g = self.add_weight(name='kappa_g',
                                      shape=(self.n_inputs, self.units),
                                      dtype='float32',
                                      initializer=self.initializer['kappa_g'],
                                      trainable=True,
                                      )
            self.offset_g = self.add_weight(name='offset_g',
                                      shape=(self.n_inputs, self.units),
                                      dtype='float32',
                                      initializer=self.initializer['offset_g'],
                                      trainable=True,
                                      )

        if self.state_type != 'gain_only':
            # don't need a d param if we only want gain
            self.amplitude_d = self.add_weight(name='amplitude_d',
                                     shape=(self.n_inputs, self.units),
                                     dtype='float32',
                                     initializer=self.initializer['amplitude_d'],
                                     trainable=True,
                                     )
            self.base_d = self.add_weight(name='base_d',
                                               shape=(self.n_inputs, self.units),
                                               dtype='float32',
                                               initializer=self.initializer['base_d'],
                                               trainable=True,
                                               )
            self.kappa_d = self.add_weight(name='kappa_d',
                                      shape=(self.n_inputs, self.units),
                                      dtype='float32',
                                      initializer=self.initializer['kappa_d'],
                                      trainable=True,
                                      )
            self.offset_d = self.add_weight(name='offset_d',
                                      shape=(self.n_inputs, self.units),
                                      dtype='float32',
                                      initializer=self.initializer['offset_d'],
                                      trainable=True,
                                      )

    def call(self, inputs, training=True):
        """
        TODO: support gain- or baseline only
        """
        inputs, s = inputs

        log.debug('s: ', s.shape)
        log.debug('inputs: ', inputs.shape)
        log.debug('amplitude_g: ', self.amplitude_g.shape)

        #if self.state_type != 'gain_only':
        _ag = tf.reshape(tf.transpose(self.amplitude_g), [1, 1, self.units, self.n_inputs])
        _bg = tf.reshape(tf.transpose(self.base_g), [1, 1, self.units, self.n_inputs])
        _kg = tf.reshape(tf.transpose(self.kappa_g), [1, 1, self.units, self.n_inputs])
        _og = tf.reshape(tf.transpose(self.offset_g), [1, 1, self.units, self.n_inputs])
        _sg = _bg + _ag * tf.exp(-tf.exp(-tf.exp(_kg) * (tf.expand_dims(s,3) - _og)))
        log.debug('_ag: ', _ag.shape)
        log.debug('_sg: ', _sg.shape)

        _ad = tf.reshape(tf.transpose(self.amplitude_d), [1, 1, self.units, self.n_inputs])
        _bd = tf.reshape(tf.transpose(self.base_d), [1, 1, self.units, self.n_inputs])
        _kd = tf.reshape(tf.transpose(self.kappa_d), [1, 1, self.units, self.n_inputs])
        _od = tf.reshape(tf.transpose(self.offset_d), [1, 1, self.units, self.n_inputs])
        _sd = _bd + _ad * tf.exp(-tf.exp(-tf.exp(_kd) * (tf.expand_dims(s,3) - _od)))

        sg = tf.reduce_sum(_sg, axis=2)
        sd = tf.reduce_sum(_sd, axis=2)
        log.debug('sg: ', sg.shape)
        return sg * inputs + sd


    def weights_to_phi(self):
        layer_values = self.layer_values
        log.info(f'Converted {self.name} to modelspec phis.')
        return layer_values
