#!/usr/bin/env python3
"""
Custom metrics for the Synoptix fire detection project.
Includes MacroF1 for macro-averaged F1 score.
"""

import tensorflow as tf
from tensorflow.keras.metrics import Metric
import tensorflow.keras.backend as K

class MacroF1(Metric):
    """Macro-averaged F1 score metric for multi-class classification."""
    def __init__(self, name='macro_f1', **kwargs):
        super(MacroF1, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros', shape=(2,))
        self.false_positives = self.add_weight(name='fp', initializer='zeros', shape=(2,))
        self.false_negatives = self.add_weight(name='fn', initializer='zeros', shape=(2,))

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.argmax(y_true, axis=-1)
        
        for i in range(2):
            tp = tf.reduce_sum(tf.cast(tf.logical_and(y_true == i, y_pred == i), tf.float32))
            fp = tf.reduce_sum(tf.cast(tf.logical_and(y_true != i, y_pred == i), tf.float32))
            fn = tf.reduce_sum(tf.cast(tf.logical_and(y_true == i, y_pred != i), tf.float32))
            
            self.true_positives.assign_add(tf.gather([tp, tp], [i]))
            self.false_positives.assign_add(tf.gather([fp, fp], [i]))
            self.false_negatives.assign_add(tf.gather([fn, fn], [i]))

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + K.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + K.epsilon())
        f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
        return tf.reduce_mean(f1)

    def reset_states(self):
        self.true_positives.assign(tf.zeros_like(self.true_positives))
        self.false_positives.assign(tf.zeros_like(self.false_positives))
        self.false_negatives.assign(tf.zeros_like(self.false_negatives))