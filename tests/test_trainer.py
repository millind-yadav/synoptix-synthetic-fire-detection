#!/usr/bin/env python3
"""
Unit tests for the Synoptix fire detection trainer.
"""

import pytest
import tensorflow as tf
from src.trainer import GCPTPUFireDetectionTrainer
from src.metrics import MacroF1

def test_macro_f1():
    """Test MacroF1 metric computation."""
    metric = MacroF1()
    y_true = tf.constant([[1, 0], [0, 1]], dtype=tf.float32)
    y_pred = tf.constant([[0.7, 0.3], [0.4, 0.6]], dtype=tf.float32)
    metric.update_state(y_true, y_pred)
    result = metric.result()
    assert result > 0, "MacroF1 should return a positive value"

@pytest.fixture
def trainer():
    """Fixture for trainer instance."""
    return GCPTPUFireDetectionTrainer(experiments_path='tests/data')

def test_trainer_init(trainer):
    """Test trainer initialization."""
    assert trainer.batch_size > 0, "Batch size should be positive"
    assert isinstance(trainer.results_path, Path), "Results path should be a Path object"
