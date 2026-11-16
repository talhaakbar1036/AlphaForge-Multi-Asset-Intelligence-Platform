"""Typed errors for the optional direction-model subsystem."""
from __future__ import annotations


class ModelError(RuntimeError):
    """Base class for direction-model failures."""


class ModelArtifactNotFound(ModelError):
    """No trained artifact exists for the requested symbol."""


class ModelDependencyError(ModelError):
    """TensorFlow / scikit-learn / joblib is not installed."""


class ModelPredictionError(ModelError):
    """The model exists but the input data cannot produce a prediction."""
