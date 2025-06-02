"""
Jaundice Detection Model Package
Contains individual and combined models:
- SimpleCNNJaundiceDetector: CNN for image-only detection (128x128)
- EfficientNetJaundiceDetector: Transfer learning with EfficientNetB0 (224x224)
- TabularJaundiceDetector: MLP for tabular data detection
- MultiInputCNNTabularDetector: CNN (images) + MLP (tabular)
- MultiInputEfficientNetTabularDetector: EfficientNetB0 (images) + MLP (tabular)
- EnsembleJaundiceDetector: Soft voting ensemble of an image model and a tabular model
"""

from .cnn_models import SimpleCNNJaundiceDetector
from .efficientnet_models import EfficientNetJaundiceDetector
from .tabular_model import TabularJaundiceDetector
from .multi_input_cnn_tabular import MultiInputCNNTabularDetector
from .multi_input_efficientnet_tabular import MultiInputEfficientNetTabularDetector
from .ensemble_model import EnsembleJaundiceDetector

__all__ = [
    'SimpleCNNJaundiceDetector',
    'EfficientNetJaundiceDetector', 
    'TabularJaundiceDetector',
    'MultiInputCNNTabularDetector',
    'MultiInputEfficientNetTabularDetector',
    'EnsembleJaundiceDetector'
] 