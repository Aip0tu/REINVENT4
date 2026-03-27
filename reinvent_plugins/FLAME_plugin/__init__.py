from .train import flam_predict
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

__version__ = "0.4.0"

__all__ = [
    'flam_predict',
]