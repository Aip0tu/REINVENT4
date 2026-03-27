"""Reinvent running modes.

Keep run mode imports lazy so missing optional dependencies in unrelated modes
do not block startup for the mode the user actually wants to run.
"""

from .create_adapter import *
from .handler import Handler


def run_sampling(*args, **kwargs):
    from reinvent.runmodes.samplers.run_sampling import run_sampling as _run_sampling

    return _run_sampling(*args, **kwargs)


def run_staged_learning(*args, **kwargs):
    from reinvent.runmodes.RL.run_staged_learning import run_staged_learning as _run_staged_learning

    return _run_staged_learning(*args, **kwargs)


def run_transfer_learning(*args, **kwargs):
    from reinvent.runmodes.TL.run_transfer_learning import run_transfer_learning as _run_transfer_learning

    return _run_transfer_learning(*args, **kwargs)


def run_scoring(*args, **kwargs):
    from reinvent.runmodes.scoring.run_scoring import run_scoring as _run_scoring

    return _run_scoring(*args, **kwargs)
