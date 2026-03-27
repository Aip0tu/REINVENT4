# """Compute scores for FLAME"""
#
# from __future__ import annotations
# from reinvent_plugins.FLAME_plugin import flam_predict
#
# __all__ = [
#     "flameabs",
#     "flameemi",
#     "flamee",
#     "flameplqy",
#     'stokes',
# ]
#
# from dataclasses import dataclass
# from typing import List
#
# from rdkit import Chem
# import numpy as np
#
# from .component_results import ComponentResults
# from reinvent_plugins.mol_cache import molcache
# from .add_tag import add_tag
# import torch
# import numpy as np
# from rdkit import Chem
#
#
# @add_tag("__parameters")
# @dataclass
# class Parameters:
#     """Parameters for the scoring component
#
#     Note that all parameters are always lists because components can have
#     multiple endpoints and so all the parameters from each endpoint is
#     collected into a list.  This is also true in cases where there is only one
#     endpoint.
#     """
#
# #     smarts: List[str]
#     pass
#
#
# @add_tag("__component")
# class flameemi:
#     def __init__(self, params: Parameters):
#         pass
#
#     def __call__(self,
#                  smiles: List[str],
#                  solvent:str = 'O',
#                 ) -> np.array:
#         scores = flam_predict('emi', [[x, solvent] for x in smiles])
#         return ComponentResults([scores])
#
# @add_tag("__component")
# class flameabs:
#     def __init__(self, params: Parameters):
#         pass
#
#     def __call__(self,
#                  smiles: List[str],
#                  solvent:str = 'O',
#                 ) -> np.array:
#         scores = flam_predict('abs', [[x, solvent] for x in smiles])
#         return ComponentResults([scores])
#
#
# @add_tag("__component")
# class flamee:
#     def __init__(self, params: Parameters):
#         pass
#
#     def __call__(self,
#                  smiles: List[str],
#                  solvent:str = 'O',
#                 ) -> np.array:
#         scores = flam_predict('e', [[x, solvent] for x in smiles])
#         return ComponentResults([scores])
#
# @add_tag("__component")
# class flameplqy:
#     def __init__(self, params: Parameters):
#         pass
#
#     def __call__(self,
#                  smiles: List[str],
#                  solvent:str = 'O',
#                 ) -> np.array:
#         scores = flam_predict('plqy', [[x, solvent] for x in smiles])
#         return ComponentResults([scores])
#
# @add_tag("__component")
# class stokes:
#     def __init__(self, params: Parameters):
#         pass
#
#     def __call__(self,
#                  smiles: List[str],
#                  solvent:str = 'O',
#                 ) -> np.array:
#         abs_scores = flam_predict('abs', [[x, solvent] for x in smiles])
#         emi_scores = flam_predict('emi', [[x, solvent] for x in smiles])
#         scores = emi_scores - abs_scores
#         scores = scores / 10
#         return ComponentResults([scores])
#
# @add_tag("__component")
# class coumarin:
#     def __init__(self, params: Parameters):
#         pass
#
#     def __call__(self,
#                  smiles: List[str],
#                 ) -> np.array:
#         coumarin_smarts = [
#             '[#8]=[#6]1:[#8]:[#6&X3;H0](:[#6&X3;H0](:[#6]:[#6]:1))'#普通香豆素
# #             'O=C1OC2=CC=CC=C2C=C1'
# #             'O=C1OC2=CC=CC=C2C3=C1C=CO3',
# #             'O=C1OC2=CC=CC=C2C3=C1C=CN3',
# #             'O=C1OC2=CC=CC=C2C3=C1C=NN3',
# #             'O=C1OC2=CC=CC=C2C3=C1N=CO3',
# #             'O=C1OC2=CC=CC=C2C3=C1C=CC=N3',
# #             'O=C1OC2=CC=CC=C2C3=C1C=CN=N3',
# #             'O=C1OC2=CC=CC=C2C3=C1N=CC=N3'
#         ]
#         patterns = [Chem.MolFromSmarts(smarts) for smarts in coumarin_smarts]
#         scores = []
#         for smile in smiles:
#             mol = Chem.MolFromSmiles(smile)
#             if mol is None:
#                 scores.append(0)  # 如果分子无法转换为Mol对象，则得分为0
#             else:
#                 # 检查分子是否匹配任何一个香豆素母核的SMARTS模式
#                 matched = any(mol.HasSubstructMatch(pattern) for pattern in patterns)
#                 if matched:
#                     scores.append(10)  # 如果匹配到任何一个SMARTS模式，则得满分
#                 else:
#                     scores.append(0)  # 如果没有匹配到任何一个SMARTS模式，则得0分
#         return ComponentResults([scores])


"""Compute scores for FLAME"""

from __future__ import annotations
from reinvent_plugins.FLAME_plugin import flam_predict

__all__ = [
    "flameabs",
    "flameemi",
    "flamee",
    "flameplqy",
    'stokes',
]

from dataclasses import dataclass
from typing import List

import numpy as np

from .component_results import ComponentResults
from .add_tag import add_tag


@add_tag("__parameters")
@dataclass
class Parameters:
    """Parameters for the scoring component

    Note that all parameters are always lists because components can have
    multiple endpoints and so all the parameters from each endpoint is
    collected into a list.  This is also true in cases where there is only one
    endpoint.
    """
    solvent: List[str] | None = None


def _get_solvent(params: Parameters, default: str = "O") -> str:
    if params and params.solvent and params.solvent[0]:
        return params.solvent[0]

    return default


@add_tag("__component")
class flameemi:
    def __init__(self, params: Parameters):
        self.solvent = _get_solvent(params)

    def __call__(self,
                 smiles: List[str],
                ) -> np.array:
        scores = flam_predict('emi', [[x, self.solvent] for x in smiles])
        return ComponentResults([scores])

@add_tag("__component")
class flameabs:
    def __init__(self, params: Parameters):
        self.solvent = _get_solvent(params)

    def __call__(self,
                 smiles: List[str],
                ) -> np.array:
        scores = flam_predict('abs', [[x, self.solvent] for x in smiles])
        return ComponentResults([scores])


@add_tag("__component")
class flamee:
    def __init__(self, params: Parameters):
        self.solvent = _get_solvent(params)

    def __call__(self,
                 smiles: List[str],
                ) -> np.array:
        scores = flam_predict('e', [[x, self.solvent] for x in smiles])
        return ComponentResults([scores])

@add_tag("__component")
class flameplqy:
    def __init__(self, params: Parameters):
        self.solvent = _get_solvent(params)

    def __call__(self,
                 smiles: List[str],
                ) -> np.array:
        scores = flam_predict('plqy', [[x, self.solvent] for x in smiles])
        return ComponentResults([scores])

@add_tag("__component")
class stokes:
    def __init__(self, params: Parameters):
        self.solvent = _get_solvent(params)

    def __call__(self,
                 smiles: List[str],
                ) -> np.array:
        abs_scores = flam_predict('abs', [[x, self.solvent] for x in smiles])
        emi_scores = flam_predict('emi', [[x, self.solvent] for x in smiles])
        scores = emi_scores - abs_scores
        scores = scores / 10
        return ComponentResults([scores])

@add_tag("__component")
class coumarin:
    def __init__(self, params: Parameters):
        pass

    def __call__(self,
                 smiles: List[str],
                ) -> np.array:
        from rdkit import Chem
        coumarin_smarts = [
            '[#8]=[#6]1:[#8]:[#6&X3;H0](:[#6&X3;H0](:[#6]:[#6]:1))'#普通香豆素
        ]
        patterns = [Chem.MolFromSmarts(smarts) for smarts in coumarin_smarts]
        scores = []
        for smile in smiles:
            mol = Chem.MolFromSmiles(smile)
            if mol is None:
                scores.append(0)  # 如果分子无法转换为Mol对象，则得分为0
            else:
                # 检查分子是否匹配任何一个香豆素母核的SMARTS模式
                matched = any(mol.HasSubstructMatch(pattern) for pattern in patterns)
                if matched:
                    scores.append(10)  # 如果匹配到任何一个SMARTS模式，则得满分
                else:
                    scores.append(0)  # 如果没有匹配到任何一个SMARTS模式，则得0分
        return ComponentResults([scores])
