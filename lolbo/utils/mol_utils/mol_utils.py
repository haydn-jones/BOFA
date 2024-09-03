import math
import os
import sys
from typing import Optional

import networkx as nx
import torch
from guacamol import standard_benchmarks
from rdkit import Chem, RDConfig
from rdkit.Chem import Descriptors, rdmolops
from torch import Tensor

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # type: ignore

guacamol_objs = {
    "med1": standard_benchmarks.median_camphor_menthol(),  #'Median molecules 1'
    "med2": standard_benchmarks.median_tadalafil_sildenafil(),  #'Median molecules 2',
    "pdop": standard_benchmarks.perindopril_rings(),  # 'Perindopril MPO',
    "osmb": standard_benchmarks.hard_osimertinib(),  # 'Osimertinib MPO',
    "adip": standard_benchmarks.amlodipine_rings(),  # 'Amlodipine MPO'
    "siga": standard_benchmarks.sitagliptin_replacement(),  #'Sitagliptin MPO'
    "zale": standard_benchmarks.zaleplon_with_other_formula(),  # 'Zaleplon MPO'
    "valt": standard_benchmarks.valsartan_smarts(),  #'Valsartan SMARTS',
    "dhop": standard_benchmarks.decoration_hop(),  # 'Deco Hop'
    "shop": standard_benchmarks.scaffold_hop(),  # Scaffold Hop'
    "rano": standard_benchmarks.ranolazine_mpo(),  #'Ranolazine MPO'
    "fexo": standard_benchmarks.hard_fexofenadine(),  # 'Fexofenadine MPO'... 'make fexofenadine less greasy'
}


GUACAMOL_TASK_NAMES = [
    "med1",
    "pdop",
    "adip",
    "rano",
    "osmb",
    "siga",
    "zale",
    "valt",
    "med2",
    "dhop",
    "shop",
    "fexo",
]


def smile_to_guacamole_score(obj_func_key: str, smile: str) -> Optional[float]:
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    func = guacamol_objs[obj_func_key]
    score = func.objective.score(smile)
    if score < 0:
        return None
    return score


def smile_to_penalized_logP(smile: str) -> Optional[float]:
    """calculate penalized logP for a given smiles string"""
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    logp = Descriptors.MolLogP(mol)  # type: ignore
    sa = sascorer.calculateScore(mol)
    cycle_length = _cycle_score(mol)
    """
    Calculate final adjusted score.
    These magic numbers are the empirical means and
    std devs of the dataset.

    I agree this is a weird way to calculate a score...
    but this is what previous papers did!
    """
    score = (
        (logp - 2.45777691) / 1.43341767 + (-sa + 3.05352042) / 0.83460587 + (-cycle_length - -0.04861121) / 0.28746695
    )
    return max(score, -float("inf"))


def _cycle_score(mol: Chem.Mol) -> int:
    cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    return cycle_length


def smiles_to_desired_scores(smiles_list: list[str], task_id: str = "logp") -> Tensor:
    scores = []
    for smiles_str in smiles_list:
        if task_id == "logp":
            score_ = smile_to_penalized_logP(smiles_str)
        else:
            score_ = smile_to_guacamole_score(task_id, smiles_str)

        if (score_ is not None) and (math.isfinite(score_)):
            scores.append(score_)
        else:
            scores.append(float("nan"))

    return torch.tensor(scores)
