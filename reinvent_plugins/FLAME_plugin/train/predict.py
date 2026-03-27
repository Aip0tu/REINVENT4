from typing import List

import torch
from tqdm import tqdm

from ..dataprocess.data import MoleculeDataLoader, MoleculeDataset, StandardScaler
from ..models import MoleculeModel


def predict(model: MoleculeModel,
            data_loader: MoleculeDataLoader,
            disable_progress_bar: bool = False,
            scaler: StandardScaler = None) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A :class:`~flam.models.model.MoleculeModel`.
    :param data_loader: A :class:`~flam.data.data.MoleculeDataLoader`.
    :param disable_progress_bar: Whether to disable the progress bar.
    :param scaler: A :class:`~flam.features.scaler.StandardScaler` object fit on the training targets.
    :return: A list of lists of predictions. The outer list is molecules while the inner list is tasks.
    """
    model.eval()

    preds = []

    for batch in data_loader:
        # Prepare batch
        batch: MoleculeDataset
        mol_batch, features_batch, mol_adj_batch, mol_dist_batch, mol_clb_batch = \
            batch.batch_graph(), batch.features(), batch.adj_features(
            ), batch.dist_features(), batch.clb_features()

        # Move data to the same device as the model
        device = next(model.parameters()).device
        
        # Handle case where mol_batch is a list of BatchMolGraph objects
        if isinstance(mol_batch, list) and len(mol_batch) > 0:
            mol_batch = mol_batch[0]
        
        # Move BatchMolGraph attributes to device
        mol_batch.f_atoms = mol_batch.f_atoms.to(device)
        mol_batch.f_bonds = mol_batch.f_bonds.to(device)
        mol_batch.a2b = mol_batch.a2b.to(device)
        mol_batch.b2a = mol_batch.b2a.to(device)
        mol_batch.b2revb = mol_batch.b2revb.to(device)
        
        # Handle other features if they exist
        if features_batch is not None and isinstance(features_batch, torch.Tensor):
            features_batch = features_batch.to(device)
        if mol_adj_batch is not None and isinstance(mol_adj_batch, torch.Tensor):
            mol_adj_batch = mol_adj_batch.to(device)
        if mol_dist_batch is not None and isinstance(mol_dist_batch, torch.Tensor):
            mol_dist_batch = mol_dist_batch.to(device)
        if mol_clb_batch is not None and isinstance(mol_clb_batch, torch.Tensor):
            mol_clb_batch = mol_clb_batch.to(device)

        # Make predictions
        with torch.no_grad():
            batch_preds = model(mol_batch, features_batch,
                                mol_adj_batch, mol_dist_batch, mol_clb_batch)

        batch_preds = batch_preds.data.cpu().numpy()

        # Inverse scale if regression
        if scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds)

        # Collect vectors
        batch_preds = batch_preds.tolist()
        preds.extend(batch_preds)

    return preds
