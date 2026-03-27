from collections import OrderedDict
import csv
import os
from typing import List, Optional, Union
import hashlib
import warnings
from pathlib import Path
import numpy as np
from tqdm import tqdm
import gdown

from .predict import predict
from ..args import PredictArgs, TrainArgs
from ..models.utils import load_args, load_checkpoint, load_scalers, makedirs, timeit, update_prediction_args
from ..dataprocess.data import get_data, get_data_from_smiles, MoleculeDataLoader, MoleculeDataset
from ..dataprocess.features import set_extra_atom_fdim, set_extra_bond_fdim

_WARNED_MODEL_CHECKSUMS = set()

@timeit()
def make_predictions(args: PredictArgs, smiles: List[List[str]] = None) -> List[List[Optional[float]]]:
    """
    Loads data and a trained model and uses the model to make predictions on the data.

    If SMILES are provided, then makes predictions on smiles.
    Otherwise makes predictions on :code:`args.test_data`.

    :param args: A :class:`~flam.args.PredictArgs` object containing arguments for
                 loading data and a model and making predictions.
    :param smiles: List of list of SMILES to make predictions on.
    :return: A list of lists of target predictions.
    """
    # print('Loading training args')
    train_args = load_args(args.checkpoint_paths[0])
    num_tasks, task_names = train_args.num_tasks, train_args.task_names

    update_prediction_args(predict_args=args, train_args=train_args)
    args: Union[PredictArgs, TrainArgs]

    if args.atom_descriptors == 'feature':
        set_extra_atom_fdim(train_args.atom_features_size)

    if args.bond_features_path is not None:
        set_extra_bond_fdim(train_args.bond_features_size)

    # print('Loading data')
    full_data = get_data_from_smiles(
        smiles=smiles,
        skip_invalid_smiles=False,
        features_generator=args.features_generator
    )

    # print('Validating SMILES')
    full_to_valid_indices = {}
    valid_index = 0
    for full_index in range(len(full_data)):
        if all(mol is not None for mol in full_data[full_index].mol):
            full_to_valid_indices[full_index] = valid_index
            valid_index += 1

    test_data = MoleculeDataset([full_data[i] for i in sorted(full_to_valid_indices.keys())])

    # Edge case if empty list of smiles is provided
    if len(test_data) == 0:
        return [None] * len(full_data)

    # print(f'Test size = {len(test_data):,}')

    # Predict with each model individually and sum predictions
    if args.dataset_type == 'multiclass':
        sum_preds = np.zeros((len(test_data), num_tasks, args.multiclass_num_classes))
    else:
        sum_preds = np.zeros((len(test_data), num_tasks))

    # Create data loader
    test_data_loader = MoleculeDataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Partial results for variance robust calculation.
    if args.ensemble_variance:
        all_preds = np.zeros((len(test_data), num_tasks, len(args.checkpoint_paths)))

    # print(f'Predicting with an ensemble of {len(args.checkpoint_paths)} models')
    for index, checkpoint_path in enumerate(args.checkpoint_paths):
        # Load model and scalers
        model = load_checkpoint(checkpoint_path, device=args.device)
        scaler, features_scaler, atom_descriptor_scaler, bond_feature_scaler = load_scalers(checkpoint_path)

        # Normalize features
        if args.features_scaling or train_args.atom_descriptor_scaling or train_args.bond_feature_scaling:
            test_data.reset_features_and_targets()
            if args.features_scaling:
                test_data.normalize_features(features_scaler)
            if train_args.atom_descriptor_scaling and args.atom_descriptors is not None:
                test_data.normalize_features(atom_descriptor_scaler, scale_atom_descriptors=True)
            if train_args.bond_feature_scaling and args.bond_features_size > 0:
                test_data.normalize_features(bond_feature_scaler, scale_bond_features=True)

        # Make predictions
        model_preds = predict(
            model=model,
            data_loader=test_data_loader,
            scaler=scaler
        )
        sum_preds += np.array(model_preds)
        if args.ensemble_variance:
            all_preds[:, :, index] = model_preds

    # Ensemble predictions
    avg_preds = sum_preds / len(args.checkpoint_paths)
    avg_preds = avg_preds.tolist()

    if args.ensemble_variance:
        all_epi_uncs = np.var(all_preds, axis=2)
        all_epi_uncs = all_epi_uncs.tolist()

    # Save predictions
    # print(f'Saving predictions to {args.preds_path}')
    assert len(test_data) == len(avg_preds)
    if args.ensemble_variance:
        assert len(test_data) == len(all_epi_uncs)
    # makedirs(args.preds_path, isfile=True)

    # Get prediction column names
    if args.dataset_type == 'multiclass':
        task_names = [f'{name}_class_{i}' for name in task_names for i in range(args.multiclass_num_classes)]
    else:
        task_names = task_names

    # Copy predictions over to full_data
    res = []
    for full_index, datapoint in enumerate(full_data):
        valid_index = full_to_valid_indices.get(full_index, None)
        preds = avg_preds[valid_index] if valid_index is not None else [0] * len(task_names)
        if args.ensemble_variance:
            epi_uncs = all_epi_uncs[valid_index] if valid_index is not None else [0] * len(task_names)

        # If extra columns have been dropped, add back in SMILES columns
        if args.drop_extra_columns:
            datapoint.row = OrderedDict()

            smiles_columns = args.smiles_columns

            for column, smiles in zip(smiles_columns, datapoint.smiles):
                datapoint.row[column] = smiles

        # Add predictions columns
        if args.ensemble_variance:
            for pred_name, pred, epi_unc in zip(task_names, preds, epi_uncs):
                datapoint.row[pred_name] = pred
                datapoint.row[pred_name+'_epi_unc'] = epi_unc
        else:
            for pred_name, pred in zip(task_names, preds):
                datapoint.row[pred_name] = pred
                res.append(pred)

    # Save
    return np.array(res)
    # return full_data
    # with open(args.preds_path, 'w') as f:
    #     writer = csv.DictWriter(f, fieldnames=full_data[0].row.keys())
    #     writer.writeheader()

    #     for datapoint in full_data:
    #         writer.writerow(datapoint.row)

    # return avg_preds

def get_md5_checksum(file_path):
    md5 = hashlib.md5()
    with open(file_path, 'rb') as file:
        for chunk in iter(lambda: file.read(4096), b''):
            md5.update(chunk)
    return md5.hexdigest()

def flam_predict(target, smiles=None) -> None:
    """Parses Chemprop predicting arguments and runs prediction using a trained Chemprop model.

    This is the entry point for the command line command :code:`flam_predict`.
    """
    checkpoint_dir = Path(__file__).resolve().parents[1] / "model_checkpoint"
    model_path = checkpoint_dir / f"{target}.pt"
    output_file = checkpoint_dir / "tmp_out.csv"
    md5_dict = {
        'abs':'5e41f194b73c34863a030a80bc43fa84',
        'emi':'23651be53f6489caf07ca39360ec0d54',
        'e': '0753b6a7114b2f2da2c2ac0ac32feb9f',
        'plqy': 'be297195bb631e2e9b46d8350c131fb6',
    }
    gd_dict = {
        'abs':'11eIKp4HYryvLc9TpDB3_d4P5WEQoN1mt',
        'emi':'1oHSAmxGbSQ58R9Tpnp6APapU7cGEGPWB',
        'e': '16__XbV35ZN6PPt2voKiD5CpjiUqkkQuI',
        'plqy': '1QFHqEShfucZ4v-lX17FBBTf6bRkWFeO6',
    }
    if target not in md5_dict.keys():
        raise ValueError('target should be one of [abs/emi/e/plqy]')

    if model_path.exists():
        if get_md5_checksum(model_path)!=md5_dict[target] and target not in _WARNED_MODEL_CHECKSUMS:
            warnings.warn(
                f"FLAME checkpoint '{model_path}' does not match the original reference checksum; "
                f"continuing with the local checkpoint for target '{target}'."
            )
            _WARNED_MODEL_CHECKSUMS.add(target)
    else:
        print(f'Downloading Model {target}.pt From Google Drive...')
        gdown.download('https://drive.google.com/uc?id='+gd_dict[target], str(model_path), quiet=False)
        if get_md5_checksum(model_path)!=md5_dict[target]:
            raise ValueError(f'model {target} download failed!')

    pred_args = ['--test_path', '', '--checkpoint_path', str(model_path),
            '--preds_path', str(output_file),'--number_of_molecules', '2', '--num_workers', '0',
            '--no_cuda']
    # '--preds_path', output_file, '--number_of_molecules', '2']

    if smiles==None:
        raise ValueError("At least One input: file or smiles")
    else:
        res = make_predictions(args=PredictArgs().parse_args(pred_args), smiles=smiles)
    return res
