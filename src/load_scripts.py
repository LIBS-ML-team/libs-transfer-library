import h5py
from pathlib import Path
from typing import Union, Tuple
import pickle
import json
import os
import gc
from tqdm import tqdm

import numpy as np
import pandas as pd

# TODO output check, verbose

def load_all_libsdata(path_to_folder: Union[str, Path]) -> Tuple[pd.DataFrame, list, pd.Series]:
  """
  Function for loading .libsdata and corresponding .libsmetadata files. Scans
  the entire folder for any such files.

  Args:
      path_to_folder (str or Path) : path to the folder to be scanned.

  Returns:
      pd.DataFrame : combined .libsdata files
      list : list of .libsmetadata files
      pd.Series : list of file labels for each entry. Can be used to connect each 
      entry to the file it originated from. 
  """
  data, metadata, samples = [], [], []
  if isinstance(path_to_folder, str):
    path_to_folder = Path(path_to_folder)
  for f in tqdm(path_to_folder.glob('**/*.libsdata')):
    try:
      meta = json.load(open(f.with_suffix('.libsmetadata'), 'r'))
    except:
      print('[WARNING] Failed to load metadata for file {}! Skipping!!!'.format(f))
      continue

    df = np.fromfile(open(f, 'rb'), dtype=np.float32)
    df = np.reshape(df, (meta['spectra'] + 1, meta['wavelengths']))
    df = pd.DataFrame(df[1:], columns=df[0])

    data.append(df)
    metadata.append(meta)
    samples += [f.stem.split('_')[0] for _ in range(len(df))]
  data = pd.concat(data, ignore_index=True)
  samples = pd.Series(samples)
  return data, metadata, samples


def load_libsdata(path_to_file: Union[str, Path]) -> Tuple[pd.DataFrame, dict]:
  """
  Function for loading a .libsdata and the corresponding .libsmetadata file.

  Args:
      path_to_file (str or Path) : path to the .libsdata or .libsmetadata file 
      to be loaded. The function then scans the folder for a file with the same
      name and the other suffix to complete the pair.

  Returns:
      pd.DataFrame : loaded data file
      dict : metadata
  """
  data, metadata = None, None
  if isinstance(path_to_file, str):
    path_to_file = Path(path_to_file)
  for f in path_to_file.parents[0].iterdir():
    if path_to_file.stem in f.stem:
      if f.suffix == '.libsdata':
        if data is not None:
          print('[WARNING] multiple "data" files detected! Using first found!!!')
        else:
          data = np.fromfile(open(f, 'rb'), dtype=np.float32)
      elif f.suffix == '.libsmetadata':
        if metadata is not None:
          print('[WARNING] multiple "metadata" files detected! Using first found!!!')
        else:
          metadata = json.load(open(f))
      else:
        print('[WARNING] unrecognized extension for file {}! Skipping!!!'.format(f))
        continue
  if data is None or metadata is None:
    raise ValueError('Data or metadata missing!')
  data = np.reshape(data, (int(metadata['spectra']) + 1, int(metadata['wavelengths'])))
  data = pd.DataFrame(data[1:], columns=data[0])
  return data, metadata


def load_contest_test_dataset(path_to_data: Union[Path, str], min_block: int=0, max_block: int=-1) -> Tuple[pd.DataFrame, pd.Series]:
  """
  Function for loading the contest test dataset.

  Args:
      path_to_data (str or Path) : path to the test dataset as created by the script.
      min_block (int) : Allows for the selection of a specific block from the 
                        original dataset. The function slices between <min_block>
                        and <max_block>.
      max_block (int) : Allows for the selection of a specific block from the 
                        original dataset. The function slices between <min_block>
                        and <max_block>.

  Returns:
      pd.DataFrame : X
      pd.Series : y
  """
  # TODO utilize a more abstract function for loading h5 data
  # TODO add downloading
  if isinstance(path_to_data, str):
      path_to_data = Path(path_to_data)
  test_data = np.ndarray((20000, 40002))
  with h5py.File(path_to_data, 'r') as test_file:
      wavelengths = train_file["Wavelengths"]["1"][:]
      for i_block, block in tqdm(test_file["UNKNOWN"].items()[min_block:max_block]):
          spectra = block[:].transpose()
          for i_spec in range(10000):
              test_data[(10000*(int(i_block)-1))+i_spec] = spectra[i_spec]
          del spectra
  test = pd.DataFrame(test_data, columns=wavelengths)
  labels = pd.DataFrame.pop('label')
  return test, labels


def load_contest_train_dataset(path_to_data: Union[Path, str], spectra_per_sample: int=100) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
  """
  Function for loading the contest train dataset.

  Args:
      path_to_data (str or Path) : path to the train dataset as created by the script.
      spectra_per_sample (int) : how many spectra will be taken from each sample.

  Returns:
      pd.DataFrame : X
      pd.Series : y
      pd.Series : list of sample labels for each entry. Can be used to connect each 
      entry to the file it originated from. 
  """
  if isinstance(path_to_data, str):
    path_to_data = Path(path_to_data)
  with h5py.File(path_to_data, 'r') as train_file:
    # Store wavelengths (calibration)
    wavelengths = pd.Series(train_file['Wavelengths']['1'])
    wavelengths = wavelengths.round(2).drop(index=[40000, 40001])

    # Store class labels
    labels = pd.Series(train_file['Class']['1']).astype(int)

    # Store spectra
    samples_per_class = labels.value_counts(sort=False) // 500
    spectra = np.empty(shape=(0, 40000))
    samples = []
    classes = []

    lower_bound = 1
    for i_class in tqdm(samples_per_class.keys()):
      for i_sample in range(lower_bound, lower_bound + samples_per_class[i_class]):
        sample = train_file["Spectra"][f"{i_sample:03d}"]
        sample = np.transpose(sample[:40000, :spectra_per_sample])
        spectra = np.concatenate([spectra, sample])
        samples.extend(np.repeat(i_sample, spectra_per_sample))
        classes.extend(np.repeat(i_class, spectra_per_sample))
      lower_bound += samples_per_class[i_class]

  samples = pd.Series(samples)
  classes = pd.Series(classes)
  return pd.DataFrame(spectra, columns=wavelengths), classes, samples


def contest_train_val_split(X, y, samples, train_size: int = .6, sps: int = 100, random_state: int=42) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
  """
  Function for splitting the contest train dataset with respect to samples.

  Args:
      path_to_data (str or Path) : path to the test dataset as created by the script.
      min_block (int) : Allows for the selection of a specific block from the 
                        original dataset. The function slices between <min_block>
                        and <max_block>.
      max_block (int) : Allows for the selection of a specific block from the 
                        original dataset. The function slices between <min_block>
                        and <max_block>.

  Returns:
      pd.DataFrame : X
      pd.Series : y
  """
  rng = np.random.default_rng(random_state)
  X['class'] = y
  X['sample'] = samples
  val_size = 1 - train_size
  train_samples = []
  val_samples = []
  for i_class in range(1, 13):
      samples = X[X['class'] == i_class]['sample'].values
      unq_sample_labels = np.unique(samples)
      train_samples.extend(rng.choice(unq_sample_labels,
                                      int(len(unq_sample_labels) * train_size),
                                      replace=False))
      val_samples.extend(rng.choice(np.setdiff1d(unq_sample_labels, train_samples),
                                    size=int(len(unq_sample_labels) * val_size),
                                    replace=False))
  val_subset = X[X['sample'].isin(val_samples)]
  X = X[X['sample'].isin(train_samples)]
  y_val = val_subset['class']
  x_val = val_subset.iloc[:, :-2]
  del val_subset
  y_train = X['class']
  x_train = X.iloc[:, :-2]
  # del would also destroy the dataset out of scope
  X = None
  return x_train, y_train, x_val, y_val
