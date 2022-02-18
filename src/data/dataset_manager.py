from pathlib import Path
from typing import Union
import pickle
import json
import os
import gc

import numpy as np
import pandas as pd

# TODO use h5py and ftputil

class DatasetManager:

  def __init__(self, project_id: str, datasets_dir: Union[Path, str] = "./datasets/", cache=False):
      self.project_id = project_id
      if isinstance(datasets_dir, str):
        datasets_dir = Path(datasets_dir)
      self.datasets_dir = datasets_dir

      # set up the dataset directory and path to it
      (datasets_dir / 'maps').mkdir(parents=True, exist_ok=True)

      # set up the map of the project
      self.map = datasets_dir / 'maps' / (project_id + '.json')

      # set up caching
      self.cache=cache
      self.loaded_files = {}


  def get_map(self):
    """Get map of files that are associated with this project.

    Returns:
        <Dict> of file aliases and file names.
    """
    if not self.map.exists():
        json.dump({}, open(self.map, 'w'))
        return {}
    return json.load(open(self.map, 'r'))

  
  def set_map(self, map):
    """Set the map of file associations of this project.

    Args:
        map (Dict) : map of associations.

    Returns:
        None.
    """
    json.dump(map, open(self.map, 'w'))


  def is_shared(self, name):
    for map in (self.datasets_dir / 'maps').glob('**/*.json'):
      if map != self.map and f in json.load(map.open('r')).values():
          return True
    return False


  def import_dataset(self, old_project_id: str, old_name: str, new_name: str):
    """Creates an association between a file used in a different project and 
    this project.

    Args:
        old_project_id (str) : Project under which the file was used previously.
        old_name (str) : Alias under which the file was stored in the previous project.
        new_name (str) : Alias under which the file is to be stored in this project.

    Returns:
        None.
    """
    other = DatasetManager(old_project_id, self.datasets_dir)

    map = self.get_map()
    map[new_name] = other.get_map()[old_name]
    self.set_map(map)

    return self.load_dataset(name)


  def add_dataset(self, ds, name: str):
    """Creates an association between this project and the given dataset. Also 
    saves the <pickle> dataset in <self.datasets_dir> folder.

    Args:
        ds : Dataset to be saved.
        name (str) : Alias under which the dataset can be then accessed.

    Returns:
        None.
    """
    # load data
    target = '{}.pkl'.format(max(int(f.stem) for f in self.datasets_dir.glob('**/*.pkl')) + 1)
    pickle.dump(ds, (self.datasets_dir / target).open('wb'))

    # update project map
    map = self.get_map()
    map[name] = target
    self.set_map(map)

  
  def load_dataset(self, name: str):
    """Loads a dataset saved through the <add_dataset> or <import_dataset> functionality.

    Args:
        name (str) : Alias of the file to be loaded.

    Returns:
        Dataset instance associated with this project under given alias.
    """
    map = self.get_map()
    if name not in map.keys():
      raise ValueError('Name not encountered. Other names: {}'.format(map.keys()))
    if self.cache and name in self.loaded_files.keys():
      print("Dataset {} loaded from cache. Use <clear_cache> and run this function again to avoid loading from cache.".format(name), flush=True)
      return self.loaded_files[name]
    try:
        df = pickle.load(open(self.datasets_dir / map[name], 'rb'))
    except FileNotFoundError:
        df = pickle.load(open(self.datasets_dir / (map[name].split(".")[0] + " (1)." + map[name].split(".")[1]), 'rb'))
    if self.cache:
      # TODO copy?
      self.loaded_files[name] = df
    return df


  def load_datasets(self, names):
    """Loads datasets saved through the <add_dataset> or <import_dataset> functionality.

    Args:
        names (list[str]) : List of file aliases to be loaded.

    Returns:
        Dataset instances associated with this project under given aliases.
    """
    return (self.load_dataset(name) for name in names)


  def remove_dataset(self, name: str, keep_file: bool=False):
    """Removes this file's association from this project.

    Args:
        keep_files (Bool) : Whether the file should be deleted, if it is not used in any other project.

    Returns:
        <True> if the file was deleted. <False> otherwise.
    """
    map = self.get_map()
    f = map.pop(name)
    self.set_map(map)

    if not keep_file and not self.is_shared(name):
      os.remove(self.datasets_dir / f)
      return True
    return False


  def end_project(self, keep_files: bool=False):
    """Dissociate any files attached to this project and call <clean_unused_files>.

    Args:
        keep_files (Bool) : Whether <clean_unused_files> should be called.

    Returns:
        None.
    """
    os.remove(self.map)
    if not keep_files:
      self.clean_unused_files()


  def clean_unused_files(self):
    """Deletes all '.pkl' files within <self.datasets_dir> that are not assigned to a project.

    Returns:
        None.
    """
    used = []
    for map in (self.datasets_dir / 'maps').glob('**/*.json'):
      used += json.load(map.open('r')).values()
    
    for f in (self.datasets_dir).glob('**/*.pkl'):
      if f.name not in used:
        os.remove(f)


  def modify_dataset(self, name, function):
    """Apply a function to a saved dataset.

    Returns:
        None.
    """
    df = function(self.load_dataset(name))
    self.add_dataset(df, name)


  def clear_cache(self):
    self.loaded_files = {}
    gc.collect()
