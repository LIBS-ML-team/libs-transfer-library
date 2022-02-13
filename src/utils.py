from typing import Iterable
import numpy as np

def round_to_pow2(x: int) -> int:
  """
  Round up to the nearets power of 2.
  """
  return 1<<(x-1).bit_length()
  
  
def spectra_pd_to_np(labels: Iterable[int], *args) -> np.ndarray:
  return np.concatenate([item.iloc[labels].to_numpy() for item in args])
