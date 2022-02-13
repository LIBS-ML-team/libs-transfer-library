from enum import Enum, auto
from typing import Generator, Tuple
import pandas as pd
import numpy as np

class IndexType(Enum):
  """
  Class describing types of index to two dimensional space mappings.
  """
  HORIZONTAL_SNAKE = auto()
  VERTICAL_SNAKE = auto()
  HORIZONTAL = auto()
  VERTICAL = auto()
  HORIZONTAL_SNAKE_OFFSET = auto()
  HORIZONTAL_SNAKE_OFFSET_ERROR = auto()
  HORIZONTAL_SNAKE_OFFSET_ERROR2 = auto()



def make_index_generator(dimensions: Tuple[int, int], index_type: IndexType) -> Generator[Tuple[int, int], None, None]:
  """
  Creates a generator yielding 2d coordinate pairs according to the <IndexType>.
  """
  if index_type == IndexType.HORIZONTAL_SNAKE:
    for j in range(dimensions[1]):
      if j % 2:
        for i in range(dimensions[0] - 1, -1, -1):
          yield i, j
      else:
        for i in range(dimensions[0]):
          yield i, j

  elif index_type == IndexType.VERTICAL_SNAKE:
    for i in range(dimensions[0]):
      if i % 2:
        for j in range(dimensions[1] - 1, -1, -1):
          yield i, j
      else:
        for j in range(dimensions[1]):
          yield i, j

  elif index_type == IndexType.HORIZONTAL:
    for j in range(dimensions[1]):
      for i in range(dimensions[0]):
        yield i, j

  elif index_type == IndexType.VERTICAL:
    for i in range(dimensions[0]):
      for j in range(dimensions[1]):
        yield i, j

  elif index_type == IndexType.HORIZONTAL_SNAKE_OFFSET:
    for j in range(dimensions[1]):
      if j % 2:
        for i in range(dimensions[0] - 1, -1, -1):
          yield i + 1, j
      else:
        for i in range(dimensions[0]):
          yield i - 1, j
          
  elif index_type == IndexType.HORIZONTAL_SNAKE_OFFSET_ERROR:
    for j in range(dimensions[1]):
      if j % 2:
        for i in range(dimensions[0] - 1, -1, -1):
          yield i - 3, j
      else:
        for i in range(dimensions[0]):
          yield i - 1, j
          
  elif index_type == IndexType.HORIZONTAL_SNAKE_OFFSET_ERROR2:
    for j in range(dimensions[1]):
      if j % 2:
        for i in range(dimensions[0] - 1, -1, -1):
          yield i + 1, j + 2
      else:
        for i in range(dimensions[0]):
          yield i - 1, j

  else:
    raise ValueError('Unrecognized index type!')


def map_from_list_data(data: pd.DataFrame, dimensions: Tuple[int, int], index_type: IndexType=IndexType.HORIZONTAL_SNAKE) -> pd.DataFrame:
  """
  Creates a 2d map from a list given it's dimensions.
  """
  matrix = pd.DataFrame(np.zeros(dimensions[::-1])).astype(float)

  missing = []
  for i, (x, y) in enumerate(make_index_generator(dimensions, index_type)):
    try:
      if data.shape[1] == 1:
        matrix[x][y] = data.loc[i]
      else:
        matrix[x][y] = i
    except (IndexError, KeyError) as e:
      missing.append(i)

  if missing:
    print('[WARNING] Encountered missing measurements with indices {}. Values set to <0.>.'.format(missing))
    
  # TODO remove transpose
  return matrix.transpose()
