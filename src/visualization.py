from src.preprocessing import LabelCropp, Integrate
from src.map_utils import IndexType, map_from_list_data
from src.metrics.oop import make_scorer

from typing import TypeVar, Tuple, Iterable, Optional, Callable
from itertools import starmap

from sklearn.pipeline import Pipeline
from scipy.spatial.distance import cosine
import numpy as np
import pandas as pd
import tensorflow as tf

import plotly.express as px
import plotly.graph_objects as go
import colorlover as cl
from plotly.graph_objs._figure import Figure

from tqdm import tqdm

RGBSCALE = ['#1f77b4', '#d62728', '#2ca02c']
T = TypeVar('T')

def plot_spectra(spectra: np.ndarray,                           \
                 calibration=None,                              \
                 title: Optional[str]=None,                     \
                 labels: Optional[Iterable[str]]=None,          \
                 colormap=cl.scales['12']['qual']['Paired'],    \
                 axes_titles: bool=True,                        \
                 opacity: float = .7,                           \
                 ) -> Figure:
    if calibration is None:
        calibration = np.arange(len(spectra[0]))
    if labels is None:
        labels = ["class {}".format(x+1) for x in range(len(spectra))]
    fig = go.Figure()
    for i in range(len(spectra)):
        fig.add_trace(
            go.Scatter(
                x = calibration,
                y = spectra[i],
                name = str(labels[i]),
                line = {'color': colormap[i % len(colormap)]},
                opacity=opacity,
            )
        )
    fig.update_layout(
        title = title,
        xaxis_title = "wavelengths [nm]" if axes_titles else "",
        yaxis_title = "intensity [A.u.]" if axes_titles else "")
    return fig
    

def plot_map(spectra: np.ndarray, dimensions: Tuple[int, int],                \
             index_type: IndexType=IndexType.HORIZONTAL,                      \
             wave_from: Optional[T]=None, wave_to: Optional[T]=None,          \
             calibration: Optional[Iterable[T]]=None,                         \
             title: Optional[str]=None,                                       \
             color:Optional[str]=None,                                        \
             ) -> Figure:
  if calibration is None:
    calibration = np.arange(len(spectra.iloc[0]))
  
  if wave_from is None:
    wave_from = calibration[0]
  if wave_to is None:
    wave_to = calibration[-1]

  pipe = Pipeline([('cropp', LabelCropp(label_from=wave_from, label_to=wave_to, labels=calibration)), ('integral', Integrate())])
  spectra = pipe.fit_transform(spectra)

  if color is not None:
    fig = px.imshow(map_from_list_data(spectra, dimensions, index_type), title=title, color_continuous_scale=color)
  else:
    fig = px.imshow(map_from_list_data(spectra, dimensions, index_type), title=title)

  return fig
  
  
# TODO colormap type
def spectrometer_step_comparison(calibrations,                                  \
                                 labels: Optional[Iterable[str]]=None,          \
                                 colormap=RGBSCALE,                            \
                                 axis_title: bool = False,                      \
                                 step: int = 200,                               \
                                 title: Optional[str]=None,                     \
                                 ) -> Figure:
  if labels is None:
    labels = ['S{}'.format(i) for i in range(len(calibrations.columns))]
  
  fig = go.Figure()
  for i in range(len(calibrations)):
    fig.add_trace(
      go.Box(
        x = calibrations[i][::step],
        marker_symbol='line-ns-open', 
        marker_color=colormap[i % len(colormap)],
        boxpoints='all',
        jitter=0,
        name=labels[i],
        line = dict(color = 'rgba(0,0,0,0)'),
        fillcolor = 'rgba(0,0,0,0)'
      )
    )
      
  if axis_title:
    fig.update_layout(xaxis_title = "wavelengths [nm]", title=title)

  return fig
  

def error_map(y_true: Iterable[T],                                             \
              y_pred: Iterable[T],                                             \
              dim: Tuple[int, int],                                            \
              error_function: Callable[[T, T], float]=cosine,                  \
              index_type: IndexType=IndexType.HORIZONTAL,                      \
              title: Optional[str]=None,                                       \
              colormap=None,                                                   \
              add_stats: bool=False,                                   \
              ) -> Tuple[Figure, float]:
  alist = pd.DataFrame(starmap(error_function, zip(y_true, y_pred)))

  if add_stats:
    if not title:
      title = ''
    title += '(avg: {}, min: {}, max: {})'.format(*(round(float(i), 2) for i in (alist.mean(), alist.min(), alist.max())))

  fig = px.imshow(map_from_list_data(alist, dim, index_type), title=title, color_continuous_scale=colormap)

  return fig
  
  
def epoch_metric_plot(metrics: pd.DataFrame,                    \
                 title: str='',                                 \
                 colormap=RGBSCALE,                             \
                 opacity: float = .7,                           \
                 norm: bool = True,                             \
                 ) -> Figure:
    if norm:
      fig = go.Figure(layout_yaxis_range=[0, 1])
    else:
      fig = go.Figure()
    for i, name in enumerate(metrics.columns):
        fig.add_trace(
            go.Scatter(
                x = metrics.index,
                y = metrics[name],
                name = str(name),
                line=dict(color=colormap[i % len(colormap)]),
                opacity=opacity,
            )
        )
    fig.update_layout(xaxis_title = "Epoch", title=title)
    return fig
    
    
def param_tuning_plot(model, param_name:str, param_values:Iterable,            \
                      scorers:Iterable, X_train, y_train=None, X_val = None,   \
                      y_val = None, colormap=RGBSCALE, opacity:float=.7,       \
                      title:Optional[str]=None,                                \
                      *fit_args, **fit_kwargs):
  """
  Used for a fixed param and multiple scorers.
  """
  scorers = [make_scorer(scorer) for scorer in scorers]
  results = {s.get_name():[] for s in scorers} # could throw because of same names
  X_val = X_val if X_val is not None else X_train
  y_val = y_val if y_val is not None else y_train
  for val in param_values:
    model.set_params(**{param_name:val})
    model.fit(X_train, y_train, *fit_args, **fit_kwargs)
    for scorer in scorers:
      results[scorer.get_name()] = scorer.score(model=model, y_true = y_val, y_pred = model.predict(X_val))
    gc.collect()
  
  # TODO standalone plot?
  fig = go.Figure()
  for name in results.keys:
    fig.add_trace(go.Scatter(x=param_values, y=results[name],
                      mode='lines',
                      line=dict(color=colormap[i % len(colormap)]),
                      opacity=opacity,
                      name=name))
  fig.update_layout(xaxis_title = "Parameter", yaxis_title="Metric", title=title)
  return fig
  
  
def reduce_plot(fig: Figure, width: int=1000, height: int=250, showlegend: bool=True) -> Figure:
    fig.update_layout(
    width=width,
    height=height,
    showlegend=showlegend,
    coloraxis=None,
    margin=dict(
        l=10,
        r=10,
        b=10,
        t=10,
        pad=4
    ))
    return fig
    
    
def add_markers(fig: Figure, markers=Iterable[Tuple[int, int]], colormap=RGBSCALE, size: int=10) -> Figure:
  for i in range(len(colors)):
    fig.add_trace(
                  go.Scatter(
                      x = [markers[i][0]],
                      y = [markers[i][1]],
                      mode="markers",
                      marker = {'color':colormap[i % len(colormap)], 'size':size},
              ))
  return fig
  
  
  from src.preprocessing import LabelCropp, Integrate
from src.map_utils import IndexType, map_from_list_data
from src.metrics.oop import make_scorer

from typing import TypeVar, Tuple, Iterable, Optional, Callable
from itertools import starmap

from sklearn.pipeline import Pipeline
from scipy.spatial.distance import cosine
import numpy as np
import pandas as pd
import tensorflow as tf

import plotly.express as px
import plotly.graph_objects as go
import colorlover as cl
from plotly.graph_objs._figure import Figure

from tqdm import tqdm

import gc


RGBSCALE = ['#1f77b4', '#d62728', '#2ca02c']
T = TypeVar('T')

def param_tuning_plot(model, param_name:str, param_values:Iterable,            \
                      scorers:Iterable, X_train, y_train=None, X_val = None,   \
                      y_val = None, colormap=RGBSCALE, opacity:float=.7,       \
                      title:Optional[str]=None, cv=5,                          \
                      *fit_args, **fit_kwargs):
  """
  Used for a fixed param and multiple scorers.
  """
  scorers = [make_scorer(scorer) for scorer in scorers]
  results = {s.get_name():[0]*len(param_values) for s in scorers} # could throw because of same names
  X_val = X_val if X_val is not None else X_train
  y_val = y_val if y_val is not None else y_train
  for i, val in enumerate(tqdm(param_values)):
    for _ in range(cv):
      tf.keras.backend.clear_session()
      gc.collect()
      model.set_params(**{param_name:val})
      model.fit(X_train, y_train, *fit_args, **fit_kwargs)
      y_pred = model.predict(X_val)
      for scorer in scorers:
        results[scorer.get_name()][i] += scorer.score(model=model, y_true = y_val, y_pred = y_pred)

  # TODO standalone plot?
  fig = go.Figure()
  for i, name in enumerate(results.keys()):
    fig.add_trace(go.Scatter(x=param_values, y=[val / max(results[name]) for val in results[name]],
                      text = [str(val / cv) for val in results[name]],
                      mode='lines',
                      line=dict(color=colormap[i % len(colormap)]),
                      opacity=opacity,
                      name=name))
  fig.update_layout(xaxis_title = "Parameter", yaxis_title="Metric", title=title)
  return fig
