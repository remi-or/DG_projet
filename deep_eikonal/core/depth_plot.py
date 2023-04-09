from typing import Tuple, Dict, Any, List, Union
import torch
from torch import Tensor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from matplotlib import colormaps

def dist_facecolor(vertices: Tensor, # V, 3
                   faces: Tensor, # 3, F
                   origin: Tensor | None = None, # 3,
                   noise: float = 0.0,
                   cmap: str = 'plasma',
                   ) -> np.ndarray:
    """Colors the (faces) of a mesh resting on some (vertices). The color is 
    proportionnal to the distance between a face's center of gravity and the 
    given (origin). Additional color noise is added if (noise) is above zero,
    in the range of [-noise, +noise]. One can choose the color map by its name
    (cmap). Returns an array of size (nb_faces, 4) where each line is the color 
    of a  face. """
    origin = torch.ones(3) if origin is None else origin
    # Compute faces' distances
    d_min, distances, d_max = np.inf, [], -1
    for face in faces.T:
        center = vertices[face].mean(0)
        distance = torch.dist(origin, center).item()
        if distance > d_max:    d_max = distance
        if distance < d_min:    d_min = distance
        distances.append(distance)
    # Normalize distances
    distances = np.array(distances)
    distances = (distances - d_min) / (d_max - d_min)
    # Turn distances to color
    colors = colormaps[cmap](1 - distances)
    # Eventualy add color noise
    if noise > 0:
        colors[:, :3] += (2*noise) * (np.random.rand(colors.shape[0], 3)-0.5)
        colors[colors > 1] = 1
        colors[colors < 0] = 0
    return colors

def draw_mesh(vertices: Tensor, # V, 3
              faces: Tensor, # 3, F
              ax: plt.Axes,
              facecolors: np.ndarray | None = None,
              ) -> Poly3DCollection:
    """Draws a mesh resting on (vertices) defined by (faces). The drawing is 
    done on a (ax) object where projection='3d' and facecolors can be applied
    through (facecolors)."""
    explicit_faces = []
    for face in faces.T:
        explicit_faces.append( vertices[face].numpy() )
    collection = Poly3DCollection(verts=explicit_faces, 
                                  facecolors=facecolors, 
                                  shade=True)
    return ax.add_collection3d(collection)

def subplots3d(nrows: int,
               ncols: int,
               figsize: Tuple[int, int] = (8, 8),
               camera: Tuple[int, int, int] = (10, 30, 0),
               named_axes: bool = True,
               ) -> Tuple[plt.Figure, Union[plt.Axes, List[plt.Axes]]]:
    """Creates a figure of size (figsize) with (nrows) rows and (ncols) columns
    where each ax is a 3d ax with a view oriented along to (camera). If the 
    (named_axes) flag is set to True, axis names are set to 'x',... Returns a 
    tuple of {fig} and {axs} where {axs} is a list of ax starting in the top
    right corner."""
    fig = plt.figure(figsize=figsize)
    axs = []
    for i in range(nrows * ncols):
        axs.append(fig.add_subplot(nrows, ncols, i+1, projection='3d'))
        axs[-1].view_init(*camera)
        if named_axes:
            for c in 'xyz':
                getattr(axs[-1], f"set_{c}label")(c)
    if nrows*ncols == 1:    
        axs = axs[0]
    return fig, axs

def format_vf(vertices: Tensor | np.ndarray,
              faces: Tensor | np.ndarray,
              normalize: bool = True,
              ) -> Tuple[Tensor, Tensor]:
    """TODO"""
    # vertices type
    vertices = torch.tensor(vertices) if isinstance(vertices, np.ndarray) \
        else vertices
    vertices = vertices.float()
    if not isinstance(vertices, Tensor):
        raise TypeError(f"Invalied type for vertices: {type(vertices) = }")
    # vertices shape
    if vertices.size(1) != 3:
        if vertices.size(0) == 3:
            vertices = vertices.transpose(0, 1)
        else:
            raise ValueError(f"Invalid vertices shape: {vertices.shape = }")
    if len(vertices.shape) != 2:
        raise ValueError(f"Invalid vertices shape: {vertices.shape = }")
    # eventualy normalize vertices
    if normalize:
        # vertices -= vertices.min().item()
        vertices /= vertices.max().item()
        vertices[:, :2] += 0.5
        vertices[:, 2] -= vertices[:, 2].min().item()
    # faces type
    faces = torch.tensor(faces) if isinstance(faces, np.ndarray) else faces
    if not isinstance(faces, Tensor):
        raise TypeError(f"Invalied type for faces: {type(faces) = }")
    faces = faces.long()
    # faces shape
    if faces.size(0) != 3:
        if faces.size(1) == 3:
            faces = faces.transpose(0, 1)
        else:
            raise ValueError(f"Invalid faces shape: {faces.shape = }")
    if len(faces.shape) != 2:
        raise ValueError(f"faces vertices shape: {faces.shape = }")
    return vertices, faces