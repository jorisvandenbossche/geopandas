from __future__ import print_function

import warnings

import numpy as np
from six import next
from six.moves import xrange
from shapely.geometry import Polygon
from matplotlib.collections import LineCollection
from matplotlib.colors import colorConverter


def is_uniform_geom_type(s):
    return (s.geom_type == s.geom_type.iloc[0]).all()


def plot_polygon(ax, poly, facecolor='red', edgecolor='black', alpha=0.5, linewidth=1.0):
    """ Plot a single Polygon geometry """
    from descartes.patch import PolygonPatch
    a = np.asarray(poly.exterior)
    if poly.has_z:
        poly = Polygon(zip(*poly.exterior.xy))

    # without Descartes, we could make a Patch of exterior
    ax.add_patch(PolygonPatch(poly, facecolor=facecolor, linewidth=0, alpha=alpha))  # linewidth=0 because boundaries are drawn separately
    ax.plot(a[:, 0], a[:, 1], color=edgecolor, linewidth=linewidth)
    for p in poly.interiors:
        x, y = zip(*p.coords)
        ax.plot(x, y, color=edgecolor, linewidth=linewidth)


def plot_multipolygon(ax, geom, facecolor='red', edgecolor='black', alpha=0.5, linewidth=1.0):
    """ Can safely call with either Polygon or Multipolygon geometry
    """
    if geom.type == 'Polygon':
        plot_polygon(ax, geom, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, linewidth=linewidth)
    elif geom.type == 'MultiPolygon':
        for poly in geom.geoms:
            plot_polygon(ax, poly, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, linewidth=linewidth)


def plot_polygon_collection(ax, geoms, values=None, colormap='Set1', facecolor=None, edgecolor=None,
                            alpha=0.5, linewidth=1.0, **kwargs):
    """ Plot a collection of Polygon geometries """

    from descartes.patch import PolygonPatch
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    patches = []

    for poly in geoms:

        a = np.asarray(poly.exterior)
        if poly.has_z:
            poly = shapely.geometry.Polygon(zip(*poly.exterior.xy))

        patches.append(Polygon(a))

    patches = PatchCollection(patches, facecolor=facecolor, linewidth=linewidth,
                              edgecolor=edgecolor, alpha=alpha, **kwargs)

    if values is not None:
        # as array -> categorical "AttributeError: 'list' object has no attribute 'ndim'" in self._A
        patches.set_array(np.asarray(values))
        patches.set_cmap(colormap)

    ax.add_collection(patches, autolim=True)
    ax.autoscale_view()
    return patches


def plot_linestring(ax, geom, color='black', linewidth=1.0):
    """ Plot a single LineString geometry """
    a = np.array(geom)
    ax.plot(a[:, 0], a[:, 1], color=color, linewidth=linewidth)


def plot_multilinestring(ax, geom, color='red', linewidth=1.0):
    """ Can safely call with either LineString or MultiLineString geometry
    """
    if geom.type == 'LineString':
        plot_linestring(ax, geom, color=color, linewidth=linewidth)
    elif geom.type == 'MultiLineString':
        for line in geom.geoms:
            plot_linestring(ax, line, color=color, linewidth=linewidth)


def plot_linestring_collection(ax, geoms, values=None, colormap='Set1', color=None, linewidth=1, **kwargs):
    """ Plot a single LineString geometry """
    lines = LineCollection([np.array(geom)[:, :2] for geom in geoms], linewidth=linewidth, color=color, **kwargs)
    if values is not None:
        lines.set_array(values)
        lines.set_cmap(colormap)
    ax.add_collection(lines, autolim=True)
    ax.autoscale_view()
    return ax, lines


def plot_point(ax, pt, marker='o', markersize=2, color='black'):
    """ Plot a single Point geometry """
    ax.plot(pt.x, pt.y, marker=marker, markersize=markersize, linewidth=0,
            color=color)


def plot_point_collection(ax, geom, values=None, marker='o', markersize=2, colormap=None, color=None, **kwargs):
    """Plot a collection of Point geometries"""
    x = [p.x for p in geom]
    y = [p.y for p in geom]
    if values is None:
        values = 'none'
    collection = ax.scatter(x, y, s=markersize, c=values, cmap=colormap, marker=marker, color=color, **kwargs)
    return collection


def gencolor(N, colormap='Set1'):
    """
    Color generator intended to work with one of the ColorBrewer
    qualitative color scales.

    Suggested values of colormap are the following:

        Accent, Dark2, Paired, Pastel1, Pastel2, Set1, Set2, Set3

    (although any matplotlib colormap will work).
    """
    from matplotlib import cm
    # don't use more than 9 discrete colors
    n_colors = min(N, 9)
    cmap = cm.get_cmap(colormap, n_colors)
    colors = cmap(range(n_colors))
    for i in xrange(N):
        yield colors[i % n_colors]


def plot_series(s, colormap=None, color=None, ax=None, linewidth=1.0,
                figsize=None, **color_kwds):
    """ Plot a GeoSeries

        Generate a plot of a GeoSeries geometry with matplotlib.

        Parameters
        ----------

        Series
            The GeoSeries to be plotted.  Currently Polygon,
            MultiPolygon, LineString, MultiLineString and Point
            geometries can be plotted.

        colormap : str (default 'Set1')
            The name of a colormap recognized by matplotlib.  Any
            colormap will work, but categorical colormaps are
            generally recommended.  Examples of useful discrete
            colormaps include:

                Accent, Dark2, Paired, Pastel1, Pastel2, Set1, Set2, Set3

        ax : matplotlib.pyplot.Artist (default None)
            ax on which to draw the plot

        linewidth : float (default 1.0)
            Line width for geometries.

        figsize : pair of floats (default None)
            Size of the resulting matplotlib.figure.Figure. If the argument
            ax is given explicitly, figsize is ignored.

        **color_kwds : dict
            Color options to be passed on to plot_polygon

        Returns
        -------

        matplotlib axes instance
    """
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect('equal')
    if colormap is None:
        if color is None:
            if 'facecolor' not in color_kwds:
                colormap = 'Set1'
    if is_uniform_geom_type(s):
        geom_type = s.geom_type.iloc[0]
        if colormap is not None:
            values = np.arange(len(s))
        else:
            values = None
        if geom_type.startswith('LineString'):
            # all the same types -> we can use Collections
            plot_linestring_collection(ax, s, values, colormap=colormap, color=color, linewidth=linewidth, **color_kwds)
        elif geom_type == 'Point':
            plot_point_collection(ax, s, values, colormap=colormap, color=color, linewidth=linewidth, **color_kwds)
        elif geom_type == 'Polygon':
            plot_polygon_collection(ax, s, values, colormap=colormap, color=color, linewidth=linewidth, **color_kwds)

    else:
        color_generator = gencolor(len(s), colormap=colormap)
        for geom in s:
            if color is None:
                col = next(color_generator)
            else:
                col = color
            if geom.type == 'Polygon' or geom.type == 'MultiPolygon':
                plot_multipolygon(ax, geom, facecolor=col, linewidth=linewidth, **color_kwds)
            elif geom.type == 'LineString' or geom.type == 'MultiLineString':
                plot_multilinestring(ax, geom, color=col, linewidth=linewidth)
            elif geom.type == 'Point':
                plot_point(ax, geom, color=col)
    plt.draw()
    return ax


def plot_dataframe(s, column=None, colormap=None, color=None, linewidth=1.0,
                   categorical=False, legend=False, ax=None,
                   scheme=None, k=5, vmin=None, vmax=None, figsize=None,
                   **color_kwds
                   ):
    """ Plot a GeoDataFrame

        Generate a plot of a GeoDataFrame with matplotlib.  If a
        column is specified, the plot coloring will be based on values
        in that column.  Otherwise, a categorical plot of the
        geometries in the `geometry` column will be generated.

        Parameters
        ----------

        GeoDataFrame
            The GeoDataFrame to be plotted.  Currently Polygon,
            MultiPolygon, LineString, MultiLineString and Point
            geometries can be plotted.

        column : str (default None)
            The name of the column to be plotted.

        categorical : bool (default False)
            If False, colormap will reflect numerical values of the
            column being plotted.  For non-numerical columns (or if
            column=None), this will be set to True.

        colormap : str (default 'Set1')
            The name of a colormap recognized by matplotlib.

        linewidth : float (default 1.0)
            Line width for geometries.

        legend : bool (default False)
            Plot a legend (Experimental; currently for categorical
            plots only)

        ax : matplotlib.pyplot.Artist (default None)
            axes on which to draw the plot

        scheme : pysal.esda.mapclassify.Map_Classifier
            Choropleth classification schemes (requires PySAL)

        vmin : float
            Minimum value for color map.

        vmax : float
            Maximum value for color map.

        k   : int (default 5)
            Number of classes (ignored if scheme is None)

        vmin : None or float (default None)

            Minimum value of colormap. If None, the minimum data value
            in the column to be plotted is used.

        vmax : None or float (default None)

            Maximum value of colormap. If None, the maximum data value
            in the column to be plotted is used.

        figsize
            Size of the resulting matplotlib.figure.Figure. If the argument
            ax is given explicitly, figsize is ignored.

        **color_kwds : dict
            Color options to be passed on to plot_polygon

        Returns
        -------

        matplotlib axes instance
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.colors import Normalize
    from matplotlib import cm

    if column is None:
        return plot_series(s.geometry, colormap=colormap, color=color,
                           ax=ax, linewidth=linewidth, figsize=figsize,
                           **color_kwds)
    else:
        if s[column].dtype is np.dtype('O'):
            categorical = True
        if categorical:
            if colormap is None:
                colormap = 'Set1'
            categories = list(set(s[column].values))
            categories.sort()
            valuemap = dict([(k, v) for (v, k) in enumerate(categories)])
            values = [valuemap[k] for k in s[column]]
        else:
            values = s[column]
        if scheme is not None:
            binning = __pysal_choro(values, scheme, k=k)
            values = binning.yb
            # set categorical to True for creating the legend
            categorical = True
            binedges = [binning.yb.min()] + binning.bins.tolist()
            categories = ['{0:.2f} - {1:.2f}'.format(binedges[i], binedges[i+1])
                          for i in range(len(binedges)-1)]
        cmap = norm_cmap(values, colormap, Normalize, cm, vmin=vmin, vmax=vmax)
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_aspect('equal')

        if is_uniform_geom_type(s):
            # all the same types -> we can use Collections
            geom_type = s.geom_type.iloc[0]
            if geom_type.startswith('LineString'):
                plot_linestring_collection(ax, s.geometry, values, colormap=colormap, linewidth=linewidth, **color_kwds)
            elif geom_type == 'Point':
                plot_point_collection(ax, s.geometry, values, colormap=colormap, color=color, linewidth=linewidth, **color_kwds)
            elif geom_type == 'Polygon':
                plot_polygon_collection(ax, s.geometry, values, colormap=colormap, color=color, linewidth=linewidth, **color_kwds)
        else:
            for geom, value in zip(s.geometry, values):
                if color is None:
                    col = cmap.to_rgba(value)
                else:
                    col = color
                if geom.type == 'Polygon' or geom.type == 'MultiPolygon':
                    plot_multipolygon(ax, geom, facecolor=col, linewidth=linewidth, **color_kwds)
                elif geom.type == 'LineString' or geom.type == 'MultiLineString':
                    plot_multilinestring(ax, geom, color=col, linewidth=linewidth)
                elif geom.type == 'Point':
                    plot_point(ax, geom, color=col)
        if legend:
            if categorical:
                patches = []
                for value, cat in enumerate(categories):
                    patches.append(Line2D([0], [0], linestyle="none",
                                          marker="o", alpha=color_kwds.get('alpha', 0.5),
                                          markersize=10, markerfacecolor=cmap.to_rgba(value)))
                ax.legend(patches, categories, numpoints=1, loc='best')
            else:
                # TODO: show a colorbar
                raise NotImplementedError
    plt.draw()
    return ax


def __pysal_choro(values, scheme, k=5):
    """ Wrapper for choropleth schemes from PySAL for use with plot_dataframe

        Parameters
        ----------

        values
            Series to be plotted

        scheme
            pysal.esda.mapclassify classificatin scheme
            ['Equal_interval'|'Quantiles'|'Fisher_Jenks']

        k
            number of classes (2 <= k <=9)

        Returns
        -------

        binning
            Binning objects that holds the Series with values replaced with
            class identifier and the bins.
    """

    try:
        from pysal.esda.mapclassify import Quantiles, Equal_Interval, Fisher_Jenks
        schemes = {}
        schemes['equal_interval'] = Equal_Interval
        schemes['quantiles'] = Quantiles
        schemes['fisher_jenks'] = Fisher_Jenks
        s0 = scheme
        scheme = scheme.lower()
        if scheme not in schemes:
            scheme = 'quantiles'
            warnings.warn('Unrecognized scheme "{0}". Using "Quantiles" '
                          'instead'.format(s0), UserWarning, stacklevel=3)
        if k < 2 or k > 9:
            warnings.warn('Invalid k: {0} (2 <= k <= 9), setting k=5 '
                          '(default)'.format(k), UserWarning, stacklevel=3)
            k = 5
        binning = schemes[scheme](values, k)
        return binning
    except ImportError:
        raise ImportError("PySAL is required to use the 'scheme' keyword")


def norm_cmap(values, cmap, normalize, cm, vmin=None, vmax=None):

    """ Normalize and set colormap

        Parameters
        ----------

        values
            Series or array to be normalized

        cmap
            matplotlib Colormap

        normalize
            matplotlib.colors.Normalize

        cm
            matplotlib.cm

        vmin
            Minimum value of colormap. If None, uses min(values).

        vmax
            Maximum value of colormap. If None, uses max(values).

        Returns
        -------
        n_cmap
            mapping of normalized values to colormap (cmap)

    """

    mn = vmin or min(values)
    mx = vmax or max(values)
    norm = normalize(vmin=mn, vmax=mx)
    n_cmap = cm.ScalarMappable(norm=norm, cmap=cmap)
    return n_cmap
