import collections
import numbers

import shapely
from shapely.geometry.base import BaseGeometry
from shapely.geometry.base import (
    GEOMETRY_TYPES as GEOMETRY_NAMES, CAP_STYLE, JOIN_STYLE)

from pandas.core.arrays import ExtensionArray
from pandas.core.dtypes.base import ExtensionDtype

import numpy as np

from . import vectorized


GEOMETRY_TYPES = [getattr(shapely.geometry, name) for name in GEOMETRY_NAMES]

opposite_predicates = {'contains': 'within',
                       'intersects': 'intersects',
                       'touches': 'touches',
                       'covers': 'covered_by',
                       'crosses': 'crosses',
                       'overlaps': 'overlaps'}

for k, v in list(opposite_predicates.items()):
    opposite_predicates[v] = k


def to_shapely(geoms):
    """ Convert array of pointers to an array of shapely objects """
    return vectorized.to_shapely(geoms)


def from_shapely(L):
    """ Convert a list or array of shapely objects to a GeometryArray """
    out = vectorized.from_shapely(L)
    return GeometryArray(out)


def from_wkb(L):
    """ Convert a list or array of WKB objects to a GeometryArray """
    out = vectorized.from_wkb(L)
    return GeometryArray(out)


def from_wkt(L):
    """ Convert a list or array of WKT objects to a GeometryArray """
    out = vectorized.from_wkt(L)
    return GeometryArray(out)


def points_from_xy(x, y):
    """ Convert numpy arrays of x and y values to a GeometryArray of points """
    out = vectorized.points_from_xy(x, y)
    return GeometryArray(out)


class GeometryDtype(ExtensionDtype):
    type = BaseGeometry
    name = 'geometry'

    @classmethod
    def construct_from_string(cls, string):
        if string == cls.name:
            return cls()
        else:
            raise TypeError("Cannot construct a '{}' from "
                            "'{}'".format(cls, string))


class GeometryArray(ExtensionArray):
    dtype = GeometryDtype()

    def __init__(self, data, base=False):
        if isinstance(data, self.__class__):
            base = [data]
            data = data.data
        elif not (getattr(data, 'dtype', None) == np.uintp):
            data = from_shapely(data)
            base = [data]
            data = data.data
        self.data = data
        self.base = base

    def __getitem__(self, idx):
        if isinstance(idx, numbers.Integral):
            return vectorized.get_element(self.data, idx)
        elif isinstance(idx, (collections.Iterable, slice)):
            return GeometryArray(self.data[idx], base=self)
        else:
            raise TypeError("Index type not supported", idx)

    def __len__(self):
        return len(self.data)

    @property
    def size(self):
        return len(self.data)

    @property
    def ndim(self):
        return 1

    def __del__(self):
        if self.base is False:
            try:
                vectorized.vec_free(self.data)
            except (TypeError, AttributeError):
                # the vectorized module can already be removed, therefore
                # ignoring such an error to not output this as a warning
                pass

    def copy(self):
        return self  # assume immutable for now

    def take(self, idx, **kwargs):

        # take on empty array
        if not len(self):
            # only valid if result is an all-missing array
            if (np.asarray(idx) == -1).all():
                return GeometryArray(
                    np.array([0]*len(idx), dtype=self.data.dtype))
            else:
                raise IndexError(
                    "cannot do a non-empty take from an empty array.")

        result = self[idx]
        result.data[idx == -1] = 0
        return result

    def fill(self, idx, value):
        """ Fill index locations with value

        Value should be a BaseGeometry

        Returns a copy
        """
        base = [self]
        if isinstance(value, BaseGeometry):
            base.append(value)
            value = value.__geom__
        elif value is None:
            value = 0
        else:
            raise TypeError("Value should be either a BaseGeometry or None, "
                            "got %s" % str(value))
        new = GeometryArray(self.data.copy(), base=base)
        new.data[idx] = value
        return new

    def fillna(self, value=None, method=None, limit=None):
        """ Fill NA/NaN values using the specified method.

        Parameters
        ----------
        value : scalar, array-like
            If a scalar value is passed it is used to fill all missing values.
            Alternatively, an array-like 'value' can be given. It's expected
            that the array-like have the same length as 'self'.
        method : {'backfill', 'bfill', 'pad', 'ffill', None}, default None
            Method to use for filling holes in reindexed Series
            pad / ffill: propagate last valid observation forward to next valid
            backfill / bfill: use NEXT valid observation to fill gap
        limit : int, default None
            If method is specified, this is the maximum number of consecutive
            NaN values to forward/backward fill. In other words, if there is
            a gap with more than this number of consecutive NaNs, it will only
            be partially filled. If method is not specified, this is the
            maximum number of entries along the entire axis where NaNs will be
            filled.

        Returns
        -------
        filled : ExtensionArray with NA/NaN filled
        """
        from pandas.api.types import is_array_like
        from pandas.util._validators import validate_fillna_kwargs
        from pandas.core.missing import pad_1d, backfill_1d

        value, method = validate_fillna_kwargs(value, method)

        mask = self.isna()

        if is_array_like(value):
            # if len(value) != len(self):
            #     raise ValueError("Length of 'value' does not match. Got ({}) "
            #                      " expected {}".format(len(value), len(self)))
            # value = value[mask]
            raise NotImplementedError

        if mask.any():
            if method is not None:
                func = pad_1d if method == 'pad' else backfill_1d
                new_values = func(self.astype(object), limit=limit,
                                  mask=mask)
                new_values = self._constructor_from_sequence(new_values)
            else:
                # fill with value
                new_values = self.fill(self.data == 0, value)
        else:
            new_values = self.copy()
        return new_values

    def __getstate__(self):
        return vectorized.serialize(self.data)

    def __setstate__(self, state):
        geoms = vectorized.deserialize(*state)
        self.data = geoms
        self.base = None

    # -------------------------------------------------------------------------
    # Geometry related methods
    # -------------------------------------------------------------------------

    def _binary_geo(self, other, op):
        """ Apply geometry-valued operation

        Supports:

        -   difference
        -   symmetric_difference
        -   intersection
        -   union

        Parameters
        ----------
        other: GeometryArray or single shapely BaseGeoemtry
        op: string
        """
        if isinstance(other, BaseGeometry):
            return GeometryArray(vectorized.binary_geo(op, self.data, other))
        elif isinstance(other, GeometryArray):
            if len(self) != len(other):
                msg = ("Lengths of inputs to not match.  Left: %d, Right: %d" %
                       (len(self), len(other)))
                raise ValueError(msg)
            return GeometryArray(
                vectorized.vector_binary_geo(op, self.data, other.data))
        else:
            raise NotImplementedError("type not known %s" % type(other))

    def _binop_predicate(self, other, op, extra=None):
        """ Apply boolean-valued operation

        Supports:

        -  contains
        -  disjoint
        -  intersects
        -  touches
        -  crosses
        -  within
        -  overlaps
        -  covers
        -  covered_by
        -  equals

        Parameters
        ----------
        other: GeometryArray or single shapely BaseGeoemtry
        op: string
        """
        if isinstance(other, BaseGeometry):
            if extra is not None:
                return vectorized.binary_predicate_with_arg(
                    op, self.data, other, extra)
            elif op in opposite_predicates:
                op2 = opposite_predicates[op]
                return vectorized.prepared_binary_predicate(
                    op2, self.data, other)
            else:
                return vectorized.binary_predicate(op, self.data, other)
        elif isinstance(other, GeometryArray):
            if len(self) != len(other):
                msg = ("Shapes of inputs to not match.  Left: %d, Right: %d" %
                       (len(self), len(other)))
                raise ValueError(msg)
            if extra is not None:
                return vectorized.vector_binary_predicate_with_arg(
                    op, self.data, other.data, extra)
            else:
                return vectorized.vector_binary_predicate(
                    op, self.data, other.data)
        else:
            raise NotImplementedError("type not known %s" % type(other))

    def covers(self, other):
        return self._binop_predicate(other, 'covers')

    def contains(self, other):
        return self._binop_predicate(other, 'contains')

    def crosses(self, other):
        return self._binop_predicate(other, 'crosses')

    def disjoint(self, other):
        return self._binop_predicate(other, 'disjoint')

    def equals(self, other):
        return self._binop_predicate(other, 'equals')

    def intersects(self, other):
        return self._binop_predicate(other, 'intersects')

    def overlaps(self, other):
        return self._binop_predicate(other, 'overlaps')

    def touches(self, other):
        return self._binop_predicate(other, 'touches')

    def within(self, other):
        return self._binop_predicate(other, 'within')

    def equals_exact(self, other, tolerance):
        return self._binop_predicate(other, 'equals_exact', tolerance)

    def is_valid(self):
        return vectorized.unary_predicate('is_valid', self.data)

    def is_empty(self):
        return vectorized.unary_predicate('is_empty', self.data)

    def is_simple(self):
        return vectorized.unary_predicate('is_simple', self.data)

    def is_ring(self):
        return vectorized.unary_predicate('is_ring', self.data)

    def has_z(self):
        return vectorized.unary_predicate('has_z', self.data)

    def is_closed(self):
        return vectorized.unary_predicate('is_closed', self.data)

    def _geo_unary_op(self, op):
        return GeometryArray(vectorized.geo_unary_op(op, self.data))

    def boundary(self):
        return self._geo_unary_op('boundary')

    def centroid(self):
        return self._geo_unary_op('centroid')

    def convex_hull(self):
        return self._geo_unary_op('convex_hull')

    def envelope(self):
        return self._geo_unary_op('envelope')

    def exterior(self):
        out = self._geo_unary_op('exterior')
        out.base = self  # exterior shares data with self
        return out

    def representative_point(self):
        return self._geo_unary_op('representative_point')

    def distance(self, other):
        if isinstance(other, GeometryArray):
            return vectorized.binary_vector_float(
                'distance', self.data, other.data)
        else:
            return vectorized.binary_float('distance', self.data, other)

    def project(self, other, normalized=False):
        op = 'project' if not normalized else 'project-normalized'
        if isinstance(other, GeometryArray):
            return vectorized.binary_vector_float_return(
                op, self.data, other.data)
        else:
            return vectorized.binary_float_return(op, self.data, other)

    def area(self):
        return vectorized.unary_vector_float('area', self.data)

    def length(self):
        return vectorized.unary_vector_float('length', self.data)

    def difference(self, other):
        return self._binary_geo(other, 'difference')

    def symmetric_difference(self, other):
        return self._binary_geo(other, 'symmetric_difference')

    def union(self, other):
        return self._binary_geo(other, 'union')

    def intersection(self, other):
        return self._binary_geo(other, 'intersection')

    def buffer(self, distance, resolution=16, cap_style=CAP_STYLE.round,
               join_style=JOIN_STYLE.round, mitre_limit=5.0):
        """ Buffer operation on array of GEOSGeometry objects """
        return GeometryArray(
            vectorized.buffer(self.data, distance, resolution, cap_style,
                              join_style, mitre_limit))

    def geom_type(self):
        """
        Types of the underlying Geometries

        Returns
        -------
        Pandas categorical with types for each geometry
        """
        x = vectorized.geom_type(self.data)

        import pandas as pd
        return pd.Categorical.from_codes(x, GEOMETRY_NAMES)

    def unary_union(self):
        """ Unary union.

        Returns a single shapely geometry
        """
        return vectorized.unary_union(self.data)

    def coords(self):
        return vectorized.coords(self.data)

    # -------------------------------------------------------------------------
    # for Series/ndarray like compat
    # -------------------------------------------------------------------------

    @property
    def shape(self):
        """ Shape of the ...

        For internal compatibility with numpy arrays.

        Returns
        -------
        shape : tuple
        """
        return tuple([len(self)])

    def to_dense(self):
        """Return my 'dense' representation

        For internal compatibility with numpy arrays.

        Returns
        -------
        dense : array
        """
        return to_shapely(self.data)

    def isna(self):
        """
        Boolean NumPy array indicating if each value is missing
        """
        return self.data == 0

    def unique(self):
        """Compute the ExtensionArray of unique values.

        Returns
        -------
        uniques : ExtensionArray
        """
        from pandas import factorize
        _, uniques = factorize(self)
        return uniques

    @property
    def nbytes(self):
        return self.data.nbytes

    # ExtensionArray specific

    @classmethod
    def _constructor_from_sequence(cls, scalars):
        """Construct a new ExtensionArray from a sequence of scalars.

        Parameters
        ----------
        scalars : Sequence
            Each element will be an instance of the scalar type for this
            array, ``cls.dtype.type``.
        Returns
        -------
        ExtensionArray
        """
        return from_shapely(scalars)

    @classmethod
    def _from_factorized(cls, values, original):
        """Reconstruct an ExtensionArray after factorization.

        Parameters
        ----------
        values : ndarray
            An integer ndarray with the factorized values.
        original : ExtensionArray
            The original ExtensionArray that factorize was called on.

        See Also
        --------
        pandas.factorize
        ExtensionArray.factorize
        """
        return from_wkb(values)

    def _values_for_argsort(self):
        # type: () -> ndarray
        """Return values for sorting.

        Returns
        -------
        ndarray
            The transformed values should maintain the ordering between values
            within the array.

        See Also
        --------
        ExtensionArray.argsort
        """
        # Note: this is used in `ExtensionArray.argsort`.
        raise TypeError("geometries are not orderable")

    def _values_for_factorize(self):
        # type: () -> Tuple[ndarray, Any]
        """Return an array and missing value suitable for factorization.

        Returns
        -------
        values : ndarray
            An array suitable for factoraization. This should maintain order
            and be a supported dtype (Float64, Int64, UInt64, String, Object).
            By default, the extension array is cast to object dtype.
        na_value : object
            The value in `values` to consider missing. This will be treated
            as NA in the factorization routines, so it will be coded as
            `na_sentinal` and not included in `uniques`. By default,
            ``np.nan`` is used.
        """
        vals = np.array([getattr(x, 'wkb', None) for x in self], dtype=object)
        return vals, None

    @classmethod
    def _concat_same_type(cls, to_concat):
        """
        Concatenate list of single blocks of the same type.
        """
        L = list(to_concat)
        x = np.concatenate([ga.data for ga in L])
        return GeometryArray(x, base=set(L))

    def __array__(self, dtype=None):
        """
        The numpy array interface.

        Returns
        -------
        values : numpy array
            A numpy array of either the specified dtype or,
            if dtype==None (default), the same dtype as
            categorical.categories.dtype
        """
        return to_shapely(self.data)
