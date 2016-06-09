from __future__ import absolute_import

import tempfile
import shutil

import numpy as np

from pandas.util.testing import assert_frame_equal

from shapely.geometry import Point, Polygon

from geopandas import GeoDataFrame, GeoSeries, read_file
from geopandas.tests.util import unittest, download_nybb, assert_geoseries_equal
from geopandas import overlay


class TestOverlayNYBB(unittest.TestCase):

    def setUp(self):
        N = 10

        nybb_filename, nybb_zip_path = download_nybb()

        self.polydf = read_file(nybb_zip_path, vfs='zip://' + nybb_filename)
        self.tempdir = tempfile.mkdtemp()
        self.crs = {'init': 'epsg:4326'}
        b = [int(x) for x in self.polydf.total_bounds]
        self.polydf2 = GeoDataFrame([
            {'geometry' : Point(x, y).buffer(10000), 'value1': x + y, 'value2': x - y}
            for x, y in zip(range(b[0], b[2], int((b[2]-b[0])/N)),
                            range(b[1], b[3], int((b[3]-b[1])/N)))], crs=self.crs)
        self.pointdf = GeoDataFrame([
            {'geometry' : Point(x, y), 'value1': x + y, 'value2': x - y}
            for x, y in zip(range(b[0], b[2], int((b[2]-b[0])/N)),
                            range(b[1], b[3], int((b[3]-b[1])/N)))], crs=self.crs)

        # TODO this appears to be necessary;
        # why is the sindex not generated automatically?
        self.polydf2._generate_sindex()

        self.union_shape = (180, 7)

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_union(self):
        df = overlay(self.polydf, self.polydf2, how="union")
        self.assertTrue(type(df) is GeoDataFrame)
        self.assertEquals(df.shape, self.union_shape)
        self.assertTrue('value1' in df.columns and 'Shape_Area' in df.columns)

    def test_union_no_index(self):
        # explicitly ignore indicies
        dfB = overlay(self.polydf, self.polydf2, how="union", use_sindex=False)
        self.assertEquals(dfB.shape, self.union_shape)

        # remove indicies from df
        self.polydf._sindex = None
        self.polydf2._sindex = None
        dfC = overlay(self.polydf, self.polydf2, how="union")
        self.assertEquals(dfC.shape, self.union_shape)

    def test_intersection(self):
        df = overlay(self.polydf, self.polydf2, how="intersection")
        self.assertIsNotNone(df['BoroName'][0])
        self.assertEquals(df.shape, (68, 7))

    def test_identity(self):
        df = overlay(self.polydf, self.polydf2, how="identity")
        self.assertEquals(df.shape, (154, 7))

    def test_symmetric_difference(self):
        df = overlay(self.polydf, self.polydf2, how="symmetric_difference")
        self.assertEquals(df.shape, (122, 7))

    def test_difference(self):
        df = overlay(self.polydf, self.polydf2, how="difference")
        self.assertEquals(df.shape, (86, 7))

    def test_bad_how(self):
        self.assertRaises(ValueError,
                          overlay, self.polydf, self.polydf, how="spandex")

    def test_nonpoly(self):
        self.assertRaises(TypeError,
                          overlay, self.pointdf, self.polydf, how="union")

    def test_duplicate_column_name(self):
        polydf2r = self.polydf2.rename(columns={'value2': 'Shape_Area'})
        df = overlay(self.polydf, polydf2r, how="union")
        self.assertTrue('Shape_Area_2' in df.columns and 'Shape_Area' in df.columns)

    def test_geometry_not_named_geometry(self):
        # Issue #306
        # Add points and flip names
        polydf3 = self.polydf.copy()
        polydf3 = polydf3.rename(columns={'geometry':'polygons'})
        polydf3 = polydf3.set_geometry('polygons')
        polydf3['geometry'] = self.pointdf.geometry.loc[0:4]
        self.assertTrue(polydf3.geometry.name == 'polygons')

        df = overlay(polydf3, self.polydf2, how="union")
        self.assertTrue(type(df) is GeoDataFrame)

        df2 = overlay(self.polydf, self.polydf2, how="union")
        self.assertTrue(df.geom_almost_equals(df2).all())

    def test_geoseries_warning(self):
        # Issue #305

        def f():
            overlay(self.polydf, self.polydf2.geometry, how="union")
        self.assertRaises(NotImplementedError, f)


class TestOverlay(unittest.TestCase):

    use_sindex = True

    def setUp(self):

        s1 = GeoSeries([Polygon([(0,0), (2,0), (2,2), (0,2)]),
                        Polygon([(2,2), (4,2), (4,4), (2,4)])])
        s2 = GeoSeries([Polygon([(1,1), (3,1), (3,3), (1,3)]),
                        Polygon([(3,3), (5,3), (5,5), (3,5)])])

        self.df1 = GeoDataFrame({'geometry': s1, 'col1':[1,2]})
        self.df2 = GeoDataFrame({'geometry': s2, 'col2':[1,2]})

        self.result  = GeoDataFrame(
            {'col1': [1, 1, np.nan, np.nan, 2, 2, 2, np.nan, 2],
             'col2': [np.nan, 1, 1, 1, np.nan, 1, np.nan, 2, 2],
             'geometry': [Polygon([(2, 1), (2, 0), (0, 0), (0, 2), (1, 2), (1, 1), (2, 1)]),
                          Polygon([(2, 1), (1, 1), (1, 2), (2, 2), (2, 1)]),
                          Polygon([(2, 1), (2, 2), (3, 2), (3, 1), (2, 1)]),
                          Polygon([(2, 2), (1, 2), (1, 3), (2, 3), (2, 2)]),
                          Polygon([(3, 2), (3, 3), (4, 3), (4, 2), (3, 2)]),
                          Polygon([(3, 3), (3, 2), (2, 2), (2, 3), (3, 3)]),
                          Polygon([(3, 3), (2, 3), (2, 4), (3, 4), (3, 3)]),
                          Polygon([(4, 3), (4, 4), (3, 4), (3, 5), (5, 5), (5, 3), (4, 3)]),
                          Polygon([(3, 4), (4, 4), (4, 3), (3, 3), (3, 4)])]
            })

    def test_union(self):
        res = overlay(self.df1, self.df2, how='union',
                      use_sindex=self.use_sindex)
        exp = self.result
        assert_frame_equal(res, exp)
        assert_geoseries_equal(res.geometry, exp.geometry)

    def test_intersection(self):
        res = overlay(self.df1, self.df2, how='intersection',
                      use_sindex=self.use_sindex)
        exp = self.result.dropna(subset=['col1', 'col2'], how='any')
        exp = exp.reset_index(drop=True)
        exp[['col1', 'col2']] = exp[['col1', 'col2']].astype('int64')
        print(exp)
        print(res)
        print(self.result.geometry[7])
        print(self.result.geometry[7].representative_point())
        print(list(self.df1.sindex.intersection(self.result.geometry[7].bounds)))
        cent = self.result.geometry[7].representative_point()
        print(cent.intersects(self.df1.geometry[1]))
        assert_frame_equal(res, exp)
        assert_geoseries_equal(res.geometry, exp.geometry)

    def test_symdiff(self):
        res = overlay(self.df1, self.df2, how='symmetric_difference',
                      use_sindex=self.use_sindex)
        exp = self.result[self.result[['col1', 'col2']].isnull().sum(1) == 1]
        exp = exp.reset_index(drop=True)
        assert_frame_equal(res, exp)
        assert_geoseries_equal(res.geometry, exp.geometry)

    def test_difference(self):
        res = overlay(self.df1, self.df2, how='difference',
                      use_sindex=self.use_sindex)
        exp = self.result.loc[[0, 4, 6]]
        exp = exp.reset_index(drop=True)
        exp['col1'] = exp['col1'].astype('int64')
        exp['col2'] = np.array([None, None, None], dtype='O')
        print(exp)
        print(res)
        assert_frame_equal(res, exp)
        assert_geoseries_equal(res.geometry, exp.geometry)

    def test_identity(self):
        res = overlay(self.df1, self.df2, how='identity',
                      use_sindex=self.use_sindex)
        exp = self.result.dropna(subset=['col1'])
        exp = exp.reset_index(drop=True)
        exp['col1'] = exp['col1'].astype('int64')
        print(exp)
        print(res)
        assert_frame_equal(res, exp)
        assert_geoseries_equal(res.geometry, exp.geometry)

    def test_nondefault_index(self):
        df1 = self.df1.copy()
        df1.index = ['row1', 'row2']
        res = overlay(df1, self.df2, how='intersection',
                      use_sindex=self.use_sindex)
        exp = self.result.dropna(subset=['col1', 'col2'], how='any')
        exp = exp.reset_index(drop=True)
        exp[['col1', 'col2']] = exp[['col1', 'col2']].astype('int64')
        assert_frame_equal(res, exp)
        assert_geoseries_equal(res.geometry, exp.geometry)


class TestOverlayNoSIndex(TestOverlay):

    use_sindex = False
