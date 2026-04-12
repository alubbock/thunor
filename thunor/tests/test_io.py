import thunor.io
import unittest
import importlib
import tempfile
import io
import pytest
import os
import pandas as pd


CSV_HEADER = 'cell.line,drug1.conc,drug1,upid,time,cell.count,well,drug1.units'


def _flatten_doses(doses):
    """Return doses as a flat, deterministically sorted DataFrame.

    Drops the 'dataset' index level when present (the CSV format has no
    dataset concept) and resets the remaining MultiIndex to columns.

    ``well_id`` is excluded: in HDF5 it is a numeric database primary key;
    after a CSV round-trip it is reconstructed as ``"<plate>__<well_num>"``.
    The pair ``(plate, well_num)`` is unique per row and stable across both
    formats, so we sort by that to give a deterministic, format-agnostic order.
    """
    d = doses.copy()
    if 'dataset' in d.index.names:
        d = d.reset_index('dataset', drop=True)
    d = d.reset_index().drop(columns='well_id')
    return d.sort_values(['plate', 'well_num']).reset_index(drop=True)


def _flatten_assays(assays, doses):
    """Return assays in a flat, deterministically sorted DataFrame.

    Joins assays with doses on ``well_id`` to replace the format-specific
    ``well_id`` with the stable ``(plate, well_num)`` pair, then sorts by
    ``(assay, plate, well_num, timepoint)`` for a format-agnostic comparison.
    """
    doses_flat = doses.reset_index()
    if 'dataset' in doses_flat.columns:
        doses_flat = doses_flat.drop(columns='dataset')
    doses_flat = doses_flat[['plate', 'well_num', 'well_id']]
    a = assays.reset_index().merge(doses_flat, on='well_id').drop(columns='well_id')
    return a.sort_values(['assay', 'plate', 'well_num', 'timepoint']).reset_index(
        drop=True
    )


def _flatten_controls(controls):
    """Return controls in a flat, deterministically sorted DataFrame.

    Drops the format-specific ``dataset`` and ``well_id`` index levels.
    ``well_num`` is a data column in controls and is stable across formats.
    """
    c = controls.copy()
    for lvl in ('dataset', 'well_id'):
        if lvl in c.index.names:
            c = c.reset_index(lvl, drop=True)
    c = c.reset_index()
    return c.sort_values(
        ['assay', 'cell_line', 'plate', 'well_num', 'timepoint']
    ).reset_index(drop=True)


def _assert_datasets_equal(d1, d2):
    assert d1.cell_lines == d2.cell_lines
    assert d1.drugs == d2.drugs
    # Use check_dtype=False / check_index_type=False so that comparisons
    # between StringDtype and object dtype (e.g. after an HDF5 round-trip
    # that normalises types) still pass.
    kw = dict(check_dtype=False, check_index_type=False)
    pd.testing.assert_frame_equal(
        _flatten_doses(d1.doses), _flatten_doses(d2.doses), **kw
    )
    pd.testing.assert_frame_equal(
        _flatten_assays(d1.assays, d1.doses), _flatten_assays(d2.assays, d2.doses), **kw
    )
    if d1.controls is None:
        assert d2.controls is None
    else:
        pd.testing.assert_frame_equal(
            _flatten_controls(d1.controls), _flatten_controls(d2.controls), **kw
        )


def _check_csv(csv_data):
    newdf = thunor.io.read_vanderbilt_hts(io.StringIO(csv_data), sep=',')
    assert isinstance(newdf, thunor.io.HtsPandas)
    return newdf


class TestWithDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ref = importlib.resources.files('thunor') / 'testdata/hts007.h5'
        with importlib.resources.as_file(ref) as filename:
            cls.hts007 = thunor.io.read_hdf(filename)

    def test_hdf5_read_write_file(self):
        with tempfile.NamedTemporaryFile(suffix='.h5') as tf:
            if os.name == 'nt':
                # Can't have two file handles on Windows
                tf.close()

            thunor.io.write_hdf(self.hts007, filename=tf.name)

            newdf = thunor.io.read_hdf(tf.name)

            _assert_datasets_equal(self.hts007, newdf)

    def test_hdf5_read_write_buffer(self):
        buf = io.BytesIO()

        thunor.io.write_hdf(self.hts007, filename=buf)

        buf.seek(0)

        newdf = thunor.io.read_hdf(buf)

        _assert_datasets_equal(self.hts007, newdf)

    def test_vanderbilt_csv_read_write(self):
        with tempfile.NamedTemporaryFile(suffix='.csv') as tf:
            if os.name == 'nt':
                # Can't have two file handles on Windows
                tf.close()

            thunor.io.write_vanderbilt_hts(self.hts007, filename=tf.name)

            newdf = thunor.io.read_vanderbilt_hts(tf.name)

            _assert_datasets_equal(self.hts007, newdf)


class TestCSV(unittest.TestCase):
    def test_csv_valid(self):
        _check_csv(CSV_HEADER + '\ncl1,0.00013,drug1,plate1,12,1234,A1,M')

    def test_csv_non_molar_unit(self):
        with pytest.raises(thunor.io.PlateFileParseException):
            _check_csv(CSV_HEADER + '\ncl1,0.00013,drug1,plate1,12,1234,A1,mM')

    def test_csv_well_out_of_range(self):
        with pytest.raises(thunor.io.PlateFileParseException):
            _check_csv(CSV_HEADER + '\ncl1,0.00013,drug1,plate1,12,1234,A99,M')

    def test_csv_non_numerical_dose(self):
        with pytest.raises(thunor.io.PlateFileParseException):
            _check_csv(CSV_HEADER + '\ncl1,dose,drug1,plate1,12,1234,A1,M')

    def test_csv_non_numerical_time(self):
        with pytest.raises(thunor.io.PlateFileParseException):
            _check_csv(CSV_HEADER + '\ncl1,0.00013,drug1,plate1,t,1234,A99,M')

    def test_csv_non_numerical_cell_count(self):
        with pytest.raises(thunor.io.PlateFileParseException):
            _check_csv(CSV_HEADER + '\ncl1,0.00013,drug1,plate1,12,X,A1,M')

    def test_csv_cell_line_missing(self):
        with pytest.raises(thunor.io.PlateFileParseException):
            _check_csv(
                'drug1.conc,drug1,upid,time,cell.count,well,drug1.units\n'
                '0.00013,drug1,plate1,12,1234,A1,M'
            )

    def test_csv_drug1_missing(self):
        with pytest.raises(thunor.io.PlateFileParseException):
            _check_csv(
                'cell.line,drug1.conc,upid,time,cell.count,well,drug1.units\n'
                'cl1,0.00013,plate1,12,1234,A1,M'
            )

    def test_csv_only_control_well(self):
        _check_csv(CSV_HEADER + '\ncl1,0,,plate1,12,1234,A1,M')

    def test_csv_only_layout_data(self):
        _check_csv('upid,well,time,cell.count\nplate1,A01,0,123')


class TestCSVTwoDrugs(unittest.TestCase):
    def test_csv_two_drugs(self):
        _check_csv(
            CSV_HEADER + ',drug2,drug2.units,drug2.conc'
            '\ncl1,0.00013,drug1,plate1,12,1234,A1,M,drug2,M,0.00010'
        )

    def test_csv_two_drugs_drug_conc_missing(self):
        with pytest.raises(thunor.io.PlateFileParseException):
            _check_csv(
                CSV_HEADER + ',drug2,drug2.units'
                '\ncl1,0.00013,drug1,plate1,12,1234,A1,M,drug2,M'
            )

    def test_csv_two_drugs_drug2_blank_conc_specified(self):
        with pytest.raises(thunor.io.PlateFileParseException):
            _check_csv(
                CSV_HEADER + ',drug2,drug2.units,drug2.conc'
                '\ncl1,0.00013,drug1,plate1,12,1234,A1,M,,M,0.00010'
            )

    def test_csv_two_drugs_drug2_blank(self):
        csv = _check_csv(
            CSV_HEADER + ',drug2,drug2.units,drug2.conc'
            '\ncl1,0.00013,drug1,plate1,12,1234,A1,M,,M,0'
        )
        # Second drug should get dropped, since it's empty
        assert len(csv.doses.index.get_level_values('drug')[0]) == 1
        assert len(csv.doses.index.get_level_values('dose')[0]) == 1


def test_read_incucyte():
    ref = importlib.resources.files('thunor') / 'testdata/test_incucyte_minimal.txt'
    with importlib.resources.as_file(ref) as filename:
        thunor.io.read_incucyte(filename)


class TestPlateMap(unittest.TestCase):
    def _plate_map(self, expected_width, expected_height):
        pm = thunor.io.PlateMap(width=expected_width, height=expected_height)
        assert pm.width == expected_width
        assert pm.height == expected_height
        assert pm.num_wells == (expected_width * expected_height)
        for i in range(pm.num_wells):
            assert pm.well_name_to_id(pm.well_id_to_name(i)) == i

    def test_plate_map_96(self):
        self._plate_map(12, 8)

    def test_plate_map_384(self):
        self._plate_map(24, 18)

    def test_plate_map_1536(self):
        self._plate_map(48, 32)
