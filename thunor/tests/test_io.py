import thunor.io
import unittest
import pkg_resources
import tempfile


def _assert_datasets_equal(d1, d2):
    assert d1.cell_lines == d2.cell_lines
    assert d1.drugs == d2.drugs
    assert d1.doses.shape[0] == d2.doses.shape[0]
    assert d1.controls.shape[0] == d2.controls.shape[0]
    assert d1.assays.shape[0] == d2.assays.shape[0]


class TestWithDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        filename = pkg_resources.resource_filename('thunor',
                                                   'testdata/hts007.h5')
        cls.hts007 = thunor.io.read_hdf(filename)

    def test_hdf5_read_write(self):
        with tempfile.NamedTemporaryFile(suffix='.h5') as tf:
            thunor.io.write_hdf(self.hts007, filename=tf.name)

            newdf = thunor.io.read_hdf(tf.name)

            _assert_datasets_equal(self.hts007, newdf)

    def test_vanderbilt_csv_read_write(self):
        with tempfile.NamedTemporaryFile(suffix='.csv') as tf:
            thunor.io.write_vanderbilt_hts(self.hts007, filename=tf.name)

            newdf = thunor.io.read_vanderbilt_hts(tf.name)

            _assert_datasets_equal(self.hts007, newdf)
