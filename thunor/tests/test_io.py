import thunor.io
import unittest
import pkg_resources
import tempfile


def _assert_datasets_equal(d1, d2):
    assert d1.cell_lines == d2.cell_lines
    assert d1.drugs == d2.drugs
    assert d1.doses.size == d2.doses.size
    assert d1.controls.size == d2.controls.size
    assert d1.assays.size == d2.assays.size


class TestWithDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.filename = pkg_resources.resource_filename('thunor',
                                                       'testdata/VU001.h5')
        cls.dataset = thunor.io.read_hdf(cls.filename)

    def test_hdf5_read_write(self):
        with tempfile.NamedTemporaryFile(suffix='.h5') as tf:
            thunor.io.write_hdf(self.dataset, filename=tf.name)

            newdf = thunor.io.read_hdf(tf.name)

            _assert_datasets_equal(self.dataset, newdf)

    def test_vanderbilt_csv_read_write(self):
        with tempfile.NamedTemporaryFile(suffix='.csv') as tf:
            thunor.io.write_vanderbilt_hts(self.dataset, filename=tf.name)

            newdf = thunor.io.read_vanderbilt_hts(tf.name)

            _assert_datasets_equal(self.dataset, newdf)
