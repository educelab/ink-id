import os
import shutil
import unittest

from inkid.ops import dummy_volpkg_path

from . import generate_subvolumes


test_output_dir = 'test_output'


class TestScriptsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        os.makedirs(test_output_dir)

    def test_generate_subvolumes(self) -> None:
        generate_subvolumes.main([
            '--input-set', os.path.join(dummy_volpkg_path(), 'working/DummyTest_grid1x2.txt'),
            '--output', os.path.join(test_output_dir, 'subvolumes'),
        ])

    def tearDown(self) -> None:
        shutil.rmtree(test_output_dir)


if __name__ == '__main__':
    unittest.main()
