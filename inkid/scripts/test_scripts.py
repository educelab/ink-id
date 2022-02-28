import os
from pathlib import Path
import shutil
import unittest

from inkid.util import dummy_volpkg_path

from . import generate_subvolumes
from . import train_and_predict


test_output_dir = Path("test_output")


class TestScriptsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        os.makedirs(test_output_dir)

    def test_generate_subvolumes(self) -> None:
        generate_subvolumes.main(
            [
                "--input-set",
                os.path.join(dummy_volpkg_path(), "working", "DummyTest_grid1x2.txt"),
                "--output",
                str(test_output_dir / "subvolumes"),
            ]
        )

    def test_train_and_predict(self) -> None:
        train_and_predict.main(
            [
                "--training-set",
                str(Path(dummy_volpkg_path()) / "working" / "DummyTest_grid1x2.txt"),
                "--output",
                str(test_output_dir / "train_and_predict"),
                "--cross-validate-on",
                "0",
                "--training-max-samples",
                "10",
                "--final-prediction-on-all",
                "--dataloaders-num-workers",
                "0",
            ]
        )

    def tearDown(self) -> None:
        shutil.rmtree(test_output_dir)


if __name__ == "__main__":
    unittest.main()
