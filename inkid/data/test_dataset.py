import os
import unittest

import inkid


class RegionSourceTestCase(unittest.TestCase):
    def test_multi_channel_ink_labels(self):
        test_file_path = os.path.join(
            os.path.dirname(inkid.__file__),
            "examples",
            "DummyTest.volpkg",
            "working",
            "DummyTestMultiChannelInkLabel.json",
        )
        region_source = inkid.data.RegionSource(test_file_path)
        self.assertTrue(region_source.is_ink(272, 148))
        self.assertFalse(region_source.is_ink(40, 40))


if __name__ == "__main__":
    unittest.main()
