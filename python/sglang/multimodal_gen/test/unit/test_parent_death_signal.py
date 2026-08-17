import os
import unittest
from unittest.mock import patch

from sglang.multimodal_gen import utils


class TestParentDeathSignal(unittest.TestCase):
    def test_explicit_disable_skips_prctl(self):
        with (
            patch.dict(os.environ, {"SGLANG_DISABLE_PDEATHSIG": "true"}),
            patch.object(utils.ctypes, "CDLL") as cdll,
        ):
            utils.kill_itself_when_parent_died()

        cdll.assert_not_called()


if __name__ == "__main__":
    unittest.main()
