import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from scripts.content_feedback import feedback
from src.recall.content_ann import ContentFaiss


class FusedIndexTest(unittest.TestCase):
    def test_only_logged_feedback_creates_pairwise_supervision(self):
        row = SimpleNamespace(
            items=np.array([2, 5, 9]), y=np.array([1, 0, 1]), positives=np.array([2, 9]), uid=1, seen=set()
        )
        counts = feedback([row], [np.array([2, 4, 5, 6, 0])])
        self.assertEqual((counts["clicked"], counts["exposed_nonclick"], counts["unknown"]), (1, 1, 2))
        self.assertEqual(
            (counts["mixed_requests"], counts["pairwise_preferences"], counts["all_clicks"]), (1, 1, 2)
        )

    def test_heat_coordinate_exclusions_tie_boundary_and_reload(self):
        # The tied text vectors extend past the initial overfetch budget.
        vectors = np.zeros((44, 2), np.float16)
        vectors[1:41, 0] = 2
        vectors[41] = [0, 1]
        prior = np.zeros(len(vectors), np.float32)
        prior[[39, 42]] = [0.25, 4]
        catalog = np.arange(1, len(vectors), dtype=np.int32)
        popular = np.array([42, 39], np.int32)
        rows = [
            SimpleNamespace(seen=set(range(1, 16))),
            SimpleNamespace(seen={42}),
            SimpleNamespace(seen=set()),
        ]
        q = np.array([[1, 0], [0, 0], [0, 1]], np.float32)
        cfg = {"nlist": 1, "nprobe": 1, "training_items": 43, "iterations": 3, "seed": 11}
        for kind in ("flat", "ivf"):
            index = ContentFaiss.build(kind, vectors, catalog, prior, 0.5, popular, cfg)
            self.assertNotIn(43, index.ids)
            with tempfile.TemporaryDirectory() as path:
                index.save(path)
                restored = ContentFaiss.load(path)
                result = restored.search(q, rows, 4)
            np.testing.assert_array_equal(result, [[42, 39, 16, 17], [39, 0, 0, 0], [42, 41, 39, 1]])


if __name__ == "__main__":
    unittest.main()
