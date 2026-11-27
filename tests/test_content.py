import unittest
from types import SimpleNamespace

import numpy as np

from src.recall.content import ContentIndex, combine


class ContentSearchTests(unittest.TestCase):
    def test_exact_fusion_matches_exhaustive_with_ties_exclusions_and_missing_text(self):
        matrix = np.array([[0, 0], [1, 0], [1, 0], [0, 1], [0, 0], [-1, 0], [0.5, 0.5], [0, -1]], np.float16)
        ids = np.arange(1, len(matrix), dtype=np.int32)
        prior = np.array([0, 0, 0, 0.7, 2, 0, 0.2, 0], np.float32)
        rows = [SimpleNamespace(seen={1}), SimpleNamespace(seen={3, 7}), SimpleNamespace(seen=set())]
        queries = np.array([[1, 0], [0, -1], [0, 0]], np.float32)
        full = ContentIndex(matrix, ids, "cpu")
        warm = ContentIndex(matrix, ids[prior[ids] > 0], "cpu")
        for k in (1, 3, 8):
            semantic = full.search(queries, rows, k)[0.0]
            warm_top = warm.search(queries, rows, k, (0.8,), prior)[0.8]
            combined = combine(semantic, warm_top, prior, 0.8)[0]
            for j in range(2):
                doc = matrix[ids].astype(np.float32)
                doc /= np.maximum(np.linalg.norm(doc, axis=1, keepdims=True), 1e-8)
                for weight, actual in ((0, semantic[0]), (0.8, combined)):
                    score = doc @ queries[j] + weight * prior[ids]
                    expected = [
                        int(ids[i])
                        for i in np.lexsort((ids, -score))
                        if ids[i] not in rows[j].seen and (doc[i].any() or weight * prior[ids[i]] > 0)
                    ][:k]
                    self.assertEqual(actual[j].tolist(), expected + [0] * (k - len(expected)))
            self.assertFalse(combined[2].any())
