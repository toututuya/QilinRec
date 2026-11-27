import unittest
from types import SimpleNamespace

import numpy as np
import torch

from src.data import Request, attach_histories, split_data
from src.features import rank_groups
from src.metrics import exposure_metrics, retrieval_metrics
from src.rank import CTRModel
from src.recall.classical import Popularity, build_itemcf, cf_candidates, cf_scores
from src.recall.fusion import fuse
from src.recall.index import CatalogIndex
from src.recall.tower import negative_mask
from src.runtime import STAGES


def row(rid,start,end,items=(1,2),y=(1,0),recent=()):
    return Request(rid,1,start,end,np.array(items,np.int32),np.array(y,np.uint8),recent)


class Contracts(unittest.TestCase):
    def test_strict_boundaries(self):
        train = [row(j,j,j+1) for j in range(30)] + [row(60,3,99)]
        test = [row(90,22,23)]
        data,info = split_data(train,test,dict(split_quantiles=[.3,.5,.8],stage_limits=[]))
        for earlier,later in zip(STAGES,STAGES[1:]):
            self.assertLess(max(r.end for r in data[earlier]),min(r.start for r in data[later]))
        self.assertGreater(info["purged_at_test"],0)
        self.assertEqual(data["test"],test)

    def test_history_requires_completed_prior_request(self):
        a = row(0,0,10,items=(4,5),recent=(9,))
        equal = row(1,10,11)
        future = row(2,12,15)
        evaluation = row(3,30,31)
        data = dict(fit=[a],tower_val=[equal],rank_train=[],rank_val=[future],test=[evaluation])
        attach_histories(data,20)
        self.assertEqual(equal.hist,())
        self.assertEqual(set(future.hist),{4,9})
        self.assertEqual(set(evaluation.hist),{4,9})

    def test_unretrieved_and_repeated_positives(self):
        r = row(0,2,3,(2,3,4),(1,1,0))
        r.seen = frozenset({3})
        m = retrieval_metrics([[2,4]],[r],20,(10,))
        self.assertEqual(m["recall@10"],.5)
        self.assertEqual(m["retained@10"],.5)
        self.assertLess(m["ndcg@10"],1)
        self.assertEqual(m["coverage@10"],.1)

    def test_negative_mask_for_duplicates_history_and_other_positives(self):
        a = row(1,0,0,(4,7),(1,1))
        a.seen = {8}
        b = row(2,0,0,(4,),(1,))
        c = row(3,0,0,(7,),(1,))
        d = row(4,0,0,(8,),(1,))
        m = negative_mask(np.array([4,4,7,8]),[a,b,c,d])
        self.assertEqual(m[0].tolist(),[False,True,True,True])
        self.assertFalse(m.diagonal().any())

    def test_itemcf_and_pop_filtering(self):
        r = row(0,2,4,(3,2),(1,1),recent=(4,))
        graph = build_itemcf([r],8,5,dict(cf_user_cap=80,cf_neighbors=5,cf_half_life_hours=5))
        mask = np.ones(8,bool)
        self.assertEqual({i for i,_ in cf_candidates(cf_scores(graph,(3,)),{3,4},mask,8)},{2})
        p = Popularity([r],8,5,1)
        self.assertLess(p.multiplier(3605),1)
        with self.assertRaises(ValueError):
            p.multiplier(4)

    def test_catalog_index_and_fusion(self):
        import faiss
        v = np.eye(30,dtype=np.float32)
        flat = faiss.IndexFlatIP(30)
        flat.add(v)
        r = row(0,0,1)
        r.seen = frozenset(range(1,26))
        top = CatalogIndex(flat,np.arange(1,31)).search(v[:1],[r],5)
        self.assertEqual({i for i,_ in top[0]},set(range(26,31)))
        self.assertEqual(fuse([[(4,.1),(3,.01)],[(3,.8)]],60,2),[3,4])

    def test_masked_pooling(self):
        cfg={"id_dim":3}
        for name in ["mlp","din"]:
            net=CTRModel(name,np.zeros((3,3),np.float32),np.arange(3,dtype=np.int32),2,cfg,2)
            target,hist=torch.rand(2,3),torch.rand(2,2,3)
            result=net.pool(target,hist,torch.tensor([[False,False],[True,False]]))
            torch.testing.assert_close(result[0],torch.zeros(3))
            torch.testing.assert_close(result[1],hist[1,0])

    def test_exposure_alignment_and_ties(self):
        r=row(1,0,1,(3,2),(0,1))
        s=np.array([.1,.1])
        groups,_=rank_groups(s,r.items,np.array([0,2]))
        self.assertEqual(groups,[[2,3]])
        m=exposure_metrics(s,[r],r.items,[0,2],True)
        self.assertEqual(m["auc"],.5)
        self.assertEqual(m["gauc"],.5)
        with self.assertRaises(ValueError):
            exposure_metrics(s,[r],r.items[::-1],[0,2])


if __name__=="__main__":
    unittest.main()
