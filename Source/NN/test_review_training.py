"""Reviewed labels must not duplicate frames or cross label boundaries."""
import unittest
import tempfile
import csv
import io
import json
import zipfile
import hashlib
from pathlib import Path
from unittest.mock import patch

import numpy as np

import train_physics_fall as trainer
from train_physics_fall import label_spans_for, reviewed_for, class_windows, sample_training, assign_splits


class ReviewTrainingTests(unittest.TestCase):
    def test_missing_class_recall_is_undefined(self):
        result=trainer.metrics(np.array([0,2]),np.array([[1,0,0],[0,0,1]]))
        self.assertEqual(result['recall'],[1,None,1])
        self.assertEqual(result['balanced_accuracy'],1)

    def review(self, **kwargs):
        return dict(sha256='abc', label='运动', ranges=[], excluded=False, **kwargs)

    def test_same_label_range_merges(self):
        review=self.review()
        review['ranges']=[dict(start=5,end=11,label='运动')]
        self.assertEqual(label_spans_for(30,0,review,'abc'),[(0,30,0)])

    def test_override_and_unlabeled_spans(self):
        review=self.review()
        review['ranges']=[dict(start=5,end=11,label='跌倒')]
        self.assertEqual(label_spans_for(30,0,review,'abc'),[(0,5,0),(5,12,1),(12,30,0)])
        review['label']=None
        self.assertEqual(label_spans_for(30,0,review,'abc'),[(5,12,1)])
        review['ranges']=[]
        self.assertEqual(label_spans_for(30,0,review,'abc'),[])

    def test_excluded_and_stale(self):
        review=self.review();review['excluded']=True
        self.assertEqual(label_spans_for(30,0,review,'abc'),[])
        with self.assertRaises(ValueError):label_spans_for(30,0,review,'changed')

    def test_malformed_ranges(self):
        for ranges in [[dict(start=1,end=31,label='运动')],
                       [dict(start=0,end=4,label='运动'),dict(start=4,end=5,label='跌倒')]]:
            review=self.review();review['ranges']=ranges
            with self.assertRaises(ValueError):label_spans_for(30,0,review,'abc')

    def test_sources_are_case_insensitive_on_windows(self):
        item=dict(source=r'F:\Data\motion.zip::clip/frames.csv')
        self.assertIs(reviewed_for('f:/data/motion.zip::clip/frames.csv',{'x':item}),item)

    def test_windows_never_cross_label_or_time_boundaries(self):
        raw=np.full((40,64),1000,np.int16)
        r=dict(raw=raw,split='train',label_spans=[(0,20,0),(20,40,1)],segments=[(0,12),(12,40)])
        clips=class_windows([r],'train')
        self.assertEqual([len(c) for c in clips[0]],[12,8])
        self.assertEqual([len(c) for c in clips[1]],[20])
        self.assertEqual(sum(len(c) for v in clips.values() for c in v),40)

    def test_short_fall_does_not_shorten_other_classes(self):
        train={k:[np.full((n,64),1000,np.int16)] for k,n in enumerate([24,8,24])}
        clips,labels=sample_training(train,np.random.default_rng(2),'independent')
        for clip,label in zip(clips,labels):self.assertEqual(len(clip),8 if label==1 else 24)

    def test_copied_sources_cannot_leak_across_splits(self):
        records=[dict(label_spans=[(0,30,0)],segments=[(0,30)],split=s,sha256='same') for s in ['train','validation']]
        assign_splits(records)
        self.assertEqual([r['split'] for r in records],['validation','validation'])

    def test_loader_applies_review_without_changing_later_member_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); folder=root/'NN'/'Data'/'运动';folder.mkdir(parents=True)
            archive=folder/'sample.zip'
            def body(value):
                out=io.StringIO();writer=csv.writer(out)
                writer.writerow(['unix_ms']+[f'zone_{i:02d}_mm' for i in range(64)])
                for i in range(30):writer.writerow([1000+i*100]+[value]*64)
                return out.getvalue().encode('utf-8')
            first,second=body(1000),body(1200)
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr('a/frames.csv',first);z.writestr('b/frames.csv',second)
            review_path=root/'annotations.json'
            annotation=dict(source=str(archive.resolve())+'::a/frames.csv',sha256=hashlib.sha256(first).hexdigest(),label='跌倒',ranges=[],excluded=False)
            review_path.write_text(json.dumps(dict(records={'a':annotation})),encoding='utf-8')
            before=archive.read_bytes()
            with patch.object(trainer,'ROOT',root):records=trainer.load(review_path)
            self.assertEqual([r['label'] for r in records],[1,0])
            self.assertEqual(archive.read_bytes(),before)


if __name__=='__main__':unittest.main()
