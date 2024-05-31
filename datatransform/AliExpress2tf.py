from transform import DataTransform
from datetime import datetime, date
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
parser = argparse.ArgumentParser(description='Transfrom original data to TFRecord')

parser.add_argument('--label', type=str, default="Label")
parser.add_argument("--store_stat", action="store_true", default=True)
parser.add_argument("--threshold", type=int, default=2)
parser.add_argument("--dataset1", type=Path, default='../data/nl/nl_data.csv')
parser.add_argument("--dataset2", type=Path, default='../data/fr/fr_data.csv')
parser.add_argument("--stats", type=Path, default='../data/AliExpress-1/stats_2')
parser.add_argument("--record", type=Path, default='../data/AliExpress-1/threshold_2')
parser.add_argument("--ratio", nargs='+', type=float, default=[0.7, 0.15, 0.15])

args = parser.parse_args()


class AilExpressTransform(DataTransform):
    def __init__(self, dataset_path1, dataset_path2, path, stats_path, min_threshold, label_index, ratio, store_stat=False, seed=2021):
        super(AilExpressTransform, self).__init__(dataset_path1, dataset_path2, stats_path, store_stat=store_stat, seed=seed)
        self.threshold = min_threshold
        self.label = label_index
        self.split = ratio
        self.path = path
        self.stats_path = stats_path
        self.name = "Id,U1,U2,U3,U4,U5,U6,U7,U8,U9,U10,U11,U12,U13,U14,U15,U16,U17,U18,U19,U20,U21,U22,U23,U24,U25,U26,U27,U28,U29,U30,U31,U32,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,I13,I14,I15,I16,I17,I18,I19,I20,I21,I22,I23,I24,I25,I26,I27,I28,I29,I30,I31,I32,I33,I34,I35,I36,I37,I38,I39,I40,I41,I42,I43,I44,I45,I46,I47,Label".split(
            ",")

    def process(self):
        self._read(name=self.name, header=None, sep=",", label_index=self.label)
        if self.store_stat:
            self.generate_and_filter(threshold=self.threshold, label_index=self.label)
        tr1, te1, val1 = self.random_split(self.domain1_data, ratio=self.split)
        self.transform_tfrecord(tr1, self.path, "train"+str(0), label_index=self.label, domain_id=0)
        self.transform_tfrecord(te1, self.path, "test"+str(0), label_index=self.label, domain_id=0)
        self.transform_tfrecord(val1, self.path, "validation"+str(0), label_index=self.label, domain_id=0)

        tr2, te2, val2 = self.random_split(self.domain2_data, ratio=self.split)
        self.transform_tfrecord(tr2, self.path, "train" + str(1), label_index=self.label, domain_id=1)
        self.transform_tfrecord(te2, self.path, "test" + str(1), label_index=self.label, domain_id=1)
        self.transform_tfrecord(val2, self.path, "validation" + str(1), label_index=self.label, domain_id=1)

    def _process_x(self):
        # print(self.data[self.data["Label"] == 1].shape)
        # print(self.domain1_data[self.domain1_data["Label"] == 1].shape)
        # print(self.domain2_data[self.domain2_data["Label"] == 1].shape)
        pass

    def _process_y(self):
        self.data = self.data.drop("Id", axis=1)
        self.domain1_data = self.domain1_data.drop("Id", axis=1)
        self.domain2_data = self.domain2_data.drop("Id", axis=1)
        self.data["Label"] = self.data["Label"].apply(lambda x: 0 if x == 0 else 1)
        self.domain1_data["Label"] = self.domain1_data["Label"].apply(lambda x: 0 if x == 0 else 1)
        self.domain2_data["Label"] = self.domain2_data["Label"].apply(lambda x: 0 if x == 0 else 1)

if __name__ == "__main__":
    tranformer = AilExpressTransform(args.dataset1, args.dataset2, args.record, args.stats,
                                 args.threshold, args.label,
                                 args.ratio, store_stat=args.store_stat)
    tranformer.process()
