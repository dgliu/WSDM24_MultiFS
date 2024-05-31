import tensorflow as tf
import argparse
import glob
import joblib
import pickle
import numpy as np
import torch
import time
from pathlib import Path

np.random.seed(2022)
use_columns = [
    '101',
    '121',
    '122',
    '124',
    '125',
    '126',
    '127',
    '128',
    '129',
    '205',
    '206',
    '207',
    '216',
    '508',
    '509',
    '702',
    '853',
    '301'
 ]
def feature_example(feature, label, domain):
    feature_des = {
        'feature': tf.train.Feature(int64_list=tf.train.Int64List(value=feature)),
        'label': tf.train.Feature(float_list=tf.train.FloatList(value=[label])),
        'domain': tf.train.Feature(float_list=tf.train.FloatList(value=[domain])),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature_des))
    return example_proto

def write_tfrecord(path, data_type, domain, data):
    print("write doamin {} tfrecord".format(domain))
    filename = path + data_type + str(domain) + '.tfrecord'
    writer = tf.io.TFRecordWriter(filename)
    c = 0
    for feature, label, domain in data:
        ex = feature_example(feature, label, domain)
        writer.write(ex.SerializeToString())
        c += 1
        if c % 100000 == 0:
            print(c)
    writer.close()

def write_tfrecord2(path, data_type, data, domain_dict):
    filename = path + data_type + '.tfrecord'
    writer = tf.io.TFRecordWriter(filename)
    c = 0
    for feature, label, domain in data:
        feature = np.append(feature, [domain_dict[int(domain)]])
        ex = feature_example(feature, label, domain)
        writer.write(ex.SerializeToString())
        c += 1
        if c % 100000 == 0:
            print(c)
    writer.close()

def gen_defaults(source_path, target_path):
    feat_map = joblib.load(source_path)
    defaults = {}
    for k in use_columns[:17]:
        # print(len(feat_map[k]))
        defaults.update({k: len(feat_map[k])})
    with open(target_path, 'wb') as fi:
        pickle.dump(defaults, fi)
    print('write defaults end')

def gen_defaults2(source_path, target_path, scenes):
    defaults = joblib.load(source_path)
    defaults.update({use_columns[-1]: len(scenes)})
    with open(target_path, 'wb') as fi:
        pickle.dump(defaults, fi)
    print('write defaults end')

class Ali_CCP_Source(object):
    def __init__(self, tfrecord_path):
        self.SAMPLES = 1
        self.FIELDS = 18
        self.tfrecord_path = tfrecord_path
        self.description = {
            "label": tf.io.FixedLenFeature([self.SAMPLES], tf.float32),
            "feat_ids": tf.io.FixedLenFeature([self.FIELDS], tf.int64),
            "feat_vals": tf.io.FixedLenFeature([self.FIELDS], tf.float32),
        }
    def get_data(self, data_type, batch_size=1):
        @tf.autograph.experimental.do_not_convert
        def read_data(raw_rec):
            example = tf.io.parse_single_example(raw_rec, self.description)
            return example['feat_ids'], example['feat_vals'], example['label']

        files = glob.glob(self.tfrecord_path + '/' + "{}*".format(data_type))
        ds = tf.data.TFRecordDataset(files).map(read_data, num_parallel_calls=tf.data.experimental.AUTOTUNE). \
            batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        for fea_ids, fea_vals, label in ds:
            yield fea_ids, fea_vals, label

    def transform_data(self, data_type, scenes = [0, 1]):
        @tf.autograph.experimental.do_not_convert
        def read_data(raw_rec):
            example = tf.io.parse_single_example(raw_rec, self.description)
            return example['feat_ids'], example['feat_vals'], example['label']

        files = glob.glob(self.tfrecord_path + '/' + "{}*".format(data_type))
        ds = tf.data.TFRecordDataset(files).map(read_data, num_parallel_calls=tf.data.experimental.AUTOTUNE). \
            batch(4096).prefetch(tf.data.experimental.AUTOTUNE)
        domain0_data = []
        domain1_data = []
        domain2_data = []
        domain_dict = {1283088: 0, 1283089: 2, 1283090: 1}
        domain_num = -1
        n = 0
        for fea_ids, fea_vals, label in ds:
            fea_ids = fea_ids.numpy()
            domain_id = fea_ids[:,-1]
            fea_ids = fea_ids[:, :17]
            label = label.numpy()
            for i in range(len(label)):
                n = n + 1
                if domain_id[i] in domain_dict:
                    domain = domain_dict[domain_id[i]]
                else:
                    domain_num = domain_num + 1
                    domain_dict[domain_id[i]] = domain_num
                    domain = domain_dict[domain_id[i]]
                if domain == scenes[0]:
                    domain0_data.append([fea_ids[i], label[i][0], domain])
                elif domain == scenes[1]:
                    domain1_data.append([fea_ids[i], label[i][0], domain])
                elif domain == scenes[2]:
                    domain2_data.append([fea_ids[i], label[i][0], domain])
                else:
                    print('domian error')
                    break
        print(domain_dict)
        print(n)
        print(len(domain0_data), len(domain1_data), len(domain2_data))
        return domain0_data, domain1_data, domain2_data

class Ali_CCP_Target(object):
    def __init__(self, tfrecord_path):
        self.SAMPLES = 1
        self.FIELDS = 17
        self.tfrecord_path = tfrecord_path
        self.description = {
            "feature": tf.io.FixedLenFeature([self.FIELDS], tf.int64),
            "label": tf.io.FixedLenFeature([self.SAMPLES], tf.float32),
            "domain": tf.io.FixedLenFeature([self.SAMPLES], tf.float32),
        }

    def get_data(self, data_type, batch_size=1):
        @tf.autograph.experimental.do_not_convert
        def read_data(raw_rec):
            example = tf.io.parse_single_example(raw_rec, self.description)
            return example['feature'], example['label'], example['domain']

        files = glob.glob(self.tfrecord_path + '/' + "{}*".format(data_type))
        ds = tf.data.TFRecordDataset(files).map(read_data, num_parallel_calls=tf.data.experimental.AUTOTUNE). \
            batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        for feature, label, domain in ds:
            yield feature, label, domain

    def transform_data(self, data_type):
        @tf.autograph.experimental.do_not_convert
        def read_data(raw_rec):
            example = tf.io.parse_single_example(raw_rec, self.description)
            return example['feature'], example['label'], example['domain']

        files = glob.glob(self.tfrecord_path + '/' + "{}*".format(data_type))
        ds = tf.data.TFRecordDataset(files).map(read_data, num_parallel_calls=tf.data.experimental.AUTOTUNE). \
            shuffle(buffer_size=5000000, seed=2022).batch(4096).prefetch(tf.data.experimental.AUTOTUNE)
        n = 0
        max_id = 0
        data = []
        for feature, label, domain in ds:
            feature = feature.numpy()
            label = label.numpy()
            domain = domain.numpy()
            tmp = feature.max()
            if tmp > max_id:
                max_id = tmp
            for i in range(len(label)):
                n = n + 1
                data.append([feature[i], label[i][0], domain[i][0]])
        print(n, len(data), max_id)
        return data, max_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transfrom original TFRecord to new TFRecord')


    parser.add_argument("--data_type", type=str, default='train')
    parser.add_argument("--stats", type=str, default='../data/ali-ccp/stats/')
    parser.add_argument("--record", type=str, default='../data/ali-ccp/tfrecord/')
    parser.add_argument("--scenes", type=int, default=[0, 1, 2])

    args = parser.parse_args()
    data_path = '../data/ali-ccp/'
    new_enum_path = '../data/ali-ccp/ctrcvr_enum_new.pkl'
    source_tfrecord_name = 'ctr_cvr.{}.tfrecord'.format(args.data_type)
    scenes = args.scenes
    data_tfrecord_path = args.record
    defaults_path = args.stats + 'defaults.pkl'

    print('write defaults begin')
    gen_defaults(new_enum_path, defaults_path)
    print('data transform begin')
    ccp = Ali_CCP_Source(data_path)
    start = time.clock()
    domain0_data, domain1_data, domain2_data = ccp.transform_data(source_tfrecord_name, scenes)
    end = time.clock()
    print("read time:%s", end - start)
    start = time.clock()
    np.random.shuffle(domain0_data)
    np.random.shuffle(domain1_data)
    np.random.shuffle(domain2_data)
    write_tfrecord(data_tfrecord_path, args.data_type, scenes[0], domain0_data)
    write_tfrecord(data_tfrecord_path, args.data_type, scenes[1], domain1_data)
    write_tfrecord(data_tfrecord_path, args.data_type, scenes[2], domain2_data)
    print('data transform end')
    end = time.clock()
    print("write time:%s", end - start)

