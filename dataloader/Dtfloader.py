import tensorflow as tf
import glob
import torch
import os

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class AliExpressLoader(object):
    def __init__(self, tfrecord_path):
        self.SAMPLES = 1
        self.FIELDS = 79
        self.tfrecord_path = tfrecord_path
        self.description = {
            "label": tf.io.FixedLenFeature([self.SAMPLES], tf.float32),
            "domain": tf.io.FixedLenFeature([self.SAMPLES], tf.float32),
            "feature": tf.io.FixedLenFeature([self.FIELDS], tf.int64),
        }

    def get_train_data(self, data_type, batch_size=1):
        @tf.autograph.experimental.do_not_convert
        def read_data(raw_rec):
            example = tf.io.parse_single_example(raw_rec, self.description)
            return example['feature'], example['label'], example['domain']

        files = glob.glob(self.tfrecord_path + '/' + "{}*".format(data_type))

        ds = tf.data.TFRecordDataset(files).map(read_data, num_parallel_calls=tf.data.experimental.AUTOTUNE). \
            shuffle(buffer_size=50000000, seed=2022).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        data = []
        for x, y, d in ds:
            x = torch.from_numpy(x.numpy())
            y = torch.from_numpy(y.numpy())
            d = torch.from_numpy(d.numpy())
            data.append([x, y, d])
        return data

    def get_data(self, data_type, batch_size=1):
        @tf.autograph.experimental.do_not_convert
        def read_data(raw_rec):
            example = tf.io.parse_single_example(raw_rec, self.description)
            return example['feature'], example['label'], example['domain']

        files = glob.glob(self.tfrecord_path + '/' + "{}*".format(data_type))
        ds = tf.data.TFRecordDataset(files).map(read_data, num_parallel_calls=tf.data.experimental.AUTOTUNE). \
            batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        for x, y, d in ds:
            x = torch.from_numpy(x.numpy())
            y = torch.from_numpy(y.numpy())
            d = torch.from_numpy(d.numpy())
            yield x, y, d


class Ali_CCPLoader(object):
    def __init__(self, tfrecord_path):
        self.SAMPLES = 1
        self.FIELDS = 17
        self.tfrecord_path = tfrecord_path
        self.description = {
            "feature": tf.io.FixedLenFeature([self.FIELDS], tf.int64),
            "label": tf.io.FixedLenFeature([self.SAMPLES], tf.float32),
            "domain": tf.io.FixedLenFeature([self.SAMPLES], tf.float32),
        }
    def get_train_data(self, data_type, batch_size=1):
        @tf.autograph.experimental.do_not_convert
        def read_data(raw_rec):
            example = tf.io.parse_single_example(raw_rec, self.description)
            return example['feature'], example['label'], example['domain']

        files = glob.glob(self.tfrecord_path + '/' + "{}*".format(data_type))
        ds = tf.data.TFRecordDataset(files).map(read_data, num_parallel_calls=tf.data.experimental.AUTOTUNE). \
            shuffle(buffer_size=50000000, seed=2022).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        data = []
        for x, y, d in ds:
            x = torch.from_numpy(x.numpy())
            y = torch.from_numpy(y.numpy())
            d = torch.from_numpy(d.numpy())
            data.append([x, y, d])
        return data

    def get_data(self, data_type, batch_size=1):
        @tf.autograph.experimental.do_not_convert
        def read_data(raw_rec):
            example = tf.io.parse_single_example(raw_rec, self.description)
            return example['feature'], example['label'], example['domain']

        files = glob.glob(self.tfrecord_path + '/' + "{}*".format(data_type))
        ds = tf.data.TFRecordDataset(files).map(read_data, num_parallel_calls=tf.data.experimental.AUTOTUNE). \
            batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        for x, y, d in ds:
            x = torch.from_numpy(x.numpy())
            y = torch.from_numpy(y.numpy())
            d = torch.from_numpy(d.numpy())
            yield x, y, d

