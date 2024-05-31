import tensorflow as tf
import argparse
import glob

def make_example(dd):
    features = {
        "feat_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=dd[1])),
        "feat_vals": tf.train.Feature(float_list=tf.train.FloatList(value=dd[2])),
        "label": tf.train.Feature(float_list=tf.train.FloatList(value=[dd[0]]))
    }
    return tf.train.Example(features=tf.train.Features(feature=features))


def make_example_scene(dd):
    features = {
        "feat_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=dd[1])),
        "feat_vals": tf.train.Feature(float_list=tf.train.FloatList(value=dd[2])),
        "label": tf.train.Feature(float_list=tf.train.FloatList(value=[dd[0]])),
        "label2": tf.train.Feature(float_list=tf.train.FloatList(value=[dd[3]])),
        "sid": tf.train.Feature(int64_list=tf.train.Int64List(value=[dd[4]]))
    }
    return tf.train.Example(features=tf.train.Features(feature=features))


def write_tfrecord(filename, data):
    print("write tfrecord")
    writer = tf.io.TFRecordWriter(filename)
    c = 0
    for dd in data:
        ex = make_example(dd)
        writer.write(ex.SerializeToString())
        c += 1
        if c % 100000 == 0:
            print(c)
    writer.close()


def read_ffm(filename, sep=' '):
    print("read ffm")
    f_data = []
    c = 0
    max_field = 0
    max_feature = 0
    with open(filename, "r") as f:
        for line in f.readlines():
            dd = line.strip().split(sep)
            label = float(dd[0].split(',')[0])
            features = dd[1:]
            fi = [int(feat.split(':')[0]) for feat in features]
            idx = [int(feat.split(':')[1]) for feat in features]
            val = [float(feat.split(':')[2]) for feat in features]
            f_data.append([label, idx, val])
            max_field = max(max_field, len(fi))
            max_feature = max(max_feature, max(idx))
            c += 1
            if c % 100000 == 0:
                print(c)
    print(max_field, max_feature)
    return f_data


def cnt_scene_num(filename, sep=' '):
    print("cnt scene")
    scene_feats = {}
    c = 0
    with open(filename, "r") as f:
        for line in f.readlines():
            dd = line.strip().split(sep)
            scene_f = dd[-1]
            if scene_f in scene_feats.keys():
                scene_feats[scene_f] += 1
            else:
                scene_feats[scene_f] = 1
            c += 1
            if c % 100000 == 0:
                print(c)
    print(scene_feats)


def read_write(filename, sep=' '):
    print("START")
    writer = tf.io.TFRecordWriter(filename+".tfrecords")
    c = 0
    max_field = 0
    max_feature = 0
    with open(filename+".ffm", "r") as f:
        for line in f.readlines():
            dd = line.strip().split(sep)
            label = float(dd[0].split(',')[0])
            features = dd[1:]
            fi = [int(feat.split(':')[0]) for feat in features]
            idx = [int(feat.split(':')[1]) for feat in features]
            val = [float(feat.split(':')[2]) for feat in features]
            dd = [label, idx, val]
            ex = make_example(dd)
            writer.write(ex.SerializeToString())
            max_field = max(max_field, len(fi))
            max_feature = max(max_feature, max(idx))
            c += 1
            if c % 100000 == 0:
                print(c)
    print(max_field, max_feature)
    writer.close()


def read_write_scene(filename, sep=' '):
    print("START")
    writer = tf.io.TFRecordWriter(filename+".tfrecords")
    c = 0
    max_field = 0
    max_feature = 0
    with open(filename+".ffm", "r") as f:
        for line in f.readlines():
            dd = line.strip().split(sep)
            label = float(dd[0].split(',')[0])
            label2 = float(dd[0].split(',')[1])
            features = dd[1:]
            fi = [int(feat.split(':')[0]) for feat in features]
            idx = [int(feat.split(':')[1]) for feat in features]
            val = [float(feat.split(':')[2]) for feat in features]
            sid = idx[-1]-1283087
            dd = [label, idx, val, label2, sid]
            ex = make_example_scene(dd)
            writer.write(ex.SerializeToString())
            max_field = max(max_field, len(fi))
            max_feature = max(max_feature, max(idx))
            c += 1
            if c % 100000 == 0:
                print(c)
    print(max_field, max_feature)
    writer.close()


def convert_ffm_to_tfrecord(ffm_file_path, tfrecord_file_path, sep = ' '):
    data = read_ffm(ffm_file_path, sep)
    write_tfrecord(tfrecord_file_path, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../rank/multi_domain/data/ali_ccp/valid', help="train file name")
    args, unparsed = parser.parse_known_args()

    # data_path = '../data/ali_ccp/ctr_cvr.train'
    # read_write(data_path)
    # cnt_scene_num('../rank/multi_domain/data/ali_ccp/train.ffm')
    #read_write_scene(args.data_path)