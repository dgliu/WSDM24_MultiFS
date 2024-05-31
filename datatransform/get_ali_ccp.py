import joblib
import argparse
import gen_tfrecords

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
    '301']
enum_path = '../data/ali-ccp/ctrcvr_enum.pkl'
new_enum_path = '../data/ali-ccp/ctrcvr_enum_new.pkl'
def reindex():
    vocabulary = joblib.load(enum_path)
    vocabulary = {key: sorted(values) for key, values in vocabulary.items()}
    global_idx = 1
    feat_map = {}
    for feat in use_columns:
        feat_map[feat] = {}
        print(len(vocabulary[feat]) + 1 + global_idx)
        for i in range(0, len(vocabulary[feat]) + 1):
            feat_map[feat][str(i)] = global_idx
            global_idx += 1
    joblib.dump(feat_map, new_enum_path, compress=3)


def process_ffm(read_path):
    feat_map = joblib.load(new_enum_path)
    write_path = read_path+".ffm"
    print(read_path)
    fw = open(write_path, 'w')
    domain_dict = {'1283088': 0, '1283089': 0, '1283090': 0}
    with open(read_path, 'r') as f:
        next(f)
        c = 0
        for line in f:
            samples = line.strip().split(",")
            labels = samples[:2]
            feats1 = samples[2:]
            feats2 = []
            for field_id, feat_o in zip(use_columns, feats1):
                feat_id = str(feat_map[field_id][feat_o])
                if field_id == '301':
                    domain_dict[feat_id] += 1
                feats2.append(field_id+":"+feat_id+":1.0")
            new_line = ",".join(labels) + " " + " ".join(feats2) + "\n"
            fw.write(new_line)
            c += 1
            if c % 100000 == 0:
                print(c)
    print(domain_dict)
    fw.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type=str, default='train', help="train/validation/test")
    args, unparsed = parser.parse_known_args()

    if args.dataset_type == 'train':
        data_path = '../data/ali-ccp/ctr_cvr.{}'.format(args.dataset_type)
        reindex()
        print('reindex finished')
        process_ffm(data_path)
        gen_tfrecords.convert_ffm_to_tfrecord(data_path + '.ffm', data_path + '.tfrecord', sep=' ')
    elif args.dataset_type == 'validation':
        data_path = '../data/ali-ccp/ctr_cvr.validation'
        process_ffm(data_path)
        gen_tfrecords.convert_ffm_to_tfrecord(data_path + '.ffm', data_path + '.tfrecord', sep=' ')
    elif args.dataset_type == 'test':
        data_path = '../data/ali-ccp/ctr_cvr.test'
        process_ffm(data_path)
        gen_tfrecords.convert_ffm_to_tfrecord(data_path + '.ffm', data_path + '.tfrecord', sep=' ')
    else:
        print("dataset_type error")



