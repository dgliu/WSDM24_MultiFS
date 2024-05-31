import pandas as pd
import numpy as np
import argparse
import os
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='nl', help="nl/fr/es/us")
    # parser.add_argument('--dataset_type', type=str, default='test', help="train/test")
    args, unparsed = parser.parse_known_args()

    user_train_data_path = '../data/{}/{}_user_train.csv'.format(args.dataset, args.dataset)
    item_train_data_path = '../data/{}/{}_item_train.csv'.format(args.dataset, args.dataset)
    train_data_path = '../data/{}/{}_train.csv'.format(args.dataset, args.dataset)
    user_test_data_path = '../data/{}/{}_user_test.csv'.format(args.dataset, args.dataset)
    item_test_data_path = '../data/{}/{}_item_test.csv'.format(args.dataset, args.dataset)
    test_data_path = '../data/{}/{}_test.csv'.format(args.dataset, args.dataset)
    data_path = '../data/{}/{}_data.csv'.format(args.dataset, args.dataset)
    user_columns = ['Id', 'U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7', 'U8', 'U9', 'U10', 'U11', 'U12', 'U13', 'U14', 'U15',
                    'U16', 'U17', 'U18', 'U19', 'U20', 'U21', 'U22', 'U23', 'U24', 'U25', 'U26', 'U27', 'U28', 'U29', 'U30',
                    'U31', 'U32']
    item_columns = ['Id', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13', 'I14', 'I15',
                    'I16', 'I17', 'I18', 'I19', 'I20', 'I21', 'I22', 'I23', 'I24', 'I25', 'I26', 'I27', 'I28', 'I29', 'I30',
                    'I31', 'I32', 'I33', 'I34', 'I35', 'I36', 'I37', 'I38', 'I39', 'I40', 'I41', 'I42', 'I43', 'I44', 'I45',
                    'I46', 'I47', 'Label']
    reset_columns = ['Id', 'U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7', 'U8', 'U9', 'U10', 'U11', 'U12', 'U13', 'U14',
                     'U15',
                     'U16', 'U17', 'U18', 'U19', 'U20', 'U21', 'U22', 'U23', 'U24', 'U25', 'U26', 'U27', 'U28', 'U29',
                     'U30',
                     'U31', 'U32', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13',
                     'I14',
                     'I15', 'I16', 'I17', 'I18', 'I19', 'I20', 'I21', 'I22', 'I23', 'I24', 'I25', 'I26', 'I27', 'I28',
                     'I29',
                     'I30', 'I31', 'I32', 'I33', 'I34', 'I35', 'I36', 'I37', 'I38', 'I39', 'I40', 'I41', 'I42', 'I43',
                     'I44',
                     'I45', 'I46', 'I47', 'Label']

    # process train data
    user_train_data = pd.read_csv(user_train_data_path, names=user_columns, header=None)
    item_train_data = pd.read_csv(item_train_data_path, names=item_columns, header=None)
    # print(user_train.head(10))
    train_data = item_train_data.merge(user_train_data, how="left", on='Id').fillna(0)
    print(train_data.head(10))
    train_data = train_data[reset_columns]
    print(train_data.head(10))
    # train_data.to_csv(train_data_path, index=False, header=None)

    # process test data
    user_test_data = pd.read_csv(user_test_data_path, names=user_columns, header=None)
    item_test_data = pd.read_csv(item_test_data_path, names=item_columns, header=None)
    # print(user_test.head(10))
    test_data = item_test_data.merge(user_test_data, how="left", on='Id').fillna(0)
    print(test_data.head(10))
    test_data = test_data[reset_columns]
    print(test_data.head(10))
    # test_data.to_csv(test_data_path, index=False, header=None)

    data = pd.concat([train_data, test_data], ignore_index=True)
    data.to_csv(data_path, index=False, header=None)