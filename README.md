## WSDM24_MultiFS

Experiments codes for the paper:

Dugang Liu, Chaohua Yang, Xing Tang, Yejing Wang, Fuyuan Lyu, Weihong Luo, Xiuqiang He, Zhong Ming, Xiangyu Zhao. MultiFS: Automated multi-scenario feature selection in deep recommender systems. In Proceedings of WSDM '24.

**Please cite our WSDM '24 paper if you use our codes. Thanks!**


## Requirement

See the contents of requirements.txt


## Data Preprocessing

Please download the original data ([AliExpress](https://tianchi.aliyun.com/dataset/74690) and [AliCCP](https://tianchi.aliyun.com/dataset/408)) and place them in the corresponding directory of data.

You can prepare the AliExpress data in the following code.

```
# process origin data
python datatransform/AliExpress_process.py --dataset 'nl'
python datatransform/AliExpress_process.py --dataset 'fr'
python datatransform/AliExpress_process.py --dataset 'es'
python datatransform/AliExpress_process.py --dataset 'us'

# datatransform --> AliExpress-1 (nlfr)
python datatransform/AliExpress2tf.py --dataset1 '../data/nl/nl_data.csv' \
        --dataset2 '../data/fr/fr_data.csv' --stats '../data/AliExpress-1/stats_2' \
        --record '../data/AliExpress-1/threshold_2'

# datatransform --> AliExpress-2 (esus)
python datatransform/AliExpress2tf.py --dataset1 '../data/es/es_data.csv' \
        --dataset2 '../data/us/us_data.csv' --stats '../data/AliExpress-2/stats_2' \
        --record '../data/AliExpress-2/threshold_2'
```

You can prepare the AliCCP data in the following code.

```
# process origin data
python datatransform/ali_ccp_aitm_process.py --dataset_type 'train'
python datatransform/ali_ccp_aitm_process.py --dataset_type 'test'

python datatransform/get_ali_ccp.py --dataset_type 'train'
python datatransform/get_ali_ccp.py --dataset_type 'validation'
python datatransform/get_ali_ccp.py --dataset_type 'test'

# datatransform
python datatransform/ali_ccp2tf.py --data_type 'train' \
--stats '../data/ali-ccp/stats/' --record '../data/ali-ccp/tfrecord/'
python datatransform/ali_ccp2tf.py --data_type 'validation' \
--stats '../data/ali-ccp/stats/' --record '../data/ali-ccp/tfrecord/'
python datatransform/ali_ccp2tf.py --data_type 'test' \
--stats '../data/ali-ccp/stats/' --record '../data/ali-ccp/tfrecord/'
```

## Usage

An example of running a backbone model:

```
python trainer.py --dataset AliExpress-1 --model deepfm
```

An example of running MultiFS:

```
# For AliExpress-1/AliExpress-2 
python multiMaskTrainer2.py --dataset AliExpress-1 --model deepfm

# For AliCCP
python multiMaskTrainer3.py --dataset ali-ccp --model deepfm
```

## 

If you have any issues or ideas, feel free to contact us ([dugang.ldg@gmail.com](mailto:dugang.ldg@gmail.com)).

