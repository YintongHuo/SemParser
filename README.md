# SemParser Replication Package

This is the replication package of our work **SemParser: A semantic parser for log analysis** published in *Proceedings of the 45th International Conference on Software and Engineering (ICSE 2023)*.

## Contents
* Data. We use `/data` to evaluate the semantics mining ability. For downstream tasks evaluation, we use existing datasets: [HDFS](https://github.com/logpai/loghub/tree/master/HDFS#hdfs_v1) and [OpenStack](https://github.com/dessertlab/OpenStack-Fault-Injection-Environment/tree/master/src/tests).
* Model. The implementation of SemParser is described in `/model`.

## Reproduce
1. Download the [word embedding file](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155160328_link_cuhk_edu_hk/EbNQg24fI_hKl0lRYgie0VkBrsVzHW4XVOdLQVeRN7Ugiw?e=Jo3V0g) and put in under `/model` folder.
2. Execute `python main.py`

## Results
We put our semantics mining experiment results under the folder `/data/prediction` and checkpoints under `model/ckpt`.


