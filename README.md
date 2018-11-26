## 项目介绍
* 该项目是天池瑞金医院知识抽取竞赛中实体识别部分的代码，模型为BiLSTM-GRU-CRF，竞赛[网址](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100066.0.0.3bea33af6U381n&raceId=231687)

## 主要package版本号

* Python 3.5.2

* TensorFlow-gpu 1.5

* jieba 0.39

* numpy 1.14.5


## 操作系统

* Ubuntu 16.04

## 操作方法

* 1. 训练时，需要将参数FLAGS.mode修改为train，测试时修改为test.

  2. data文件夹下除了数据，还有模型文件保存至ckpt_best，日志文件夹为log，验证集结果文件夹result.

## reference:

* [https://github.com/kyzhouhzau/BERT-NER](https://github.com/kyzhouhzau/BERT-NER)

* [https://github.com/zjy-ucas/ChineseNER](https://github.com/zjy-ucas/ChineseNER)


