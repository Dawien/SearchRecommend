# DeepFM
Code of papers: DeepFM: A Factorization-Machine based Neural Network for CTR Prediction

This code is realized by tensorflow.
Reference：https://github.com/xiaoleiHou214/Basic-DeepFM-model.git

1、DeepFM是FM + DNN的组合，可以滴低阶特征交互和高阶特征交互进行建模，不需要额外进行特征工程；
2、模型wide-FM部分与deep-DNN部分共享输入，且共享Embedding向量，可以高效的进行联合训练；
3、不需要利用FM进行Embedding的预训练。

工程结构：
1、DeepFM.py -- 模型的主要构建部分
2、DeepFMTrain.py -- 模型训练
3、DeepFMEval.py -- 模型的推断预测
4、DateProces.py -- 数据处理，将数据转换成特征索引和特征值存在，并统计特征的维度
5、ParseConfig.py -- 解析参数文件：config.conf
6、run_deep_fm.py -- 主函数入口，训练模型并验证评估

数据转换说明，结合数据处理代码可以更直观的理解
1、需要给每一个特征对应一个K维的Embedding，那么就需要得到连续变量或离散变量对应的特征索引feature_index，举例：
fileds：         性别                 周几                  职业
range：         男/女                周一~周天             学生/白领
one-hot：
             男          女           一        二        三        四        五        六        七        学生        白领
feature index：
             0           1            2         3        4         5         6        7         8         9          10
实例：  男/二/学生     对应的feature index： 0，3，9
       女/四/白领     对应的feature index： 1，5，10

考虑到特征会有连续变量，也可以给他们创建对应的特征索引，如：
fileds：        性别            周几                  职业                  体重                工资
range：         男/女           周一~周天            学生/白领             xx KG                xx W
one-hot：
      男        女      一      二     三     四     五     六     七     学生     白领     体重     工资
feature index： 
      0         1       2        3     4      5      6      7      8       9        10       11       12
实例：  男/二/学生/50/300       对应的feature index： 0，3，9，11，12
       女/四/白领/48/1000      对应的feature index： 1，5，10，11，12
但在FM中，不仅仅是有Embedding的內积，特征值也是需要的。对于离散变量来说，特征值 = 1，对于连续变量来说，特征值就是其本身了。如此一来，我们就能得到如下的数据格式：
fileds：         性别            周几                  职业                  体重                工资
range：          男/女           周一~周天            学生/白领             xx KG                xx W
one-hot：
        男       女      一     二     三     四     五     六     七     学生     白领     体重     工资
feature index： 
        0        1       2      3      4      5      6      7      8       9        10       11       12
实例：  男/二/学生/50/300  -  feature index： 0，3，9，11，12
                         - feature value： 1，1，1，50，300
       女/四/白领/48/1000  - feature index： 1，5，10，11，12
                          - feature value：1，1，1，48，1000
通过对上述数据的观察，可以知道，feature_size = 13，field_size = 5
