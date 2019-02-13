# -*- coding: utf-8 -*-  
# __author__ = 'ryan.zhuang'
# 2019 02月 13日
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import History

from deepctr import SingleFeat

from deepctr.models import AutoInt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

if __name__ == "__main__":
    data = pd.read_csv('../../examples/criteo_sample.txt')
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']


    # 对离散的特征进行one-hot编码
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    # 对连续的特征归一化
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 统计每一个离散特征value的个数
    sparse_feature_list = [SingleFeat(feat, data[feat].nunique())
                           for feat in sparse_features]
    dense_feature_list = [SingleFeat(feat, 0)
                          for feat in dense_features]

    train_data, test_data = train_test_split(data, test_size=0.2)

    model = AutoInt({"sparse": sparse_feature_list, "dense": dense_feature_list})
    model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )

    train_model_input = [train_data[feat.name].values for feat in sparse_feature_list] + \
                        [train_data[feat.name].values for feat in dense_feature_list]
    test_model_input = [test_data[feat.name].values for feat in sparse_feature_list] + \
                       [test_data[feat.name].values for feat in dense_feature_list]

    history = model.fit(train_model_input, train_data[target].values, batch_size=256, epochs=200, verbose=2, validation_split=0.2, )

    pred_ans = model.predict(test_model_input, batch_size=256)
    print("test LogLoss", round(log_loss(test_data[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test_data[target].values, pred_ans), 4))