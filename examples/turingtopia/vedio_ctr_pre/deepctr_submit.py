# -*- coding:utf-8 -*-
"""

Author:
    ruiyan zry,15617240@qq.com

"""
import os

from sklearn.metrics import log_loss, roc_auc_score, f1_score
from tensorflow.python.keras import backend as K

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from deepctr.models import DeepFM
from deepctr.inputs import SparseFeat, DenseFeat, get_input_feature_names
from tensorflow.python import keras

import matplotlib.pyplot as plt
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

""" 运行DeepFM """

BATCH_SIZE = 10240

path = '~/Downloads/'
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')

# print(train.head())
# data = pd.concat([train, test], ignore_index=False, sort=False)
data = train.append(test).reset_index(drop=True)

app = pd.read_csv("~/Downloads/app.csv")
app['applist'] = app['applist'].apply(lambda x: str(x)[1:-2])
app['applist'] = app['applist'].apply(lambda x: str(x).replace(' ', '|'))
app = app.groupby('deviceid')['applist'].apply(lambda x: '|'.join(x)).reset_index()
app['app_len'] = app['applist'].apply(lambda x: len(x.split('|')))
data = data.merge(app[['deviceid', 'app_len']], how='left', on='deviceid')
del app

user = pd.read_csv("~/Downloads/user.csv")
user = user.drop_duplicates('deviceid')
data = data.merge(user[['deviceid', 'level', 'personidentification', 'followscore', 'personalscore', 'gender']],
                  how='left', on='deviceid')
del user

# print(data.head()) timestamp：代表改用户点击改视频的时间戳，如果未点击则为NULL。 deviceid：用户的设备id。 newsid：视频的id。
# guid：用户的注册id。 pos：视频推荐位置。
# app_version：app版本。 device_vendor：设备厂商。 netmodel：网络类型。 osversion：操作系统版本。
# lng：经度。 lat：维度。 device_version：设备版本。
# ts：视频暴光给用户的时间戳。
#
# id,target,timestamp,deviceid,newsid,guid,pos,app_version,device_vendor,netmodel,osversion,lng,lat,
# device_version,ts id,deviceid,newsid,guid,pos,app_version,device_vendor,netmodel,osversion,lng,lat,device_version,
# ts 单值类别特征
# fix_len_category_columns = ['app_version', 'device_vendor', 'netmodel', 'osversion', 'device_version']

fix_len_category_columns = [i for i in train.columns if i not in ['id', 'lat', 'lng', 'target', 'timestamp', 'ts']] + ['level']

# 数值特征
fix_len_number_columns = []
# fix_len_number_columns = ['timestamp', 'lng', 'lat', 'ts']

target = ['target']

data[fix_len_category_columns] = data[fix_len_category_columns].fillna('-1', )
data[fix_len_number_columns] = data[fix_len_number_columns].fillna(0, )

for feat in fix_len_category_columns:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])

# mms = MinMaxScaler(feature_range=(0, 1))
# data[fix_len_number_columns] = mms.fit_transform(data[fix_len_number_columns])

fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                          for feat in fix_len_category_columns] + [DenseFeat(feat, 1, ) for feat in
                                                                   fix_len_number_columns]

dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns
feature_names = get_input_feature_names(linear_feature_columns + dnn_feature_columns)

train = data[~data['target'].isnull()]
test = data[data['target'].isnull()]
# print (train['target'].head(100))
# print ("\n\n\n\n===========\n\n\n\n")
# print (test.head(100))

train, valid = train_test_split(train, test_size=0.2)
train_model_input = {name: train[name] for name in feature_names}
valid_model_input = {name: valid[name] for name in feature_names}
test_model_input = {name: test[name] for name in feature_names}

# device = 'cuda:0'
"""第一步：初始化一个模型类"""
model = DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns, task='binary',
               l2_reg_embedding=1e-5)


def my_f1_score(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score_result = 2 * (precision * recall) / (precision + recall)
    return f1_score_result


"""第二步：调用compile()函数配置模型的优化器、损失函数、评价函数"""
model.compile("adam", "binary_crossentropy",
              metrics=[my_f1_score], )
callbacks = [keras.callbacks.EarlyStopping(
    patience=5, min_delta=1e-4)]

"""第三步：调用fit()函数训练模型"""
history = model.fit(train_model_input, train[target].values, batch_size=BATCH_SIZE, epochs=2,
                    validation_split=0.2, verbose=1,callbacks =callbacks)

"""预测"""
val_pred_ans = model.predict(valid_model_input, BATCH_SIZE)
val_pred_ans = val_pred_ans.reshape(val_pred_ans.shape[0])

# result = test['id']
# result.loc[:, 'result'] = val_pred_ans

print("valid LogLoss", round(log_loss(valid[target].values, val_pred_ans), 4))
print("valid AUC", round(roc_auc_score(valid[target].values, val_pred_ans), 4))
print("valid f1", f1_score(valid[target].values, np.round(val_pred_ans)))


# 5. 画曲线
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(-0.5, 1.5)
    plt.show()


#
plot_learning_curves(history)

test_pred_ans = model.predict(test_model_input, BATCH_SIZE)
test_pred_ans = np.round(test_pred_ans.reshape(test_pred_ans.shape[0]))
with open("result.csv", "w") as f:
    f.write("id,target\n")
    for i, j in zip(test['id'].values.tolist(), test_pred_ans):
        f.write(str(i) + "," + str(j) + "\n")
