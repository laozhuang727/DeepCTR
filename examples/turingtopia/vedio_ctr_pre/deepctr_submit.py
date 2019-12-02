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
from tqdm import tqdm

from deepctr.models import DeepFM
from deepctr.inputs import SparseFeat, DenseFeat, get_input_feature_names
from tensorflow.python import keras

import matplotlib.pyplot as plt
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

""" 运行DeepFM """

BATCH_SIZE = 10240
EPOCHS = 3

path = '~/Downloads/'
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')

print (u"正负样本分布", train['target'].value_counts())

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


# ====
# 类别特征count特征
sparse_col_list = [i for i in train.columns if i not in ['id', 'lat', 'lng', 'target', 'timestamp', 'ts']] + ['level']
for i in tqdm(sparse_col_list):
    data['{}_count'.format(i)] = data.groupby(['{}'.format(i)])['id'].transform('count')

# 类别特征五折转化率特征
data['ID'] = data.index
data['fold'] = data['ID'] % 5
data.loc[data.target.isnull(), 'fold'] = 5
target_feat = []
for i in tqdm(sparse_col_list):
    target_feat.extend([i + '_mean_last_1'])
    data[i + '_mean_last_1'] = None
    for fold in range(6):
        data.loc[data['fold'] == fold, i + '_mean_last_1'] = data[data['fold'] == fold][i].map(
            data[(data['fold'] != fold) & (data['fold'] != 5)].groupby(i)['target'].mean()
        )
    data[i + '_mean_last_1'] = data[i + '_mean_last_1'].astype(float)
# ====

fix_len_category_columns = ['app_version', 'device_vendor', 'device_version', 'deviceid', 'guid', 'netmodel',
                            'newsid', 'osversion', 'gender', 'pos']
# fix_len_category_columns = [i for i in train.columns if i not in ['id', 'lat', 'lng', 'target', 'timestamp', 'ts']] + [
#     'level']

# 去掉'ts','lat', 'lng',
number_filter_cols = ['id', 'ID', 'fold', 'target', 'timestamp'] + fix_len_category_columns
# 数值特征
fix_len_number_columns = [i for i in data.columns if i not in number_filter_cols]

print ("fix_leng_category_cols:{}", fix_len_category_columns)
print ("fix_len_number_columns:{}", fix_len_number_columns)

target = ['target']

data[fix_len_category_columns] = data[fix_len_category_columns].fillna('-1', )
data[fix_len_number_columns] = data[fix_len_number_columns].fillna(-1, )

for feat in fix_len_category_columns:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])

mms = MinMaxScaler(feature_range=(0, 1))
data[fix_len_number_columns] = mms.fit_transform(data[fix_len_number_columns])

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
history = model.fit(train_model_input, train[target].values, batch_size=BATCH_SIZE, epochs=EPOCHS,
                    validation_split=0.2, verbose=1, callbacks=callbacks)

"""预测"""
val_pred_ans = model.predict(valid_model_input, BATCH_SIZE)
val_pred_ans = val_pred_ans.reshape(val_pred_ans.shape[0])

# result = test['id']
# result.loc[:, 'result'] = val_pred_ans

print("valid LogLoss", round(log_loss(valid[target].values, val_pred_ans), 4))
print("valid AUC", round(roc_auc_score(valid[target].values, val_pred_ans), 4))
print("valid f1", f1_score(valid[target].values, np.round(val_pred_ans)))


# 生成文件，因为类别不平衡预测出来概率值都很小，根据训练集正负样本比例，来确定target
# 0.893464711为训练集中0样本的比例，阈值可以进一步调整
test_pred_ans = model.predict(test_model_input, BATCH_SIZE)
test_pred_ans = test_pred_ans.reshape(test_pred_ans.shape[0])
test_pred_ans = np.round(test_pred_ans)

# test_pred_ans = (test_pred_ans >= 0.5 * 0.893464711).astype(int)

with open("result.csv", "w") as f:
    f.write("id,target\n")
    for i, j in zip(test['id'].values.tolist(), test_pred_ans):
        f.write(str(i) + "," + str(j) + "\n")

print "Success write to result.csv"

# 5. 画曲线
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(-0.5, 1.5)
    plt.show()


#
plot_learning_curves(history)


