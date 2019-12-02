# -*- coding:utf-8 -*-
"""

Author:
    ruiyan zry,15617240@qq.com

"""

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

train = pd.read_csv("~/Downloads/train.csv")
test = pd.read_csv("~/Downloads/test.csv")
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

# 对object类型特征进行编码
lbl = LabelEncoder()
object_col = [i for i in data.select_dtypes(object).columns if i not in ['id']]
for i in tqdm(object_col):
    data[i] = lbl.fit_transform(data[i].astype(str))

feature_name = [i for i in data.columns if i not in ['id', 'target', 'ID', 'fold', 'timestamp']]
tr_index = ~data['target'].isnull()
X_train = data[tr_index].reset_index(drop=True)[feature_name].reset_index(drop=True)
y = data[tr_index]['target'].reset_index(drop=True)
X_test = data.loc[data['id'].isin(test['id'].unique())][feature_name].reset_index(drop=True)

lgb_param = {
    'learning_rate': 0.1,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'feature_fraction': 0.6,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'num_leaves': 1000,
    'verbose': -1,
    'max_depth': -1,
    'seed': 2019,
    'n_jobs': 8
}


def eval_func(y_pred, train_data):
    y_true = train_data.get_label()
    score = f1_score(y_true, np.round(y_pred))
    return 'f1', score, True


print(X_train.shape, X_test[feature_name].shape)
oof = np.zeros(X_train.shape[0])
prediction = np.zeros(X_test.shape[0])
seeds = [19970412, 1, 4096, 2048, 1024]
num_model_seed = 1
for model_seed in range(num_model_seed):
    oof_lgb = np.zeros(X_train.shape[0])
    prediction_lgb = np.zeros(X_test.shape[0])
    skf = StratifiedKFold(n_splits=5, random_state=seeds[model_seed], shuffle=False)
    for index, (train_index, test_index) in enumerate(skf.split(X_train, y)):
        print(index)
        train_x, test_x, train_y, test_y = X_train.iloc[train_index], X_train.iloc[test_index], y.iloc[train_index], \
                                           y.iloc[test_index]
        lgb_train = lgb.Dataset(train_x, train_y)
        lgb_valid = lgb.Dataset(test_x, test_y, reference=lgb_train)
        lgb_model = lgb.train(lgb_param, lgb_train, num_boost_round=40000, valid_sets=[lgb_valid],
                              valid_names=['valid'], early_stopping_rounds=50, feval=eval_func,
                              # categorical_feature=cate_feat
                              verbose_eval=3)

        oof_lgb[test_index] += lgb_model.predict(test_x)
        prediction_lgb += lgb_model.predict(X_test[feature_name]) / 5

    print('AUC', roc_auc_score(y, oof_lgb))
    print(prediction_lgb.mean())
    oof += oof_lgb / num_model_seed
    prediction += prediction_lgb / num_model_seed
print('AUC', roc_auc_score(y, oof))
print('f1', f1_score(y, np.round(oof)))

# 生成文件，因为类别不平衡预测出来概率值都很小，根据训练集正负样本比例，来确定target
# 0.8934642948637943为训练集中0样本的比例，阈值可以进一步调整
submit = test[['id']]
submit['target'] = prediction
submit['target'] = submit['target'].rank()
submit['target'] = (submit['target'] >= submit.shape[0] * 0.8934642948637943).astype(int)
submit.to_csv("lgb.csv", index=False)

plt.figure(figsize=(12,6))
lgb.plot_importance(lgb_model, max_num_features=30)
plt.title("Featurertances")
plt.show()