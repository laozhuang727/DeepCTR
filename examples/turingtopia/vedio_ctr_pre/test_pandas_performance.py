import time
import unittest
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)

    def test_apply(self):
        start = time.time()
        app = pd.read_csv("~/Downloads/app.csv")
        app['applist'] = app['applist'].apply(lambda x: str(x)[1:-2])
        app['applist'] = app['applist'].apply(lambda x: str(x).replace(' ', '|'))
        app = app.groupby('deviceid')['applist'].apply(lambda x: '|'.join(x)).reset_index()
        app['app_len'] = app['applist'].apply(lambda x: len(x.split('|')))
        del app

        print ("duration: {}", time.time() - start)


if __name__ == '__main__':
    unittest.main()
