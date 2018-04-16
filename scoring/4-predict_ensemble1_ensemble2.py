import settings
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

ENSEMBLE1_DIR = settings.TMP_DIR + '/ensemble1'
ENSEMBLE2_DIR = settings.TMP_DIR + '/ensemble2'


def process_ensemble1():
    df_ens1 = pd.read_csv(ENSEMBLE1_DIR + '/weighted_ensemble1_nodules_v29.csv')
    df_ens1['id'] = df_ens1['patient'].apply(lambda x: x.split('_')[0])

    train_filter = pd.notnull(df_ens1['cancer']).values
    ens1_cols = df_ens1.drop(['id', 'cancer', 'patient'], 1).columns
    X = df_ens1.loc[train_filter][ens1_cols].values
    Y = df_ens1.loc[train_filter]['cancer'].values

    ens1_lr = LogisticRegression(penalty='l1', C=10000)
    Yh = cross_val_predict(ens1_lr, X, Y, cv=25, method='predict_proba', n_jobs=6)[:, 1]
    df_ens1.loc[train_filter, 'yh_ens1'] = Yh
    ens1_lr.fit(X, Y)
    Xtest = df_ens1.loc[~train_filter][ens1_cols].values

    if Xtest.shape[0] > 0:
        df_ens1.loc[~train_filter, 'yh_ens1'] = ens1_lr.predict_proba(Xtest)[:, 1]
    else:
        print 'Found no missing labels for ensemble 1. Does this make sense?'

    df_ens1.set_index('id', inplace=True)
    df_ens1 = df_ens1[['yh_ens1']]
    df_ens1.to_csv(ENSEMBLE1_DIR + '/predictions.csv')


def process_ensemble2(names):
    dfs = []
    for name in names:
        dfs.append(pd.read_csv(ENSEMBLE2_DIR + '/model_features_' + name + '.csv'))

    df_ens2 = None
    for i, (df, name) in enumerate(zip(dfs, names)):
        df['id'] = df['patient'].apply(lambda x: x.split('_')[0])

        train_filter = pd.notnull(df['cancer']).values
        x_cols = df.drop(['id', 'cancer', 'patient'], 1).columns
        X = df.loc[train_filter][x_cols].values
        Y = df.loc[train_filter]['cancer'].values

        lr = LogisticRegression(penalty='l1', C=10000)
        Yh = cross_val_predict(lr, X, Y, cv=25, method='predict_proba', n_jobs=5)[:, 1]
        df.loc[train_filter, 'yh_' + name] = Yh
        lr.fit(X, Y)
        Xtest = df.loc[~train_filter][x_cols].values

        if Xtest.shape[0] > 0:
            df.loc[~train_filter, 'yh_' + name] = lr.predict_proba(Xtest)[:, 1]
        else:
            print 'Found no missing labels for ensemble 2. Does this make sense?'

        df = df[['id', 'yh_' + name]]
        if df_ens2 is None:
            df_ens2 = df
        else:
            df_ens2 = pd.merge(df_ens2, df, how='outer', on='id')
        print names[i]

    df_ens2 = df_ens2.set_index('id')
    df_ens2['yh_ens2'] = np.mean([df_ens2[c] for c in df_ens2.columns if c[:2] == 'yh'], axis=0)
    df_ens2 = df_ens2[['yh_ens2']]
    df_ens2.to_csv(ENSEMBLE2_DIR + '/predictions.csv')


if __name__ == '__main__':
    np.random.seed(42)
    process_ensemble1()
    process_ensemble2(['37', '37b', '37c', '37d', '37f', '37g'])
