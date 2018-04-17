import settings
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

ENSEMBLE1_DIR = settings.TMP_DIR + '/ensemble1'
ENSEMBLE2_DIR = settings.TMP_DIR + '/ensemble2'


def process_ensemble1_train(labels):
    df_ens1 = pd.read_csv(settings.MODEL_DIR + '/prediction/weighted_ensemble_v1_nodulesv29_stage1.csv')
    df_ens1['id'] = df_ens1['patient'].apply(lambda x: x.split('_')[0])
    df_masses = pd.read_csv(settings.MODEL_DIR + '/prediction/stage1_masses_predictions.csv')
    df_masses = df_masses.rename(columns={'patient_id': 'id', 'prediction': 'mass_pred'})
    df_ens1 = pd.merge(left=df_ens1, right=df_masses, how='outer', on='id')
    df_ens1 = pd.merge(left=labels, right=df_ens1, how='outer', on='id')
    train_filter = pd.notnull(df_ens1['cancer']).values
    ens1_cols = df_ens1.drop(['id', 'cancer', 'patient'], 1).columns
    X = df_ens1.loc[train_filter][ens1_cols].values
    Y = df_ens1.loc[train_filter]['cancer'].values
    ens1_lr = LogisticRegression(penalty='l1', C=10000)
    Yh = cross_val_predict(ens1_lr, X, Y, cv=25, method='predict_proba', n_jobs=6)[:, 1]
    df_ens1.loc[train_filter, 'yh_ens1'] = Yh
    ens1_lr.fit(X, Y)
    return ens1_lr, ens1_cols


def process_ensemble2_train(names, labels):
    df_masses = pd.read_csv(settings.MODEL_DIR + '/prediction/stage1_masses_predictions.csv')
    df_masses = df_masses.rename(columns={'patient_id': 'id', 'prediction': 'mass_pred'})

    ens2_models = []
    ens2_columns = []

    dfs_sg1 = []
    for name in names:
        dfs_sg1.append(pd.read_csv(settings.MODEL_DIR + '/prediction/model_features_stage1_' + name + '.csv'))
    df_ens2 = df_masses[['id']]

    for i, (df, name) in enumerate(zip(dfs_sg1, names)):
        df['id'] = df['patient'].apply(lambda x: x.split('_')[0])
        df = pd.merge(left=labels, right=df, how='outer', left_on='id', right_on='id')
        df = pd.merge(left=df, right=df_masses, how='outer', on='id')
        train_filter = pd.notnull(df['cancer']).values
        x_cols = df.drop(['id', 'cancer', 'patient'], 1).columns
        X = df.loc[train_filter][x_cols].values
        Y = df.loc[train_filter]['cancer'].values
        lr = LogisticRegression(penalty='l1', C=10000)
        Yh = cross_val_predict(lr, X, Y, cv=25, method='predict_proba', n_jobs=5)[:, 1]
        df.loc[train_filter, 'yh_' + name] = Yh
        lr.fit(X, Y)

        ens2_models.append(lr)
        ens2_columns.append(x_cols)

        Xtest = df.loc[~train_filter][x_cols].values
        if Xtest.shape[0] > 0:
            df.loc[~train_filter, 'yh_' + name] = lr.predict_proba(Xtest)[:, 1]
        else:
            print 'found no missing labels for ens2 stage1. does this make sense?'
        df = df[['id', 'yh_' + name]]
        df_ens2 = pd.merge(df_ens2, df, how='outer', on='id')
    return ens2_models, ens2_columns


def process_ensemble1(ensemble1_models, ensemble1_columns):
    df_ens1 = pd.read_csv(ENSEMBLE1_DIR + '/weighted_ensemble1_nodules_v29.csv')
    df_ens1['id'] = df_ens1['patient'].apply(lambda x: x.split('.')[0])

    Xtest = df_ens1[ensemble1_columns].values
    Yh = ensemble1_models.predict_proba(Xtest)[:, 1]
    df_ens1['yh_ens1'] = Yh

    df_ens1.set_index('id', inplace=True)
    df_ens1 = df_ens1[['yh_ens1']]
    df_ens1.to_csv(ENSEMBLE1_DIR + '/predictions.csv')


def process_ensemble2(names, ensemble2_models, ensemble2_columns):
    dfs = []
    for name in names:
        dfs.append(pd.read_csv(ENSEMBLE2_DIR + '/model_features_' + name + '.csv'))

    df_ens2 = None
    for i, dfi, model, name, column in zip(range(7), dfs, ensemble2_models, names, ensemble2_columns):
        dfi = dfi.rename(columns={dfi.columns[0]: 'patient'})
        dfi['id'] = dfi['patient'].apply(lambda x: x.split('_')[0])
        X = dfi[column].values
        Yh = model.predict_proba(X)[:, 1]
        dfi['yh_' + name] = Yh
        dfi = dfi[['id', 'yh_' + name]]
        if df_ens2 is None:
            df_ens2 = dfi
        else:
            df_ens2 = pd.merge(df_ens2, dfi, how='outer', on='id')

    df_ens2['yh_ens2'] = np.mean([df_ens2[c] for c in df_ens2.columns if c[:2] == 'yh'], axis=0)
    df_ens2 = df_ens2[['id', 'yh_ens2']]
    df_ens2.to_csv(ENSEMBLE2_DIR + 'predictions.csv')


if __name__ == '__main__':
    np.random.seed(42)

    labels = pd.read_csv(settings.MODEL_DIR + '/prediction/stage1plus2_labels.csv')
    names = ['37', '37b', '37c', '37d', '37f', '37g']

    ensemble1_models, ensemble1_columns = process_ensemble1_train(labels)
    ensemble2_models, ensemble2_columns = process_ensemble2_train(names, labels)

    process_ensemble1(ensemble1_models, ensemble1_columns)
    process_ensemble2(names, ensemble2_models, ensemble2_columns)
