import settings
import pandas as pd

ENSEMBLE1_DIR = settings.TMP_DIR + '/ensemble1'
ENSEMBLE2_DIR = settings.TMP_DIR + '/ensemble2'

if __name__ == '__main__':
    dh_ens1 = pd.read_csv(ENSEMBLE1_DIR + '/predictions.csv')
    dh_ens2 = pd.read_csv(ENSEMBLE2_DIR + '/predictions.csv')

    df = dh_ens1
    df = pd.merge(left=df, right=dh_ens2, how='inner', left_on='id', right_on='id')

    df['cancer'] = 0.7 * df['yh_ens1'] + 0.3 * df['yh_ens2']
    # df['id'] = df['patient_id']
    df = df[['id', 'cancer']]
    df.to_csv(settings.TMP_DIR + '/final_predictions.csv')
