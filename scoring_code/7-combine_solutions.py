import pandas as pd
import numpy as np

if __name__ == '__main__':
    np.random.seed(42)
    dh_ens1 = pd.read_csv("ens1_preds_stage2.csv")
    dh_ens2 = pd.read_csv("ens2_preds_stage2.csv")
    jul_preds_test = pd.read_csv("julian_preds_test.csv")
    df = pd.merge(left=jul_preds_test, right=dh_ens1, how='inner', left_on='patient_id', right_on='id')
    df = pd.merge(left=df, right=dh_ens2, how='inner', left_on='patient_id', right_on='id')
    df['cancer'] = 0.4 * df['yh_jul'] + 0.6 * (0.7 * df['yh_ens1'] + 0.3 * df['yh_ens2'])
    df['id'] = df['patient_id']
    df = df[['id', 'cancer']]
    df.to_csv('final_predictions_dh_blend.csv', index=False)
