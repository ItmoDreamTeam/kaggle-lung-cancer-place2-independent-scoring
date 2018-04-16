import settings
import os
import sys
import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.cluster import DBSCAN
from keras.models import load_model
from multiprocessing import Pool

INPUT_DIR = settings.TMP_DIR + '/v1_nodules'
OUTPUT_DIR = settings.TMP_DIR + '/ensemble2'


def random_perturb(Xbatch, rotate=False):
    # apply some random transformations
    swaps = np.random.choice([-1, 1], size=(Xbatch.shape[0], 3))
    Xcpy = Xbatch.copy()
    for i in range(Xbatch.shape[0]):
        # (1,64,64,64)
        # random
        Xcpy[i] = Xbatch[i, :, ::swaps[i, 0], ::swaps[i, 1], ::swaps[i, 2]]
        txpose = np.random.permutation([1, 2, 3])
        Xcpy[i] = np.transpose(Xcpy[i], tuple([0] + list(txpose)))
        if rotate:
            # arbitrary rotation is composition of two
            Xcpy[i, 0] = ndimage.interpolation.rotate(Xcpy[i, 0], np.random.uniform(-5, 5), axes=(1, 0), order=1,
                                                      reshape=False, cval=-1000, mode='nearest')
            Xcpy[i, 0] = ndimage.interpolation.rotate(Xcpy[i, 0], np.random.uniform(-5, 5), axes=(2, 1), order=1,
                                                      reshape=False, cval=-1000, mode='nearest')

    return Xcpy


def get_loc_features(locs, malig_scores, sizes):
    normalized_locs = locs.astype('float32') / sizes.astype('float32')

    # location of the most malignant tumor
    loc_from_malig = normalized_locs[np.argmax(malig_scores)]

    dist_mat = np.zeros((locs.shape[0], locs.shape[0]))
    for i, loc_a in enumerate(locs):
        for j, loc_b in enumerate(locs):
            dist_mat[i, j] = np.mean(np.abs(loc_a - loc_b))

    dbs = DBSCAN(eps=60, min_samples=2, metric='precomputed', leaf_size=2).fit(dist_mat)
    num_clusters = np.max(dbs.labels_) + 1
    num_noise = (dbs.labels_ == -1).sum()

    # new feature: sum of malig_scores but normalizing by cluster
    cluster_avgs = []
    for clusternum in range(num_clusters):
        cluster_avgs.append(malig_scores[dbs.labels_ == clusternum].mean())

    # now get the -1's
    for i, (clusterix, malig) in enumerate(zip(dbs.labels_, malig_scores)):
        if clusterix == -1:
            cluster_avgs.append(malig)

    weighted_mean_malig = np.mean(cluster_avgs)

    # size of biggest cluster
    sizes = np.bincount(dbs.labels_[dbs.labels_ > 0])
    if len(sizes) > 0:
        maxsize = np.max(sizes)
    else:
        maxsize = 1
    n_nodules = float(locs.shape[0])

    return np.concatenate([loc_from_malig, normalized_locs.std(axis=0),
                           [float(num_clusters) / n_nodules, float(num_noise) / n_nodules, weighted_mean_malig,
                            float(maxsize) / n_nodules]])


def process_voxels(model, voxels, locs, sizes):
    # mapping from the set of voxels for a patient to a set of features
    n_TTA = 10
    preds = []
    preds.append(np.concatenate(model.predict(voxels, batch_size=64), axis=1))

    # parallelism for data augmentation
    for i in range(n_TTA):
        preds.append(np.concatenate(model.predict(random_perturb(voxels, i < 4), batch_size=64), axis=1))

    preds = np.stack(preds)  # axis0 is over each TTA
    Xmean = np.mean(preds, axis=0)  # mean over tta, still have a set of predictions per voxel
    Xcomb = Xmean

    # axis 0 is per voxel, axis 1 is per feature
    # max and sum.
    xmax = np.max(Xcomb, axis=0)
    xsd = np.std(Xcomb, axis=0)

    location_feats = get_loc_features(locs, Xcomb[:, 3], sizes)
    return np.concatenate([xmax, xsd, location_feats], axis=0)


def score_model(modelpath, name):
    # score a model on the train and test set
    model = load_model(modelpath)
    files = [f.replace('vox_', '') for f in os.listdir(INPUT_DIR) if 'vox_' in f]

    all_features = []
    for i, patient in enumerate(files):
        patient_vox = np.load(os.path.join(INPUT_DIR, 'vox_' + patient))  # voxels[filter]
        patient_locs = np.load(os.path.join(INPUT_DIR, 'cents_' + patient))  # locations[filter]
        patient_sizes = np.load(os.path.join(INPUT_DIR, 'shapes_' + patient))  # sizes[filter]
        print patient_vox.shape[0], 'nodules for patient', patient, 'number', i + 1, 'of', len(files), 'model', name

        features = process_voxels(model, patient_vox, patient_locs, patient_sizes)
        all_features.append(features)
        X = np.stack(all_features)

    df = pd.DataFrame(data=X, index=files)
    df.to_csv(OUTPUT_DIR + '/' + name + '.csv')
    return


def score_wrapper(args):
    return score_model(*args)


if __name__ == '__main__':
    sys.setrecursionlimit(10000)

    model37 = settings.MODEL_DIR + '/ensemble2/model_des_v37_mse_64_finetune_04.h5'
    model37b = settings.MODEL_DIR + '/ensemble2/model_des_v37b_mse_64_finetune_04.h5'
    model37c = settings.MODEL_DIR + '/ensemble2/model_des_v37c_64_finetune_04.h5'
    model37d = settings.MODEL_DIR + '/ensemble2/model_des_v37d_64_finetune_04.h5'
    model37f = settings.MODEL_DIR + '/ensemble2/model_des_v37f_64_finetune_04.h5'
    model37g = settings.MODEL_DIR + '/ensemble2/model_des_v37g_64_finetune_04.h5'
    # model38 = settings.MODEL_DIR + '/ensemble2/model_des_v38_mse_64_finetune_04.h5'

    models = [model37, model37b, model37c, model37d, model37f, model37g]
    names = ['37', '37b', '37c', '37d', '37f', '37g']
    names = ['model_features_' + n for n in names]

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    pool = Pool(7)
    args = zip(models, names)
    pool.map(score_wrapper, args)
    pool.close()
