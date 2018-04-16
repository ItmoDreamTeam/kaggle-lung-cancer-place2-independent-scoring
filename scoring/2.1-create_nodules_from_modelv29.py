import settings
import os
import numpy as np
from keras.models import load_model

DATA_DIR = settings.TMP_DIR + '/1mm'
OUTPUT_BASE_PATH = settings.TMP_DIR + '/v29_nodules'


def find_start(arr, thresh=-800):
    # determine when the arr first exceeds thresh
    for i in range(arr.shape[0]):
        if arr[i] > thresh:
            return np.clip(i - 5, 0, arr.shape[0])
    return 0


def crop(img, thresh=-800):
    sum0 = img.mean(axis=(1, 2))
    sum1 = img.mean(axis=(0, 2))
    sum2 = img.mean(axis=(0, 1))

    start0 = find_start(sum0, thresh) + 1
    end0 = -1 * (find_start(sum0[::-1], thresh) + 1)
    start1 = find_start(sum1, thresh) + 1
    end1 = -1 * (find_start(sum1[::-1], thresh) + 1)
    start2 = find_start(sum2, thresh) + 1
    end2 = -1 * (find_start(sum2[::-1], thresh) + 1)
    assert start0 < int(img.shape[0] * .5) < img.shape[0] - end0, 'bad crop ' + str(start0) + ' ' + str(
        end0) + ' ' + str(img.shape[0])
    assert start1 < int(img.shape[1] * .5) < img.shape[1] - end1, 'bad crop ' + str(start1) + ' ' + str(
        end1) + ' ' + str(img.shape[1])
    assert start2 < int(img.shape[2] * .5) < img.shape[2] - end2, 'bad crop ' + str(start2) + ' ' + str(
        end2) + ' ' + str(img.shape[2])
    assert end0 < 0 and end1 < 0 and end2 < 0, 'one end >= 0'
    assert start0 > 0 and start1 > 0 and start2 > 0, 'one start <= 0'

    return img[start0:end0, start1:end1, start2:end2]


def get_strides(steps, size, offset, VOXEL_SIZE):
    if steps * VOXEL_SIZE < size - 2 * offset:
        # not enough coverage. start and end are modified
        start = (size - steps * VOXEL_SIZE) / 2
        end = size - start - VOXEL_SIZE
    else:
        start = offset
        end = size - VOXEL_SIZE - offset
    return list(np.around(np.linspace(start, end, steps)).astype('int32'))


def img_to_vox(img, VOXEL_SIZE):
    # first let's just get the minimum amount of coverage
    samples0 = int(img.shape[0] / float(VOXEL_SIZE)) + 4
    samples1 = int(img.shape[1] / float(VOXEL_SIZE)) + 4
    samples2 = int(img.shape[2] / float(VOXEL_SIZE)) + 4

    ixs0 = get_strides(samples0, img.shape[0], 0, VOXEL_SIZE)
    ixs1 = get_strides(samples1, img.shape[1], 0, VOXEL_SIZE)
    ixs2 = get_strides(samples2, img.shape[2], 0, VOXEL_SIZE)

    subvoxels = []
    locations = []
    centroids = []
    for i0, x0 in enumerate(ixs0):
        for i1, x1 in enumerate(ixs1):
            for i2, x2 in enumerate(ixs2):
                subvoxels.append(img[x0:x0 + VOXEL_SIZE, x1:x1 + VOXEL_SIZE, x2:x2 + VOXEL_SIZE])
                assert subvoxels[-1].shape == (VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE), 'bad subvoxel shape ' + str(
                    subvoxels[-1].shape) + ' ' + str([x0, x1, x2]) + ' ' + str(img.shape)
                locations.append((i0, i1, i2))
                centroids.append((x0 + VOXEL_SIZE / 2, x1 + VOXEL_SIZE / 2, x2 + VOXEL_SIZE / 2))
    X = np.stack(subvoxels, axis=0)
    X = np.expand_dims(X, 1)
    # normalized locations
    # allows us to de-weight certain places...

    return X, locations, centroids


def get_interesting_ixs(preds):
    # return the indices of interest
    ixs = []
    for i in range(preds.shape[0]):
        if preds[i, 0] > 5:
            ixs.append(i)
        elif preds[i, 1] > 0.3:
            ixs.append(i)
        elif preds[i, 2] > 0.3:
            ixs.append(i)
        elif preds[i, 3] > 0.3:
            ixs.append(i)

    if len(ixs) == 0:
        ixs = [np.argmax(preds[:, 3])]
    return np.array(ixs)


def load_and_txform_file(file, model, VOXEL_SIZE, batch_size):
    # read and convert to voxels
    xorig = np.load(os.path.join(DATA_DIR, file))
    x = np.clip(xorig, -1000, 400)
    try:
        x = crop(x)
    except:
        print 'couldn\'t crop', file
        print 'trying again with diff threshold'
        try:
            x = crop(x, -900)
        except:
            print 'still couldn\'t crop.'
            try:
                x = crop(x, -1000)
            except:
                print 'failed to crop at all :('
                exit()

    x = ((x + 1000.) / (400. + 1000.)).astype('float32')
    voxels, locs, centroids = img_to_vox(x, VOXEL_SIZE)
    # predict on voxels, keep top N ROIs
    preds = model.predict(voxels, batch_size=batch_size)

    if type(preds) == list:
        preds = np.concatenate(preds, axis=1)

    topNixs = get_interesting_ixs(preds)

    topNvox = voxels[topNixs]
    topNcentroids = np.array(centroids)[topNixs]

    return topNvox, topNcentroids, [x.shape] * topNcentroids.shape[0]


if __name__ == '__main__':
    VOXEL_SIZE = 64
    model_batch_size = 64

    model = load_model(settings.MODEL_DIR + '/ensemble1/model_LUNA_64_v29_14.h5')
    train_files = [f for f in os.listdir(DATA_DIR)]

    if not os.path.exists(OUTPUT_BASE_PATH):
        os.mkdir(OUTPUT_BASE_PATH)

    for i, file in enumerate(train_files):
        vox, cents, shapes = load_and_txform_file(file, model, VOXEL_SIZE, model_batch_size)
        print file, i, 'of', len(train_files), 'n_nod', vox.shape[0]
        np.save(os.path.join(OUTPUT_BASE_PATH, 'vox_' + file), vox)
        np.save(os.path.join(OUTPUT_BASE_PATH, 'cents_' + file), cents)
        np.save(os.path.join(OUTPUT_BASE_PATH, 'shapes_' + file), np.array(shapes))
