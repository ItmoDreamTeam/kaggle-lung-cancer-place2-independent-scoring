import settings
import os
import numpy as np
import dicom as dicom
import scipy.ndimage
from joblib import Parallel, delayed

INPUT_DIR = settings.DATASET_DIR
OUTPUT_DIR = settings.TMP_DIR + '/1mm'


def load_scan(path):
    slices = [dicom.read_file(os.path.join(path, s)) for s in os.listdir(path)]
    slices.sort(key=lambda x: x.ImagePositionPatient[2], reverse=True)
    depths = [slice.ImagePositionPatient[2] for slice in slices]
    # remove slices with acquisition numbers not = 1

    if len(depths) != len(set(depths)):
        # duplicated positions!
        print 'file', path, 'has duplicate ImagePositionPatients'
        slices.sort(key=lambda x: x.InstanceNumber)
        slices = [s for s in slices if s.AcquisitionNumber == 1]

    slice_thickness = slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2]
    if slice_thickness == 0:
        print 'image with zero slice thickness', path
        assert False

    ip0, ip1 = slices[0].ImagePositionPatient[:2]
    err_msg = False
    for s in slices:
        s.SliceThickness = slice_thickness
        assert s.ImagePositionPatient[0] == ip0 and s.ImagePositionPatient[1] == ip1, 'error'
        if 'SliceLocation' not in s or s.SliceLocation != s.ImagePositionPatient[2] and err_msg == False:
            print 'weird patient to QA', path
            err_msg = True

        orient = map(float, s.ImageOrientationPatient)
        if orient != [1, 0, 0, 0, 1, 0]:
            print 'bad orient'
            print s.ImageOrientationPatient
            print orient
            assert False

    return slices


def get_pixels_hu(scans):
    image = np.stack([s.pixel_array * s.RescaleSlope + s.RescaleIntercept for s in scans], axis=2).astype(np.int16)
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image < -1990] = -1000
    return np.array(image)


def resample(image, scan, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    spacing = map(float, (scan[0].PixelSpacing + [scan[0].SliceThickness]))
    spacing = np.array(list(spacing))
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    return image, new_spacing


def process_patient(patient):
    # read, transform and save
    scans = load_scan(os.path.join(INPUT_DIR, patient))  # matches last dimensions of px_raw
    px_raw = get_pixels_hu(scans)  # voxel
    px_rescaled, _ = resample(px_raw, scans, new_spacing=[1, 1, 1])
    np.save(os.path.join(OUTPUT_DIR, patient + '.npy'), px_rescaled)


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    patients = os.listdir(INPUT_DIR)
    patients.sort()
    Parallel(n_jobs=8, verbose=1)(delayed(process_patient)(patient) for patient in patients)
