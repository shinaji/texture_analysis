#-*- coding:utf-8 -*-
"""
    main 

    Copyright (c) 2016 Tetsuya Shinaji

    This software is released under the MIT License.

    http://opensource.org/licenses/mit-license.php
    
    Date: 2016/02/15

"""

import numpy as np
from matplotlib import pyplot as plt
import datetime
from TextureAnalysis import GLHA
from TextureAnalysis import GLCM_3D
from TextureAnalysis import NGTDM_3D
from TextureAnalysis import GLSZM_3D
from TextureAnalysis.Utils import normalize
import glob
import pandas as pd
import os
import json
import copy
import argparse
from pydicom.dataset import Dataset, FileDataset
import random

__version__ = "0.1.0"

def main():

    parser = argparse.ArgumentParser(
        description='Texture Analysis test tool for PET images')
    parser.add_argument('--num_levels', '-n', type=int,
                        default=64,
                        help='Number of gray levels')
    parser.add_argument('--d_glcm', type=int,
                        default=1,
                        help='Distance parameter value of GLCM')
    parser.add_argument('--d_ngtdm', type=int,
                        default=1,
                        help='Distance parameter value of NGTDM')
    parser.add_argument('--data_dir_path', '-d', type=str,
                        default='./data/PA*',
                        help='Directory of target data files.')
    parser.add_argument('--out', '-o', type=str,
                        default='./results',
                        help='Directory to output the results')
    parser.add_argument('--save_voi_as_dicom', action='store_true',
                        default=False,
                        dest='save_voi_as_dicom',
                        help='Save VOI image as dicom')
    parser.add_argument('--save_glcm_as_png', action='store_true',
                        default=False,
                        dest='save_glcm_as_png',
                        help='Save GLCM as image data')
    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.mkdir(args.out)

    target = args.data_dir_path
    target = os.path.join(target, 'ST0/*ctr*[!_].npy')
    files = glob.glob(target)
    files.sort()
    print(files)
    files = np.array(files)

    all_data = []
    for i in range(len(files)):
        data = np.load(files[i])
        json_fname = copy.deepcopy(files[i]).replace('.npy', '_meta_data_.json')
        with open(json_fname.replace('\\', '/'), 'r') as f:
            json_data = json.load(f)

        patient_name = json_data['patient_name']
        series_description = json_data['series_description']
        radiopharmaceutical_info = json_data['radiopharmaceutical']
        if radiopharmaceutical_info.find('FDG') >= 0:
            radiopharmaceutical_info = 'FDG'
        elif radiopharmaceutical_info.find('FLT') >= 0:
            radiopharmaceutical_info = 'FLT'
        ref_roi_number = json_data['ref_roi_number'] if (
            json_data['ref_roi_number'] >= 0) else 'all'
        n_voxels = json_data['n_voxels']
        suv_conversion_coeff = json_data['SUV_conversion_coeff']
        voi_id = json_data['target_ctr_idx'] if (
            json_data['target_ctr_idx'] >= 0) else 'all'
        voi_volume = json_data['roi_volume']
        voi_min_value = np.unique(data[data>=0]).min()
        voi_max_value = np.unique(data[data>=0]).max()
        voi_mean_value = data[data>=0].mean()
        voi_var_value = data[data>=0].var()

        voi_info_labels = ['name', 'radiopharmaceutical_info',
                           'roi_series_description',
                           'voi_id', 'roi_num', 'data_filename',
                           'voi_min_value', 'voi_max_value', 'voi_mean_value',
                           'voi_var_value', 'voi_volume [ml]', 'n_voxels']
        voi_info_values = [patient_name, radiopharmaceutical_info, series_description,
                           voi_id, ref_roi_number, os.path.basename(files[i]),
                           voi_min_value, voi_max_value, voi_mean_value,
                           voi_var_value, voi_volume, n_voxels]

        if n_voxels <= 1:
            continue


        scale = args.num_levels

        non_croppped_roi_fname = \
            copy.deepcopy(files[i]).replace('.npy', '_non_cropped_.npy')
        non_masked_roi_fname = \
            copy.deepcopy(files[i]).replace('.npy', '_non_cropped_non_masked_.npy')
        non_cropped_roi = np.load(non_croppped_roi_fname)
        non_masked_roi = np.load(non_masked_roi_fname)
        non_cropped_roi, _, _ = normalize(
            non_cropped_roi, 0, scale-1, voi_min_value)
        if args.save_voi_as_dicom:
            convert_npy_to_dicom(
                '{}/{}_'.format(args.out, radiopharmaceutical_info) +
                os.path.basename(
                    copy.deepcopy(files[i])[0:-9] +
                    'roi_no_{}_norm_.dcm'.format(ref_roi_number)),
                non_cropped_roi,
                pixel_spacing=json_data['pixel_spacing'],
                slice_thickness=json_data['slice_thickness'],
            )
            convert_npy_to_dicom(
                '{}/{}_'.format(args.out, radiopharmaceutical_info) +
                os.path.basename(
                    copy.deepcopy(files[i])[0:-9] +
                    'roi_no_{}_org_.dcm'.format(ref_roi_number)),
                non_masked_roi,
                pixel_spacing=json_data['pixel_spacing'],
                slice_thickness=json_data['slice_thickness'],
            )

        glha = GLHA(data.flatten(),
                    level_min=0, level_max=scale-1,
                    threshold=voi_min_value)
        glha_labels, glha_values = glha.print_features()
        assert glha.hist.sum() == (data>=voi_min_value).sum(), \
            "{} {} {}".format(n_voxels, glha.hist.sum(), (data>=voi_min_value).sum())

        glcm = GLCM_3D(data, d=args.d_glcm,
                       level_min=0, level_max=scale-1,
                       threshold=voi_min_value)
        glcm_labels, glcm_values = glcm.print_features(show_figure=False)
        if args.save_glcm_as_png:
            plt.imshow(glcm.matrix*100, origin='lower')
            plt.xlabel('Normalized neighbour pixel value')
            plt.ylabel('Normalized center pixel value')
            cbar = plt.colorbar()
            cbar.set_label('Probability [%]')
            plt.title('{} roi#{} \nseries_description={}\nn_voxel={}'.format(
                patient_name, ref_roi_number, series_description, n_voxels),
            fontsize=9)
            plt.savefig('{}/GLCM_{}_{}_sd_{}_roi_{}.png'.format(
                args.out,
                radiopharmaceutical_info,
                patient_name,
                series_description,
                ref_roi_number),
                dip=100)
            # plt.show()
            plt.clf()

        ngtdm = NGTDM_3D(data, d=args.d_ngtdm,
                         level_min=1, level_max=scale,
                         threshold=voi_min_value)

        ntdm_labels, ntdm_values = ngtdm.print_features(show_figure=False)

        glszm = GLSZM_3D(data,
                         level_min=1, level_max=scale,
                         threshold=voi_min_value)
        glszm_labels, glszm_values = glszm.print_features(show_figure=False)

        labels = voi_info_labels + glha_labels + glcm_labels + ntdm_labels + glszm_labels
        values = voi_info_values + glha_values + glcm_values + ntdm_values + glszm_values

        df = pd.DataFrame([values])
        df.columns = labels

        all_data.append(df)

    filename = '{}/{}_results.xlsx'.format(
        args.out, datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S'))
    writer = pd.ExcelWriter(filename)
    all_data = pd.concat(all_data)
    all_data.to_excel(
        writer,
        sheet_name='results',
        index=False,
    )
    conf = pd.DataFrame(
        np.array(['Num gray levels: %d' % scale,
                  'Distance (GLCM): %d' % args.d_glcm,
                  'Distance (NGTDM): %d' % args.d_ngtdm]).reshape(-1, 1))
    conf.to_excel(
        writer,
        sheet_name='parameters',
        header=False,
        index=False,
    )
    writer.save()


def convert_npy_to_dicom(fname, npy_array,
                         slice_thickness=None,
                         pixel_spacing=None):
    """
    convert npy array to dicom
    :param fname: file name
    :param npy_array: npy array
    :param slice_thickness: slice thickness
    :param pixel_spacing: pixel spacing
    :return:  dcm
    """

    uint16_img = np.array(npy_array)
    uint16_img = (
        (uint16_img - uint16_img.min()) /
        (uint16_img.max() - uint16_img.min()) * (2**16 - 1)
    ).astype(np.uint16)
    dim = len(uint16_img.shape)
    if dim == 1:
        raise Exception('Cannot convert 1D array to dicom')
    elif dim == 2:
        uint16_img = uint16_img[np.newaxis, :, :]
    elif dim > 3:
        raise Exception('{}D array is not supported.'.format(dim))
    x_min = npy_array.min()
    x_max = npy_array.max()
    x_max_min = x_max - x_min
    t_max = (2**16) - 1
    slope = x_max_min / t_max
    intercept = x_min

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '0.0.000.000000.0.0.0.0.0.00'
    file_meta.MediaStorageSOPInstanceUID = \
        '333.333.0.0.0.333.333333333.{}'.format(
            datetime.now().timestamp())
    file_meta.ImplementationClassUID = '0.0.0.0'
    dcm = FileDataset(fname, {}, file_meta=file_meta, preamble=b'\0' * 128)

    dcm.Modality = 'OT'
    dcm.ImageType = ['ORIGINAL', 'PRIMARY']

    dcm.ContentDate = datetime.now().strftime('%Y%m%d')
    dcm.ContentTime = datetime.now().strftime('%H%M%S')
    dcm.InstanceCreationDate = datetime.now().strftime('%Y%m%d')
    dcm.InstanceCreationTime = datetime.now().strftime('%H%M%S')
    dcm.SeriesDate = datetime.now().strftime('%Y%m%d')
    dcm.SeriesTime = datetime.now().strftime('%H%M%S')
    dcm.AcquisitionTime = datetime.now().strftime('%H%M%S')
    dcm.PatientName = os.path.basename(fname)
    dcm.PatientBirthDate = datetime.now().strftime('%Y%m%d')
    dcm.PatientAge = '000Y'
    dcm.PatientSize = 1
    dcm.PatientWeight = 1
    dcm.PatientID = os.path.basename(fname)
    dcm.PatientSex = 'O'
    dcm.StudyDescription = os.path.basename(fname)
    dcm.StudyDate = datetime.now().strftime('%Y%m%d')
    dcm.StudyTime = datetime.now().strftime('%H%M%S')
    dcm.StudyID = os.path.basename(fname)
    dcm.SeriesDescription = os.path.basename(fname)
    dcm.SamplesPerPixel = 1
    dcm.PhotometricInterpretation = 'MONOCHROME1'
    dcm.PixelRepresentation = 0  # unsigned 0, signed 1
    dcm.HighBit = 16
    dcm.BitsStored = 16
    dcm.BitsAllocated = 16
    dcm.SmallestImagePixelValue = uint16_img.min()
    dcm.LargestImagePixelValue = uint16_img.max()
    dcm.Columns = uint16_img.shape[2]
    dcm.Rows = uint16_img.shape[1]
    dcm.NumberOfFrames = uint16_img.shape[0]
    dcm.NumberOfSlices = uint16_img.shape[0]
    dcm.ImagesInAquisition = uint16_img.shape[0]
    dcm.RescaleIntercept = intercept
    dcm.RescaleSlope = slope
    dcm.SliceVector = (np.arange(uint16_img.shape[0]) + 1).tolist()
    dcm.FrameIncrementPointer = [(0x0054, 0x0080)]

    dcm.PixelData = uint16_img.tostring()
    dcm.SliceThickness = 1 if slice_thickness is None else slice_thickness
    ps = 1 if pixel_spacing is None else pixel_spacing
    if isinstance(ps, list) or isinstance(ps, np.ndarray):
        dcm.PixelSpacing = [ps[0], ps[1]]
    else:
        dcm.PixelSpacing = [ps, ps]
    dcm.InstanceCreatorUID = '333.333.0.0.0'
    dcm.SOPClassUID = '0.0.000.00000.0.0.0.0.0.00'
    dcm.SOPInstanceUID = '333.333.0.0.0.{}'.format(
        datetime.now().timestamp())
    dcm.StudyInstanceUID = '333.333.0.0.0.{}'.format(datetime.now().timestamp())
    dcm.SeriesInstanceUID = '333.333.0.0.0.{}.3333'.format(
        datetime.now().timestamp())
    dcm.FrameOfReferenceUID = dcm.StudyInstanceUID
    dcm.SeriesNumber = 0
    dcm.InstanceNumber = 0
    dcm.BodyPartExamined = 'UNKNOWN'
    dcm.Manufacturer = 'DicomConversionUtils'
    dcm.DeviceSerialNumber = ''
    dcm.AcquisitionTerminationCondition = 'MANU'
    dcm.SoftwareVersions = 'UNKNOWN'
    dcm.AccessionNumber = '{:13d}'.format(random.randint(0, 1e13))
    dcm.InstitutionName = 'DicomConversionUtils'

    dcm.save_as(fname)
    return dcm


if __name__ == '__main__':
    main()