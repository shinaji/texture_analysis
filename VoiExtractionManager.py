#-*- coding:utf-8 -*-
"""
    VoiExtractionManager

    Copyright (c) 2016 Tetsuya Shinaji

    This software is released under the MIT License.

    http://opensource.org/licenses/mit-license.phps
    
    Date: 2016/02/10

"""

import pydicom as dicom
try:
    from pydicom.contrib import pydicom_series
except ImportError:
    from pydicom_ext import pydicom_series
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import glob
import datetime
import traceback
import os
import copy
import json
from Utils import is_inside
import time
import argparse

__version__ = "0.1.0"

class VoiExtractionManager:

    def __init__(self, pet, ct, ctr, target_ctr_idx=None):
        """
        init
        :param pet: pet dicom series
        :param ct: ct dicom series
        :param ctr: contour dicom
        :param target_ctr_idx: target contour index. if -1, use all contours
        """

        if len(ctr.ReferencedFrameOfReferenceSequence) == 1:
            raise Exception(
                "The given contour data has only one reference sequence."
                "Two reference sequences (CT and PET) are required to "
                "perform the voi extraction of PET images."
                "Probably, the given contour data contains reference sequence "
                "for CT images."
            )

        self.pet = pet  # copy.deepcopy(pet)
        self.ct = ct  # copy.deepcopy(ct)
        self.ctr = copy.deepcopy(ctr)
        self.ct_img = ct[0].get_pixel_array()
        self.pet_img = pet[0].get_pixel_array()

        self.ct_spacing = self.ct[0].info.PixelSpacing
        self.ct_slice_thickness = self.ct[0].info.SliceThickness
        self.ct_origin = self.ct[0].info.ImagePositionPatient

        self.pet_spacing = self.pet[0].info.PixelSpacing
        self.pet_slice_thickness = self.pet[0].info.SliceThickness
        self.pet_origin = self.pet[0].info.ImagePositionPatient

        self.suv_coeff = self.__get_SUV_conversion_coeff()

        self.target_ctr_sequence = target_ctr_idx
        self.rois = []
        self.rois_co_slice_idxs = None
        self.sop_instance_uids_of_rois = []
        self.masked_roi_imgs = None
        self.masked_roi_imgs_suv = None
        self.non_cropped_masked_roi_imgs = None
        self.co_pet_imgs = None
        self.update(None, target_ctr_idx)

    def update(self, ctr=None, target_ctr_idx=None):
        """
        update extraction result
        :param ctr: contour dicom
        :param target_ctr_idx: target contour index
        """

        if ctr is not None:
            self.ctr = copy.deepcopy(ctr)
        if target_ctr_idx is not None:
            self.target_ctr_sequence = target_ctr_idx
        else:
            self.target_ctr_sequence = 0

        if (len(self.ctr.ROIContourSequence) > 1) and (target_ctr_idx is None):
            print('---WARNING---')
            print(
                'This contour dicom contains {} contours, but target counter '
                'index was not given. Only first contour will be processed.'.format(
                    len(self.ctr.ROIContourSequence)
                )
            )
        self.rois = []
        self.__read_roi_data()
        (self.masked_roi_imgs, self.non_cropped_masked_roi_imgs,
         self.co_pet_imgs, self.rois_co_slice_idxs
         ) = self.__get_masked_roi_image()
        self.masked_roi_imgs_suv = self.masked_roi_imgs * self.suv_coeff

    def __get_masked_roi_image(self):
        """
        return masked pet image. outside of roi is filled with -1.
        The size of roi image is cropped to remove wasted space.
        :return: cropped images, non-cropped images, corresponding pet images,
                  corresponding pet slice indices of ROIs
        """
        pet_imgs = []
        roi_imgs = []
        roi_indices = []
        for i in range(len(self.rois)):
            idx_p, img_p, org_p, img_c, org_c = self.__get_corresponding_image(i)
            pet_imgs.append(img_p)
            roi_img = self.__mask_outside(img_p, self.rois[i])
            roi_imgs.append(roi_img)
            roi_indices.append(idx_p)

        if self.target_ctr_sequence >= 0:
            if not np.unique(roi_indices).shape[0] == (
                        np.unique(roi_indices).max() -
                        np.unique(roi_indices).min() + 1):
                raise Exception('Some slices between start slice and '
                                'end slice are missing. \n{}\n{}'.format(
                    np.unique(roi_indices),
                    list(range(np.unique(roi_indices).min(),
                         np.unique(roi_indices).max()+1))
                ))

        n_slices = np.unique(roi_indices).max() - \
                   np.unique(roi_indices).min() + 1
        co_pet_imgs = np.ones(
            (n_slices,
             roi_imgs[0].shape[0],
             roi_imgs[0].shape[1])
        ) * -1
        rois_co_slice_idx = []
        masked_roi_imgs = np.ones(
            (n_slices,
             roi_imgs[0].shape[0],
             roi_imgs[0].shape[1])
        ) * -1
        for i, idx in enumerate(roi_indices):
            masked_roi_imgs[idx - min(roi_indices)][roi_imgs[i] >= 0] = \
                roi_imgs[i][roi_imgs[i] >= 0]
            co_pet_imgs[idx - min(roi_indices)] = pet_imgs[i]
            rois_co_slice_idx.append(idx - min(roi_indices))

        if self.target_ctr_sequence == -1:
            for i, p_idx in enumerate(
                    range(min(roi_indices), max(roi_indices)+1)):
                co_pet_imgs[i] = np.array(
                    self.pet[0]._datasets[p_idx].pixel_array *
                    self.pet[0]._datasets[p_idx].RescaleSlope +
                    self.pet[0]._datasets[p_idx].RescaleIntercept)

        s = np.concatenate((self.pet_spacing, [self.pet_slice_thickness]))
        pos_min = np.array(self.pet_img.shape[1:3])
        pos_max = np.array([0, 0])
        for i in range(len(self.rois)):
            pos = ((self.rois[i] - self.pet_origin) / s)[:, 0:2].astype(np.int)
            pos_max[0] = np.max([np.max(pos[:, 0]), pos_max[0]])
            pos_max[1] = np.max([np.max(pos[:, 1]), pos_max[1]])
            pos_min[0] = np.min([np.min(pos[:, 0]), pos_min[0]])
            pos_min[1] = np.min([np.min(pos[:, 1]), pos_min[1]])

        pos_min[0] = max(pos_min[0] - 2, 0)
        pos_min[1] = max(pos_min[1] - 2, 0)
        pos_max[0] = min(pos_max[0] + 2, masked_roi_imgs.shape[1])
        pos_max[1] = min(pos_max[1] + 2, masked_roi_imgs.shape[2])

        cropped_roi = np.array(masked_roi_imgs)[:, pos_min[1]:pos_max[1], :]
        cropped_roi = cropped_roi[:, :, pos_min[0]:pos_max[0]]

        return cropped_roi, masked_roi_imgs, co_pet_imgs, rois_co_slice_idx


    def __read_roi_data(self):
        """
        read roi data from contour dicom
        """
        self.rois = []
        self.sop_instance_uids_of_rois = []
        if self.target_ctr_sequence >= 0:
            self.__read_one_ctr_sequence(self.target_ctr_sequence)
        elif self.target_ctr_sequence == -1:
            for target in range(len(self.ctr.ROIContourSequence)):
                if hasattr(self.ctr.ROIContourSequence[target],
                           'ContourSequence'):
                    self.__read_one_ctr_sequence(target)
                else:
                    print('Cannot find ContourSequence at '
                          'target ctr idx {}'.format(target))
        else:
            raise Exception('Unknown target contour sequence number.'
                            '{} was given.'.format(self.target_ctr_sequence))

    def __read_one_ctr_sequence(self, target_ctr_sequence):
        """
        read target contour sequence
        :param target_ctr_sequence: target contour sequence
        """
        for i in range(
                len(self.ctr.ROIContourSequence[
                        target_ctr_sequence].ContourSequence)):
            roi = np.array(
                self.ctr.ROIContourSequence[
                    target_ctr_sequence].ContourSequence[i].ContourData)
            roi = roi.reshape((roi.size // 3, 3))
            if not (self.ctr.ROIContourSequence[
                        target_ctr_sequence].ContourSequence[
                        i].ContourGeometricType == "CLOSED_PLANAR"):
                raise Exception(
                    "{} is unsupported contour geometric type...".format(
                        self.ctr.ROIContourSequence[
                            target_ctr_sequence].ContourSequence[
                            i].ContourGeometricType))
            self.rois.append(roi)
            self.sop_instance_uids_of_rois.append(
                self.ctr.ROIContourSequence[
                    target_ctr_sequence].ContourSequence[
                    i].ContourImageSequence[0].ReferencedSOPInstanceUID)

    def save_voi(self, dirname, output_voi_figs=False):
        """
        save extracted roi
        :param dirname: directory name for save
        :param output_voi_figs: if True, voi extraction result figures are saved
        """

        filename = '{}/{}_{}_ctr{:02d}'.format(
            dirname,
            self.ctr.PatientName,
            self.ctr.SeriesDescription,
            self.target_ctr_sequence).replace(" ", "_")

        np.save(filename + '.npy', self.masked_roi_imgs_suv)
        np.save(filename + '_non_cropped_.npy', self.non_cropped_masked_roi_imgs * self.suv_coeff)
        np.save(filename + '_non_cropped_non_masked_.npy', self.co_pet_imgs * self.suv_coeff)
        roi_volume = (self.masked_roi_imgs[self.masked_roi_imgs >= 0].size *
                      self.pet_slice_thickness * np.prod(self.pet_spacing) /
                      1000)
        radiopharmaceutical = \
            self.pet[0].info.RadiopharmaceuticalInformationSequence[
                0].Radiopharmaceutical
        json_data = {
            'radiopharmaceutical': radiopharmaceutical,
            'roi_volume': roi_volume,
            'patient_name': str(self.ctr.PatientName),
            'series_description': str(self.ctr.SeriesDescription),
            'target_ctr_idx': self.target_ctr_sequence,
            'n_voxels': self.masked_roi_imgs[self.masked_roi_imgs >= 0].size,
            'ref_roi_number': -1 if self.target_ctr_sequence == -1 else (
                self.ctr.ROIContourSequence[
                    self.target_ctr_sequence].ReferencedROINumber),
            'SUV_conversion_coeff': self.suv_coeff,
            'n_pet_slices': self.pet_img.shape[0],
            'pixel_spacing': list(self.pet[0].info.PixelSpacing),
            'slice_thickness': self.pet[0].info.SliceThickness,
        }
        with open(filename + '_meta_data_.json', 'w') as f:
            json.dump(json_data, f, sort_keys=True, indent=2)

        if output_voi_figs:
            self.__save_voi_fig(filename)

    def __save_voi_fig(self, filename):
        """
        save extraction result figures
        """
        my_cmap = cm.viridis
        my_cmap.set_under(alpha=0)
        figs = [plt.figure(figsize=(10, 10)) for i in
                range(self.non_cropped_masked_roi_imgs.shape[0])]
        axes = [fig.add_subplot(111) for fig in figs]
        vmin = max(0, self.non_cropped_masked_roi_imgs[
            self.non_cropped_masked_roi_imgs >= 0].min())
        vmax = self.non_cropped_masked_roi_imgs[
            self.non_cropped_masked_roi_imgs >= 0].max()
        for idx in np.unique(self.rois_co_slice_idxs):
            axes[idx].imshow(self.co_pet_imgs[idx],
                             vmin=self.co_pet_imgs[idx].min(),
                             vmax=self.co_pet_imgs[idx].max(),
                             cmap='gray')
            axes[idx].imshow(self.non_cropped_masked_roi_imgs[idx],
                             cmap=my_cmap,
                             vmin=vmin,
                             vmax=vmax,
                             alpha=0.5)
            axes[idx].axis('off')

        s = np.concatenate((self.pet_spacing, [self.pet_slice_thickness]))
        for i, roi in enumerate(self.rois):
            pos = ((roi - self.pet_origin) / s)[:, 0:2]
            pos = np.vstack((pos, [pos[0]]))
            slice_idx = self.rois_co_slice_idxs[i]
            axes[slice_idx].plot(np.round(pos[:, 0] - 0.5) + 0.5,
                                 np.round(pos[:, 1] - 0.5) + 0.5,
                                 '-', color='#00FF00')

        for i, fig in enumerate(figs):
            fig.savefig(filename + '_{:02d}_.png'.format(i),
                        figsize=(20, 20))
            fig.clf()

    def __get_SUV_conversion_coeff(self):
        """
        return conversion coefficient to convert intensity to SUV
        :return conversion coefficient
        """

        p = self.pet[0].info

        scan_time_str = str(p[0x0008, 0x0021].value) + str(
            p[0x0008, 0x0031].value)
        content_time_str = str(p[0x0008, 0x0023].value) + str(
            p[0x0008, 0x0033].value)
        scan_time = datetime.datetime.strptime(scan_time_str, '%Y%m%d%H%M%S')
        content_time = datetime.datetime.strptime(content_time_str[0:14],
                                                  '%Y%m%d%H%M%S')
        delta = content_time - scan_time
        if delta.total_seconds() <= 0:
            raise Exception("Error: Series Date/Time is not correct.")
        if len(p.RadiopharmaceuticalInformationSequence) > 1:
            raise Exception('More than two radiopharmaceutical'
                            ' information was detected')

        t_half = float(
            p.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife)
        if hasattr(p.RadiopharmaceuticalInformationSequence[0],
                   "RadiopharmaceuticalStartDateTime"):
            measured_time_str = str(p.RadiopharmaceuticalInformationSequence[
                                        0].RadiopharmaceuticalStartDateTime)
        elif hasattr(p.RadiopharmaceuticalInformationSequence[0],
                   "RadiopharmaceuticalStartTime"):
            print("Warning!!!\nRadiopharmaceuticalInformationSequence only "
                  "has RadiopharmaceuticalStartTime tag. Assume the date is "
                  "same as SeriesDate.\n"
                  "You must carefully check if SUV value is correct.")
            measured_time_str = \
                str(p[0x0008, 0x0021].value) + \
                str(p.RadiopharmaceuticalInformationSequence[0
                    ].RadiopharmaceuticalStartTime)
        else:
            raise Exception("Error: This dicom file dose not include "
                            "RadiopharmaceuticalStartDateTime data")
        measured_time = datetime.datetime.strptime(measured_time_str,
                                                   '%Y%m%d%H%M%S.%f')
        activity = float(
            p.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose)
        actual_activity = activity * (
            2 ** (-(scan_time - measured_time).total_seconds() / t_half))

        return p.PatientWeight * 1000. / actual_activity



    def __get_corresponding_image(self, target_roi_index):
        """
        return corresponding PET/CT iamges and those origin (mm)  
        :param target_roi_index: target roi index
        """

        for i in range(len(self.ct[0]._datasets)):
            uid_c = self.sop_instance_uids_of_rois[target_roi_index]
            if self.ct[0]._datasets[i].SOPInstanceUID == uid_c:
                img_c = np.array(
                    self.ct[0]._datasets[i].pixel_array * self.ct[
                        0]._datasets[i].RescaleSlope + self.ct[
                        0]._datasets[i].RescaleIntercept)
                org_c = np.array(self.ct[0]._datasets[i].ImagePositionPatient)

        ct_ctr_img_seq = copy.deepcopy(
            self.ctr.ReferencedFrameOfReferenceSequence[
                0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[
                0].ContourImageSequence)
        pet_ctr_img_seq = copy.deepcopy(
            self.ctr.ReferencedFrameOfReferenceSequence[
                1].RTReferencedStudySequence[0].RTReferencedSeriesSequence[
                0].ContourImageSequence)

        if not ct_ctr_img_seq[0].ReferencedSOPClassUID == 'CT Image Storage':
            if pet_ctr_img_seq[0].ReferencedSOPClassUID == 'CT Image Storage':
                if (ct_ctr_img_seq[0].ReferencedSOPClassUID ==
                        'Positron Emission Tomography Image Storage'):
                    tmp = copy.deepcopy(ct_ctr_img_seq)
                    ct_ctr_img_seq = copy.deepcopy(pet_ctr_img_seq)
                    pet_ctr_img_seq = copy.deepcopy(tmp)
                else:
                    raise Exception('Cannot find contour data for the PET image')
            else:
                raise Exception('Cannot find contour data for the CT image')

        for i in range(len(ct_ctr_img_seq)):
            if uid_c == ct_ctr_img_seq[i].ReferencedSOPInstanceUID:
                uid_p = pet_ctr_img_seq[i].ReferencedSOPInstanceUID
        idx_p = []
        img_p = []
        org_p = []
        for i in range(len(self.pet[0]._datasets)):
            if self.pet[0]._datasets[i].SOPInstanceUID == uid_p:
                idx_p.append(i)
                img_p.append(np.array(
                    self.pet[0]._datasets[i].pixel_array *
                    self.pet[0]._datasets[i].RescaleSlope +
                    self.pet[0]._datasets[i].RescaleIntercept))
                org_p.append(np.array(
                    self.pet[0]._datasets[i].ImagePositionPatient))
        if not len(idx_p) == 1:
            if len(idx_p) == 0:
                raise Exception(
                    'Cannot find corresponding image for '
                    'the target roi index {}.'.format(target_roi_index))
            else:
                raise Exception(
                    'Multiple pet images was detected as the corresponding '
                    'frame of the target roi index {}'.format(target_roi_index)
                )
        return idx_p[0], img_p[0], org_p[0], img_c, org_c


    def __mask_outside(self, pet_img_2d, roi):
        """
        return masked pet image. outside of roi is filled with -1.
        :param pet_img_2d: a slice of pet images correspond to the roi
        :param roi: array which contains roi vertex positions
        :return: masked pet image
        """
        s = np.concatenate((self.pet_spacing, [self.pet_slice_thickness]))
        points = []
        for i in range(len(roi)):
            points.append((roi[i] - self.pet_origin) / s)
        points = np.array(points)
        points = np.round(points[:, 0:2] - 0.5, decimals=0) + 0.5

        xmin = max(0, points[:, 0].min()-3)
        xmax = min(pet_img_2d.shape[1], points[:, 0].max()+3)
        ymin = max(0, points[:, 1].min()-3)
        ymax = min(pet_img_2d.shape[0], points[:, 1].max()+3)

        x, y = np.meshgrid(np.arange(int(xmin), int(xmax)),
                           np.arange(int(ymin), int(ymax)))
        x = x.flatten()[:, np.newaxis]
        y = y.flatten()[:, np.newaxis]
        z0 = np.hstack((x, y)).astype(int)

        z1 = np.array(points)
        not_duplication_mask = np.ones(z1.shape[0], dtype=bool)
        for i in range(z1.shape[0] - 1):
            if np.all(z1[i] == z1[i + 1]):
                not_duplication_mask[i+1] = False
        z1 = z1[not_duplication_mask]
        if not np.all(z1[0] == z1[-1]):
            z1 = np.vstack((z1, [z1[0]]))
        img = is_inside(z0, z1, pet_img_2d)

        return img


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Voi extraction tool')
    parser.add_argument('--data_dir_path', '-d', type=str,
                        default='./data/PA*',
                        help='Directory of target data files.')
    args = parser.parse_args()

    for pa_index in range(len(glob.glob(args.data_dir_path))):
        try:
            target_dir = args.data_dir_path.replace('*', '%d' % pa_index)
            target_dir = os.path.join(target_dir, 'ST0')
            print(target_dir)
            pet = pydicom_series.read_files('{}/SE0/'.format(target_dir))
            ct = pydicom_series.read_files('{}/SE1/'.format(target_dir))
            # check if pet and ct are not flipped
            if not hasattr(pet[0].info,
                           'RadiopharmaceuticalInformationSequence'):
                begin = time.time()
                # tmp_pet = copy.deepcopy(pet)
                # pet = copy.deepcopy(ct)
                # ct = copy.deepcopy(tmp_pet)
                tmp_pet = pet
                pet = ct
                ct = tmp_pet
                print('FLIP! {:.3f}s'.format(time.time()-begin))

            ctrs = []
            n = len(glob.glob(os.path.join(target_dir, 'SE*')))
            for i in range(2, n):
                ctr_dcm_fname = '{}/SE{}/IM0'.format(target_dir, i)
                if os.path.exists(ctr_dcm_fname):
                    print('Load contour: {}'.format(ctr_dcm_fname))
                    tmp_ctr = dicom.read_file(ctr_dcm_fname)
                    if hasattr(tmp_ctr, 'ROIContourSequence'):
                        ctrs.append(tmp_ctr)
                else:
                    print('Not found: {}'.format(ctr_dcm_fname))

            for i, ctr in enumerate(ctrs):
                for idx in range(len(ctr.ROIContourSequence)):
                    try:
                        manager = VoiExtractionManager(pet, ct, ctr, idx)
                        manager.save_voi(target_dir, output_voi_figs=True)
                    except:
                        print(traceback.format_exc())
                        print('Error: ctr_id {}', i)
                if len(ctr.ROIContourSequence) > 1:
                    try:
                        manager = VoiExtractionManager(pet, ct, ctr, -1)
                        manager.save_voi(target_dir, output_voi_figs=True)
                    except:
                        print(traceback.format_exc())
                        print('Error: ctr_id {}', i)

        except:
            print(traceback.format_exc())
            print('Error:', target_dir)

