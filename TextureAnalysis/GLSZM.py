#-*- coding:utf-8 -*-
"""
    GLSZM 

    Copyright (c) 2016 Tetsuya Shinaji

    This software is released under the MIT License.

    http://opensource.org/licenses/mit-license.php
    
    Date: 2016/01/31

"""

import numpy as np
from matplotlib import pyplot as plt
from TextureAnalysis.Utils import normalize
from scipy.ndimage import measurements


class GLSZM:
    """
    Gray Level Size Zone Matrix
    """
    def __init__(self, img, level_min=1, level_max=256, threshold=None):
        """
        initialize

        :param img: normalized image
        :param level_min: min intensity of normalized image
        :param level_max: max intensity of normalized image
        :param threshold: threshold of the minimal value
        """

        assert len(img.shape) == 2, 'image must be 2D'
        self.img, self.slope, self.intercept = \
            normalize(img, level_min, level_max, threshold)
        self.n_level = (level_max - level_min) + 1
        self.level_min = level_min
        self.level_max = level_max
        self.matrix, self.zone_sizes  = self._construct_matrix()
        self.features = self._calc_features()

    def _calc_features(self):
        """
        calculate feature values

        :return: feature values
        """

        features ={}
        mat = self.matrix
        zone_sizes = self.zone_sizes

        omega = mat.flatten().sum()
        min_size = zone_sizes.min()
        max_size = zone_sizes.max()
        j = np.array(range(min_size, max_size+1))[np.newaxis, :]
        j = np.vstack((j,)*mat.shape[0])
        i = np.array(range(self.level_min, self.level_max+1))[:, np.newaxis]
        i = np.hstack((i,)*mat.shape[1])
        small_area_emp = (mat / (j**2)).sum() / omega
        large_area_emp = (mat * (j**2)).sum() / omega
        low_intensity_emp = (mat / (i**2)).sum() / omega
        high_intensity_emp = (mat * (i**2)).sum() / omega
        intensity_variability = ((mat / (i**2)).sum(axis=1) ** 2).sum() / omega
        size_zone_variability = ((mat / (j**2)).sum(axis=0) ** 2).sum() / omega #?
        zone_percentage = omega / (mat * (j**2)).sum()
        low_intensity_small_area_emp = (mat / (i**2) / (j**2)).sum() / omega
        high_intensity_small_area_emp = (mat * (i**2) * (j**2)).sum() / omega
        low_intensity_large_area_emp = (mat * (j**2) / (i**2)).sum() / omega
        high_intensity_large_area_emp = (mat * (i**2) / (j**2)).sum() / omega

        features['small_area_emp'] = small_area_emp
        features['large_area_emp'] = large_area_emp
        features['low_intensity_emp'] = low_intensity_emp
        features['high_intensity_emp'] = high_intensity_emp
        features['intensity_variability'] = intensity_variability
        features['size_zone_variability'] = size_zone_variability
        features['zone_percentage'] = zone_percentage
        features['low_intensity_small_area_emp'] = low_intensity_small_area_emp
        features['high_intensity_small_area_emp'] = high_intensity_small_area_emp
        features['low_intensity_large_area_emp'] = low_intensity_large_area_emp
        features['high_intensity_large_area_emp'] = high_intensity_large_area_emp


    def _construct_matrix(self):
        """
        construct GLSZ-Matrix

        :return: GLSZ-Matrix
        """
        s = [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]]
        elements = []
        for i in range(self.level_min, self.level_max+1):
            assert i >= 0, 'level mast be positivee value or 0.'
            tmp_img = np.array(self.img)
            tmp_img = (tmp_img == i)
            labeled_array, num_features = measurements.label(tmp_img,
                                                             structure=s)
            for label in range(1, num_features+1):
                size = (labeled_array.flatten() == label).sum()
                elements.append([i, size])

        elements = np.array(elements)
        min_element_size = elements[:, 1].min()
        rows = (self.level_max - self.level_min) + 1
        cols = elements[:, 1].max() - min_element_size + 1
        mat = np.zeros((rows, cols), dtype=np.float)
        zone_sizes = np.unique(elements[:, 1])
        for element in elements:
            mat[element[0], element[1]-min_element_size] += 1

        return mat, zone_sizes


if __name__ == '__main__':
    pass