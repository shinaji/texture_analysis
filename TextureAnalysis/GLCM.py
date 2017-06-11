#-*- coding:utf-8 -*-
"""
    GLCM 

    Copyright (c) 2016 Tetsuya Shinaji

    This software is released under the MIT License.

    http://opensource.org/licenses/mit-license.php
    
    Date: 2016/01/29

"""

import numpy as np
from matplotlib import pyplot as plt
from TextureAnalysis.Utils import normalize


class GLCM:
    """
    Gray-Level Co-occurrence Matrix
    """
    def __init__(self, img, theta=[0, 1], level_min=1, level_max=256, threshold=None):
        """
        initialize

        :param img: normalized image
        :param theta: definition of neighbor
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
        self.theta = theta
        self.matrix = self._construct_matrix()
        self.features = self._calc_features()

    def _calc_features(self):
        """
        calculate feature values

        :return: feature values
        """

        I, J = np.ogrid[self.level_min:self.level_max+1,
                        self.level_min:self.level_max+1]
        mat = np.array(self.matrix)
        features = {}
        features['uniformity'] = (mat**2).sum()
        features['entropy'] = -(mat[mat>0]*np.log(mat[mat>0])).sum()
        features['dissimilarity'] = (mat*np.abs(I-J)).sum()
        features['contrast'] = (mat*((I-J)**2)).sum()
        features['homogeneity'] = (mat/(1+np.abs(I-J))).sum()
        features['inverse difference moment'] = (mat/(1+(I-J)**2)).sum()
        features['maximum probability'] = mat.max()*100
        return features


    def _construct_matrix(self):
        """
        construct GLC-Matrix

        :return: GLC-Matrix
        """

        mat = np.zeros((self.n_level, self.n_level)).astype(np.float)
        unique = np.unique(self.img)
        width = self.img.shape[1]
        height = self.img.shape[0]
        print(self.img.max())
        print(self.img.min())
        for uni in unique:
            if uni < self.level_min:
                continue
            indices = np.argwhere(self.img == uni)
            for idx in indices:
                pos = np.array(idx + self.theta, dtype=np.int32)
                if 0 <= pos[0] < height and 0 <= pos[1] < width:
                    neighbor_value = self.img[pos[0], pos[1]]
                    if neighbor_value >= self.level_min:
                        mat[self.img[idx[0], idx[1]]-self.level_min,
                            self.img[pos[0], pos[1]]-self.level_min] += 1
                pos = idx + (self.theta*-1)
                if 0 <= pos[0] < height and 0 <= pos[1] < width:
                    neighbor_value = self.img[pos[0], pos[1]]
                    if neighbor_value >= self.level_min:
                        mat[self.img[idx[0], idx[1]]-self.level_min,
                            self.img[pos[0], pos[1]]-self.level_min] += 1
        return mat


if __name__ == '__main__':

    suvs = np.load("suvs.npy")
    mats = []
    for i in range(suvs.shape[0]):
        glcm = GLCM(suvs[i], theta=np.array([0, 1]),
                    level_min=1, level_max=32,
                    threshold=suvs.max()*0.5)
        mats.append(glcm.matrix)
    mats = np.array(mats)
    print(mats)
    print(mats.shape)