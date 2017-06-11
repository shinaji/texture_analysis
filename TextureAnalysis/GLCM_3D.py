#-*- coding:utf-8 -*-
"""
    GLCM_3D

    Copyright (c) 2016 Tetsuya Shinaji

    This software is released under the MIT License.

    http://opensource.org/licenses/mit-license.php

    Date: 2016/01/29

"""

import numpy as np
from matplotlib import pyplot as plt
from TextureAnalysis.Utils import normalize


class GLCM_3D:
    """
    c
    """
    def __init__(self, img, d=1, level_min=1, level_max=256, threshold=None):
        """
        initialize

        :param img: normalized image
        :param d: distance from center
        :param level_min: min intensity of normalized image
        :param level_max: max intensity of normalized image
        :param threshold: threshold of the minimal value
        """

        assert len(img.shape) == 3, 'image must be 3D'
        self.img, self.slope, self.intercept = \
            normalize(img, level_min, level_max, threshold)
        self.n_level = (level_max - level_min) + 1
        self.level_min = level_min
        self.level_max = level_max
        self.d = d
        if self.d > 1:
            raise Exception("d>1 has not been implemented yet....")
        self.matrix_non_norm = self._construct_matrix()
        self.matrix = self.matrix_non_norm/self.matrix_non_norm.sum()
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

    def print_features(self, show_figure=False):
        """
        print features

        :param show_figure: if True, show figure
        """

        print("----GLCM 3D-----")
        feature_labels = []
        feature_values = []
        for key in sorted(self.features.keys()):
            print("{}: {}".format(key, self.features[key]))
            feature_labels.append(key)
            feature_values.append(self.features[key])

        if show_figure:
            plt.imshow(self.matrix,
                       origin='lower',
                       interpolation='none')
            plt.show()

        return feature_labels, feature_values


    def _construct_matrix(self):
        """
        construct GLC-Matrix

        :return: GLC-Matrix
        """

        mat = np.zeros((self.n_level, self.n_level)).astype(np.float)
        unique = np.unique(self.img)
        depth = self.img.shape[0]
        height = self.img.shape[1]
        width = self.img.shape[2]
        for uni in unique:
            if uni < self.level_min:
                continue
            indices = np.argwhere(self.img == uni)
            for idx in indices:
                positions = [
                    idx + np.array([-1, -1, -1]),
                    idx + np.array([-1, -1,  0]),
                    idx + np.array([-1, -1,  1]),
                    idx + np.array([-1,  0, -1]),
                    idx + np.array([-1,  0,  0]),
                    idx + np.array([-1,  0,  1]),
                    idx + np.array([-1,  1, -1]),
                    idx + np.array([-1,  1,  0]),
                    idx + np.array([-1,  1,  1]),
                    idx + np.array([ 0, -1, -1]),
                    idx + np.array([ 0, -1,  0]),
                    idx + np.array([ 0, -1,  1]),
                    idx + np.array([ 0,  0, -1]),
                    idx + np.array([ 0,  0,  1]),
                    idx + np.array([ 0,  1, -1]),
                    idx + np.array([ 0,  1,  0]),
                    idx + np.array([ 0,  1,  1]),
                    idx + np.array([ 1, -1, -1]),
                    idx + np.array([ 1, -1,  0]),
                    idx + np.array([ 1, -1,  1]),
                    idx + np.array([ 1,  0, -1]),
                    idx + np.array([ 1,  0,  0]),
                    idx + np.array([ 1,  0,  1]),
                    idx + np.array([ 1,  1, -1]),
                    idx + np.array([ 1,  1,  0]),
                    idx + np.array([ 1,  1,  1]),
                ]
                for pos in positions:
                    if 0 <= pos[0] < depth and 0 <= pos[1] < height and 0 <= pos[2] < width:
                        neighbor_value = self.img[pos[0], pos[1], pos[2]]
                        if neighbor_value >= self.level_min:
                            mat[self.img[idx[0], idx[1], idx[2]]-self.level_min,
                                self.img[pos[0], pos[1], pos[2]]-self.level_min] += 1
        return mat


if __name__ == '__main__':

    suvs = np.load("suvs.npy")
    mats = []
    for i in range(suvs.shape[0]):
        glcm = GLCM_3D(suvs, theta=np.array([0, 1]),
                    level_min=1, level_max=32,
                    threshold=suvs.max()*0.5)
        mats.append(glcm.matrix)
    mats = np.array(mats)
    print(mats)
    print(mats.shape)