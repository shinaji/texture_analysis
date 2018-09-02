#-*- coding:utf-8 -*-
"""
    NGTDM 

    Copyright (c) 2016 Tetsuya Shinaji

    This software is released under the MIT License.

    http://opensource.org/licenses/mit-license.php
    
    Date: 2016/01/31

"""

import numpy as np
from matplotlib import pyplot as plt
from TextureAnalysis.Utils import normalize
from scipy import signal

class NGTDM:
    """
    Neighbourhood Gray-Tone-Difference Matrix
    """
    def __init__(self, img, d=1, level_min=1, level_max=256, threshold=None):
        """
        initialize

        :param img: normalized image
        :param d: distance
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
        self.d = d
        self.s, self.p, self.ng, self.n2 = self._construct_matrix()
        self.features = self._calc_features()

    def _calc_features(self):
        """
        calculate feature values

        :return: feature values
        """
        features = {}
        I, J = np.ogrid[self.level_min:self.level_max+1,
                        self.level_min:self.level_max+1]
        pi = np.hstack((self.p[:, np.newaxis],)*len(self.p))
        pj = np.vstack((self.p[np.newaxis, :],)*len(self.p))
        # ipi = pi * np.hstack((np.arange(len(self.p))[:, np.newaxis],)*len(self.p))
        # jpj = pj * np.vstack((np.arange(len(self.p))[np.newaxis, :],)*len(self.p))
        # ipi = self.p * np.arange(self.level_min, self.level_max + 1)[:,np.newaxis]
        # jpj = self.p * np.arange(self.level_min, self.level_max + 1)[np.newaxis,:]
        ipi = np.hstack(
            ((self.p*np.arange(1, len(self.p)+1))[:, np.newaxis],) * len(self.p))
        jpj = np.vstack(
            ((self.p*np.arange(1, len(self.p)+1))[np.newaxis, :],) * len(self.p))
        pisi = pi * np.hstack((self.s[:, np.newaxis],)*len(self.p))
        pjsj = pj * np.vstack((self.s[np.newaxis, :],)*len(self.p))
        fcos = 1.0 / (1e-6 + (self.p*self.s).sum())
        fcon = 1.0 / (self.ng*(self.ng-1)) * (pi*pj*(I-J)**2).sum() * (self.s.sum()/self.n2)
        mask1 = np.logical_and(pi > 0, pj > 0)
        mask2 = self.p > 0
        if (np.abs(ipi[mask1] - jpj[mask1])).sum() == 0:
            fbus = np.inf
        else:
            fbus = (self.p*self.s)[mask2].sum() / (np.abs(ipi[mask1] - jpj[mask1])).sum()
        fcom = (np.abs(I-J)[mask1] / (self.n2*(pi+pj)[mask1]) * (pisi + pjsj)[mask1]).sum()
        fstr = ((pi + pj) * (I-J)**2).sum() / (1e-6+self.s.sum())
        features['coarseness'] = fcos
        features['contrast'] = fcon
        features['busyness'] = fbus
        features['complexity'] = fcom
        features['strength'] = fstr
        return features

    def _construct_matrix(self):
        """
        construct NGTD-Matrix

        :return: NGTD-Matrix
        """

        assert self.d > 0, 'd must be grater than 1'
        assert self.level_min > 0, 'lower level must be greater than 0'
        # w = (2 * self.d + 1)**2
        kernel = np.ones((2*self.d+1, 2*self.d+1))
        kernel[self.d, self.d] = 0
        h, w = self.img.shape
        A = signal.convolve2d(self.img, kernel, mode='valid')
        A *= (1/((2 * self.d + 1)**2-1))
        s = np.zeros(self.n_level)
        p = np.zeros_like(s)
        crop_img = np.array(self.img[self.d:h-self.d, self.d:w-self.d])
        for i in range(self.level_min, self.level_max+1):
            indices = np.argwhere(crop_img == i)
            s[i-self.level_min] = np.abs(i - A[indices[:, 0], indices[:, 1]]).sum()
            p[i-self.level_min] = float(len(indices)) / np.prod(crop_img.shape)
        ng = np.sum(np.unique(crop_img)>=0)
        n2 = np.prod(crop_img.shape)
        return s, p, ng, n2


if __name__ == '__main__':
    pass