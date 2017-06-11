#-*- coding:utf-8 -*-
"""
    GLHA 

    Copyright (c) 2016 Tetsuya Shinaji

    This software is released under the MIT License.

    http://opensource.org/licenses/mit-license.php
    
    Date: 2016/01/29

"""

import numpy as np
from matplotlib import pyplot as plt
from TextureAnalysis.Utils import normalize

class GLHA:
    """
    Gray Level Histogram Analysis
    """
    def __init__(self, img, level_min=1, level_max=256, threshold=None):
        """
        initialize

        :param img: normalized image
        :param level_min: min intensity of normalized image
        :param level_max: max intensity of normalized image
        :param threshold: threshold of the minimal value
        """

        self.img, self.slope, self.intercept = \
            normalize(img, level_min, level_max, threshold)
        self.n_level = (level_max - level_min) + 1
        self.level_min = level_min
        self.level_max = level_max

        hist, bin_edges = np.histogram(self.img.flatten(),
                                       bins=self.n_level,
                                       range=[level_min, level_max],
                                       density=False)
        self.hist = np.array(hist)
        hist, bin_edges = np.histogram(self.img.flatten(),
                                       bins=self.n_level,
                                       range=[level_min, level_max],
                                       density=True)
        self.bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        self.p_glha = np.array(hist)
        self.features = self._calc_features()

    def _calc_features(self):
        """
        calculate feature values

        :return: feature values
        """

        features = {}
        x = np.arange(self.level_min, self.level_max+1)
        features['mean'] = (self.p_glha*x).sum()
        features['sd'] = np.sqrt((self.p_glha*(x-features['mean'])**2).sum())
        features['skewness'] = \
            (self.p_glha*(x-features['mean'])**3).sum() / features['sd']**3
        features['kurtosis'] = \
            (self.p_glha*(x-features['mean'])**4).sum() / features['sd']**4

        return features

    def print_features(self, show_figure=False):
        """
        print features

        :param show_figure: if True, show figure
        """

        print("----GLHA-----")
        feature_labels = []
        feature_values = []
        for key in sorted(self.features.keys()):
            print("{}: {}".format(key, self.features[key]))
            feature_labels.append(key)
            feature_values.append(self.features[key])

        if show_figure:
            plt.plot(self.bin_centers, self.p_glha, 'o-b', label='Dencity')
            plt.plot([self.features['mean'], self.features['mean']],
                     [0, self.p_glha.max()*1.2], '-r', label='mean')
            plt.plot([self.features['mean']-self.features['sd'],
                      self.features['mean']-self.features['sd']],
                     [0, self.p_glha.max()*1.2], '-.r', label='sd lower')
            plt.plot([self.features['mean']+self.features['sd'],
                      self.features['mean']+self.features['sd']],
                     [0, self.p_glha.max()*1.2], '-.r', label='sd upper')
            plt.legend(loc=0, numpoints=1)
            plt.ylim(0, self.p_glha.max()*1.2)
            plt.show()

        return feature_labels, feature_values

if __name__ == '__main__':
    pass