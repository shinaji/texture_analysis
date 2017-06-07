#-*- coding:utf-8 -*-
"""
    Utills 

    Copyright (c) 2016 Tetsuya Shinaji

    This software is released under the MIT License.

    http://opensource.org/licenses/mit-license.php
    
    Date: 2016/01/29

"""

import numpy as np
from matplotlib import pyplot as plt


def normalize(img, level_min=1, level_max=256, threshold=None):
    """
    normalize the given image

    :param img: image array
    :param level_min: min intensity of normalized image
    :param level_max: max intensity of normalized image
    :param threshold: threshold of the minimal value

    :return: normalized image array, slope, intercept
    """

    tmp_img = np.array(img)
    if threshold is None:
        threshold = tmp_img.min()
    tmp_img[tmp_img<threshold] = -1
    assert level_min < level_max, "level_min must be smaller than level_max"
    slope = (level_max - level_min) / (img.max() - threshold)
    intercept = - threshold * slope
    tmp_img = tmp_img * slope + intercept + level_min
    return np.round(tmp_img, decimals=0).astype(np.int32), slope, intercept
