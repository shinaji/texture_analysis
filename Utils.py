# -*- coding:utf-8 -*-
"""
    Utils.py

    Copyright (c) 2017 Tetsuya Shinaji

    This software is released under the MIT License.

    http://opensource.org/licenses/mit-license.php

    Date: 2017/01/03

    The major design of routine for performing the "point in polygon" inclusion
    was performed by softSurfer.
    The softSurfer's code was translated to Python by Maciej Kalisiak.
    Here is the original copyright notice.

    Copyright 2001, softSurfer (www.softsurfer.com)
    This code may be freely used and modified for any purpose
    providing that this copyright notice is included with it.
    SoftSurfer makes no warranty for this code, and cannot be held
    liable for any real or imagined damage resulting from its use.
    Users of this code must verify correctness for their application.

    translated to Python by Maciej Kalisiak <mac@dgp.toronto.edu>

    http://www.dgp.toronto.edu/~mac/e-stuff/point_in_polygon.py

"""
import numpy as np


def is_inside(z0, z1, pet_img):
    """
    check if z0 is within z1 polygon
    :param z0: point
    :param z1: vertex positions of roi polygon
    :param pet_img: pet image
    :return:
    """
    tmp = np.ones_like(pet_img) * -1
    for i in range(len(z0)):
        if __get_wn(z0[i], z1):
            tmp[z0[i][1], z0[i][0]] = pet_img[z0[i][1], z0[i][0]]
    return tmp


def __is_left(P0, P1, P2):
    """
    check if P2 is left of the vector P0->P1
    :param P0: vertex position of polygon
    :param P1: vertex position of polygon next to p1
    :param P2: target position for this check process
    """
    return (P1[0] - P0[0]) * (P2[1] - P0[1]) - (P2[0] - P0[0]) * (P1[1] - P0[1])


def __get_wn(z0, z1):
    """
    retrun wn value of the Winding Number Algorithm
    :param z0: target position for this check process
    :param z1: vertex positions of polygon
    :return: wn
    """

    wn = 0
    for i in range(z1.shape[0] - 1):

        v1 = z1[i + 1] - z1[i]
        v2 = z1[i + 1] - z0
        lv1 = np.linalg.norm(v1)
        lv2 = np.linalg.norm(v2)
        if lv2 == 0:
            return 1
        if (np.abs(np.dot(v1 / lv1, v2 / lv2) - 1) <= 1e-5) and (
                        lv1 >= lv2) and (lv1 > np.linalg.norm(v1 + v2)):
            return 1

        if z1[i, 1] <= z0[1]:  # start y <= P[1]
            if z1[i + 1, 1] > z0[1]:  # an upward crossing
                if __is_left(z1[i], z1[i + 1], z0) > 0:  # P left of edge
                    wn += 1  # have a valid up intersect
        else:  # start y > P[1] (no test needed)
            if z1[i + 1, 1] <= z0[1]:  # a downward crossing
                if __is_left(z1[i], z1[i + 1], z0) < 0:  # P right of edge
                    wn -= 1  # have a valid down intersect
    return wn
