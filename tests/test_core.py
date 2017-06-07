# -*- coding: utf-8 -*-

from nose.tools import with_setup, raises, ok_, timed
import os
import copy
import numpy as np
from .context import TextureAnalysis


class TestCore:

    simple_test_data = np.zeros((4, 4, 4), dtype=np.float64)
    simple_test_data[0] = -1
    simple_test_data[1] = 1
    simple_test_data[2] = 2
    simple_test_data[3] = 3

    glcm = TextureAnalysis.GLCM_3D(simple_test_data, d=1,
                                   level_min=0, level_max=2, threshold=1)
    glha = TextureAnalysis.GLHA(simple_test_data,
                                level_min=0, level_max=2, threshold=1)
    glszm = TextureAnalysis.GLSZM_3D(simple_test_data,
                                     level_min=1, level_max=3, threshold=1)
    ngtdm = TextureAnalysis.NGTDM_3D(simple_test_data, d=1,
                                     level_min=1, level_max=3, threshold=1)

    @classmethod
    def setup_class(clazz):
        print('Test start!')

    @classmethod
    def teardown_class(clazz):
        print('Done!')

    def setup(self):
        pass

    def teardown(self):
        pass

    def test_glcm(self):
        assert np.all(self.glcm.matrix_non_norm ==
                      np.array([[84, 100, 0],
                                [100, 84, 100],
                                [0, 100, 84]])), \
            'Constructed matrix is not correct.'
