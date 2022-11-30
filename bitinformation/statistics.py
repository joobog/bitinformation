#!/usr/bin/env python3

import math
import numpy as np

class Stats:
    @staticmethod
    def l_1_norm(v1, v2):
        assert(v1.size == v2.size)
        diff = np.absolute(v1 - v2)
        sum = np.sum(diff)
        return sum

    @staticmethod
    def l_inf(v1, v2):
        assert(v1.size == v2.size)
        return np.max(np.absolute(v1 - v2))

    @staticmethod
    def rmse(v1, v2):
        print(v1.shape, v2.shape)
        assert(v1.size == v2.size)
        tmp = v1-v2
        return np.sqrt(np.sum(tmp*tmp))

    @staticmethod
    def rrmse(v1, v2):
        assert(v1.size == v2.size)
        '''relative root mean squared error
        * Excellent when RRMSE < 10%
        * Good when RRMSE is between 10% and 20%
        * Fair when RRMSE is between 20% and 30%
        * Poor when RRMSE > 30%'''

        num = np.sum(np.square(v1 - v2))
        den = np.sum(np.square(v2))
        squared_error = num/den
        rrmse_loss = np.sqrt(squared_error)
        return rrmse_loss
