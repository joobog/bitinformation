#!/usr/bin/env python3

import numpy as np

class Stats:
    @staticmethod
    def l_1_norm(v1, v2):
        assert v1.size == v2.size
        diff = np.absolute(v1 - v2)
        vsum = np.sum(diff)
        return vsum

    @staticmethod
    def l_inf(v1, v2):
        assert v1.size == v2.size
        return np.max(np.absolute(v1 - v2))

    @staticmethod
    def rmse(actual, pred):
        assert actual.size == pred.size
        return np.sqrt(np.sum(np.square(actual - pred)))

    @staticmethod
    def rrmse(actual, pred):
        '''relative root mean squared error
        * Excellent when RRMSE < 10%
        * Good when RRMSE is between 10% and 20%
        * Fair when RRMSE is between 20% and 30%
        * Poor when RRMSE > 30%'''
        assert actual.size == pred.size
        num = np.sum(np.square(actual - pred)) / actual.size
        den = np.sum(np.square(pred))
        squared_error = num/den
        rrmse_loss = np.sqrt(squared_error)
        return rrmse_loss

    @staticmethod
    def maxre(actual, pred):
        ''' Relative error '''
        assert actual.size == pred.size
        diff = actual.max() - actual.min()
        rel_er = np.absolute(actual - pred).max() / diff
        return rel_er
