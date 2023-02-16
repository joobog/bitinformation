import numpy as np
import pandas as pd

import time
from datetime import datetime
from eccodes import *

from preprocessor import *
from analyser import *
from statistics import *
from simple_packing import ConstantFieldException, Underflow
from statistics import Stats

class Tool:
    def __init__(self):
        self.__timestamp = datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M')
        self.__preprocessor_cache = dict()

    def __preprocess(self, values, preprocessor):
        try:
            data, params = preprocessor.encode(values)
        except ConstantFieldException:
            data = np.zeros(values.size, dtype=np.int16)
            params = dict()
        except Underflow as e:
            print(f'Error: Underflow {e}')
            raise
        return data, params

    # def __get_info(self, values, preprocessor, analyser):
    #     data = self.__preprocess(values, preprocessor)
    #     analysis = analyser.analyse(values)
    #     return data, analysis

    def __create_mask_from_metadata(self, md):
        T = Preprocessor.get_optimal_numpy_type(md['preprocessor_bits_per_value'])
        mask = T(0x0)
        for _ in range(0, md['preprocessor_bits_per_value']-md['nbits_used']):
            mask = mask << T(0x1) | T(0x1)
        mask = ~mask
        return mask

    def compare(self, values1, values2, md1, md2, preprocessor):
        assert md1 is not None
        assert md2 is not None
        mask1 = self.__create_mask_from_metadata(md1)
        mask2 = self.__create_mask_from_metadata(md2)
        # assert mask1 == mask2
        preprocessed_values1, _ = self.__preprocess(values1, preprocessor)
        preprocessed_values2, _ = self.__preprocess(values2, preprocessor)
        assert preprocessed_values1.itemsize == np.dtype(type(mask1)).itemsize
        assert preprocessed_values2.itemsize == np.dtype(type(mask2)).itemsize
        masked1 = preprocessed_values1 & mask1
        masked2 = preprocessed_values2 & mask2
        status = np.all(masked1 == masked2)
        return {'status': status, 'data1': masked1, 'data2': masked2}


    def compute_stats(self, values1, values2):
        stats = dict()
        stats['rrmse'] = Stats.rrmse(values1, values2)
        stats['rmse'] = Stats.rmse(values1, values2)
        stats['l_1_norm'] = Stats.l_1_norm(values1, values2)
        stats['l_inf'] = Stats.l_inf(values1, values2)
        stats['maxre'] = Stats.maxre(values1, values2)
        return stats

    def preprocess_and_analyse(self, values, preprocessor, analyser):
        # try:
        data, _ = preprocessor.encode(values)
        # except ConstantFieldException:
        #     data = np.zeros(values.size, dtype=np.int16)
        # except Underflow as e:
        #     print(f'Error: Underflow {e}')
        #     raise
        analysis = analyser.analyse(data=data) # bitinformation, nbits_used, masked_data, mask, equal
        return analysis
