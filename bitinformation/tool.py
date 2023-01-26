import numpy as np
import pandas as pd

import time
from datetime import datetime
from eccodes import *

from preprocessor import *
from analyser import *
from statistics import *

class Tool:
    def __init__(self):
        self.__timestamp = datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M')
        self.__preprocessor_cache = dict()

    def __collect_metadata(self, handle, data, preprocessor, info):
        metadata = dict()
        metadata['timestamp'] = self.__timestamp
        metadata['nbits_total'] = data.itemsize * 8
        metadata['nbits_value'] = preprocessor.bits_per_value()
        metadata['nbits_used'] = info.nbits_used - (metadata['nbits_total'] - metadata['nbits_value'])
        metadata['preprocessor'] = preprocessor.name
        metadata['const'] = codes_get_long(handle, 'isConstant')
        metadata['short_name'] = codes_get_string(handle, 'shortName')
        # metadata['param_id'] = codes_get_string(handle, 'paramId')
        # metadata['nvalues'] = size
        # metadata['nvalues_total'] = nvalues_total
        # metadata['rmse'] = Stats.rmse(data, optimized_data)
        # metadata['rrmse'] = Stats.rrmse(data, optimized_data)
        # metadata['l_inf'] = Stats.l_inf(data, optimized_data)
        # metadata['l_1_norm'] = Stats.l_1_norm(data, optimized_data)
        kiter = codes_keys_iterator_new(handle, namespace='mars')
        while codes_keys_iterator_next(kiter):
            key = codes_keys_iterator_get_name(kiter)
            value = codes_get_string(handle, key)
            metadata[key] = value
            # print(key, value)
        codes_keys_iterator_delete(kiter)
        return metadata

    def __preprocess(self, values, preprocessor):
        try:
            data = preprocessor.convert(values)
        except ConstantFieldException:
            data = np.zeros(values.size, dtype=np.int16)
        except Underflow as e:
            print(f'Error: Underflow {e}')
            traceback.print_exc()
            sys.exit(1)
        return data

    def __get_info(self, handle, preprocessor, analyser):
        values = codes_get_values(handle)
        data = self.__preprocess(values, preprocessor)
        analysis = analyser.analyse(values)
        return  (values, data, analysis)

    def compare(self, handle1, handle2, preprocessor, analyser):
        exit_status = 0
        msg_count = 0;

        values1, preprocessed_values1, analysis2 = self.__get_info(handle1, preprocessor, analyser)
        values2, preprocessed_values2, analysis2 = self.__get_info(handle2, preprocessor, analyser)

        res = self.__parameters.lookup(stream, nbits, precision=precision)

        status = np.all(info1.masked_data == info2.masked_data)

        # if self.__verbose > 0:
        #     df = pd.DataFrame({'Value1':values1, 'Value2':values2, 'diff':(masked_data1 - masked_data2)}).reset_index()
        #     df['index'] += 1
        #     df = df[df['diff'] != 0.0]
        #     if not df.empty:
        #         print(df)
        #     stats = dict()
        #     stats['RMSE'] = Stats.rmse(values1, values2)
        #     stats['RRMSE'] = Stats.rrmse(values1, values2)
        #     stats['L-Inf'] = Stats.l_inf(values1, values2)
        #     stats['L-1-Norm'] = Stats.l_1_norm(values1, values2)
        #     stats['ndiff_values'] = df.shape[0]
        #     print(stats)


        return exit_status


    def stats(self, handle, preprocessor, analyser):
        values = codes_get_values(handle)

        try:
            data = preprocessor.convert(values)
        except ConstantFieldException:
            data = np.zeros(values.size, dtype=np.int16)
        except Underflow as e:
            print(f'Error: Underflow {e}')
            traceback.print_exc()
            sys.exit(1)

        # Analyse
        info = analyser.analyse(data=data) # bitinformation, nbits_used, masked_data, mask, equal
        metadata = self.__collect_metadata(handle=handle, data=data, preprocessor=preprocessor, info=info)
        return (metadata, info)
