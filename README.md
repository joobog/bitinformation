# Bit-Information-Content Tool

[![publish](https://github.com/joobog/bitinformation/actions/workflows/publish.yaml/badge.svg)](https://github.com/joobog/bitinformation/actions/workflows/publish.yaml)

[![test](https://github.com/joobog/bitinformation/actions/workflows/ci.yaml/badge.svg)](https://github.com/joobog/bitinformation/actions/workflows/ci.yaml)

## Install

    python3 -m pip install bitinformation

## Examples

### Compute bit information

    import numpy as np
    import bitinformation.bitinformation as bit
    data = np.random.rand(10000)  
    bi = bit.BitInformation()
    bi.bitinformation(data)

### Compare data

    import numpy as np
    import bitinformation.bitinformation as bit
    data1 = np.random.rand(10000)  
    data2 = np.random.rand(10000)  
    res = bit.compare_data(data1, data2)

### Compare GRIB files

    import numpy as np
    import bitinformation.bitinformation as bit
    fn1 = "grib.grib"
    res = bit.compare_data(data1, data2)

This tool is based on work by Klöwer et. al:
<https://github.com/milankl/BitInformation.jl>
