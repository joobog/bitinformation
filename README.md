[![publish](https://github.com/joobog/bitinformation/actions/workflows/publish.yaml/badge.svg)](https://github.com/joobog/bitinformation/actions/workflows/publish.yaml)
[![test](https://github.com/joobog/bitinformation/actions/workflows/ci.yaml/badge.svg)](https://github.com/joobog/bitinformation/actions/workflows/ci.yaml)

# Bit-Information-Content Tool

The method calculates how much information content each bit in a number has.
In essence, it is a statistical analysis of bit sequences.
For example, according to this approach, random sequences of binary values and or a sequences of ones or zeros contain no information.
Once a sequence has a structure, the information content is non-zero.

    [0101010101010101] # low information content
    [1111111111111111] # zero information content
    [0000000000000000] # zero information content
    [0000111100001111] # high information content

## Algorithm
The following example explains the algorithm step by step without using formulas, when possible.

In the first step, assume there is a sequence `S` of 4-bit numbers. 
The sequence `S` is split into two arrays `A` and `B`. 
`A` is created by removing the last element from `S` and B, by removing the first element.
The example below uses Python notatation to illustrate that.

    S = [0, 1, 2, 3, 4, 5, 6, 7]

    A = S[:-1] = [0, 1, 2, 3, 4, 5, 6]
    B = S[1: ] = [1, 2, 3, 4, 5, 6, 7]


The next step is presented as a spreadsheet. 
In our example we work with 4-bit numbers, so we can identify each bit with the index `i = [0, 1, 2, 3]`.
For illustration, we exapand our table with `i` and `A`, and `i` and `B`.
`A'` and `B'` are the binary representations of the columns `A` and `B`, respectively.
The columns `A'[i]` and `B'[i]` are the bits at the position `i`.



 | i | A | B | A'=bin(A) | B'=bin(B) | A'[i] | B'[i] | seq = A'[idx]B'[idx] | 
 | - | - | - | -         | -         | -     | -     | -                    | 
 | 0 | 0 | 1 | 0000      | 0001      | 0     | 1     | 01                   | 
 | 0 | 1 | 2 | 0001      | 0010      | 1     | 0     | 10                   | 
 | 0 | 2 | 3 | 0010      | 0011      | 0     | 1     | 01                   | 
 | 0 | 3 | 4 | 0011      | 0100      | 1     | 0     | 10                   | 
 | 0 | 4 | 5 | 0100      | 0101      | 0     | 1     | 01                   | 
 | 0 | 5 | 6 | 0101      | 0110      | 1     | 0     | 10                   | 
 | 0 | 6 | 7 | 0110      | 0111      | 0     | 1     | 01                   | 
 | 1 | 0 | 1 | 0000      | 0001      | 0     | 0     | 00                   | 
 | 1 | 1 | 2 | 0001      | 0010      | 0     | 1     | 01                   | 
 | 1 | 2 | 3 | 0010      | 0011      | 1     | 1     | 11                   | 
 | 1 | 3 | 4 | 0011      | 0100      | 1     | 0     | 10                   | 
 | 1 | 4 | 5 | 0100      | 0101      | 0     | 0     | 00                   | 
 | 1 | 5 | 6 | 0101      | 0110      | 0     | 1     | 01                   | 
 | 1 | 6 | 7 | 0110      | 0111      | 1     | 1     | 11                   | 
 | 2 | 0 | 1 | 0000      | 0001      | 0     | 0     | 00                   | 
 | 2 | 1 | 2 | 0001      | 0010      | 0     | 0     | 00                   | 
 | 2 | 2 | 3 | 0010      | 0011      | 0     | 0     | 00                   | 
 | 2 | 3 | 4 | 0011      | 0100      | 0     | 1     | 01                   | 
 | 2 | 4 | 5 | 0100      | 0101      | 1     | 1     | 11                   | 
 | 2 | 5 | 6 | 0101      | 0110      | 1     | 1     | 11                   | 
 | 2 | 6 | 7 | 0110      | 0111      | 1     | 1     | 11                   | 
 | 3 | 0 | 1 | 0000      | 0001      | 0     | 0     | 00                   | 
 | 3 | 1 | 2 | 0001      | 0010      | 0     | 0     | 00                   | 
 | 3 | 2 | 3 | 0010      | 0011      | 0     | 0     | 00                   | 
 | 3 | 3 | 4 | 0011      | 0100      | 0     | 0     | 00                   | 
 | 3 | 4 | 5 | 0100      | 0101      | 0     | 0     | 00                   | 
 | 3 | 5 | 6 | 0101      | 0110      | 0     | 0     | 00                   | 
 | 3 | 6 | 7 | 0110      | 0111      | 0     | 0     | 00                   | 


The next stpe is groupping the table by `(i, seq)` columns and count the occurences.
`p` is the probability with wich a sequence at bit position `i` occurs.

| i | seq | count | p = count/7 | 
| - | -   | -     | -           | 
| 0 | 00  | 0     | 0.000       | 
| 0 | 01  | 4     | 0.571       | 
| 0 | 10  | 3     | 0.429       | 
| 0 | 11  | 0     | 0.000       | 
| 1 | 00  | 2     | 0.286       | 
| 1 | 01  | 2     | 0.286       | 
| 1 | 10  | 1     | 0.143       | 
| 1 | 11  | 2     | 0.286       | 
| 2 | 00  | 3     | 0.429       | 
| 2 | 01  | 1     | 0.143       | 
| 2 | 10  | 0     | 0.000       | 
| 2 | 11  | 3     | 0.429       | 
| 3 | 00  | 7     | 1.000       | 
| 3 | 01  | 0     | 0.000       | 
| 3 | 10  | 0     | 0.000       | 
| 3 | 11  | 0     | 0.000       | 

In the last step we compute the mutual information.
To do that we take the columns `i` and `p` from the table and reshape them so that we have the probabilities for each sequence, i.e., `p00`, `p01`, `p10`, `p11`, in separate columns. 
This allows us to continue our example as a spreadsheet.


  <!--- `p0x = p00+p01` is the probability that `seq` starts with a `0`.-->
  <!--- `px0 = p00+p10` is the probability that `seq` ends with a `0`.-->
  <!--- `p1x = p10+p11` is the probability that `seq` starts with a `1`.-->
  <!--- `px1 = p01+p11` is the probability that `seq` ends with a `1`.-->


<!--Formula for computing mutual informaiton.-->

<!--M = p00 * log(p00 / p0x / px0) +-->
<!--    p01 * log(p01 / p0x / px1) +-->
<!--    p10 * log(p10 / p1x / px0) +-->
<!--    p11 * log(p11 / p1x / px1)-->

<!--    M = M / log(2)-->

<!--| bit | p00   | p01   | p10   | p11   | p0x   | p1x   | px0   | px1   | M00   | M01    | M10   | M11   | M     | -->
<!--| -   | -     | -     | -     | -     | -     | -     | -     | -     | -     | -      | -     | -     | -     | -->
<!--| 0   | 0.000 | 0.571 | 0.429 | 0.000 | 0.571 | 0.429 | 0.429 | 0.571 | 0.000 | 0.210  | 0.000 | 0.000 | 0.699 | -->
<!--| 1   | 0.286 | 0.286 | 0.143 | 0.286 | 0.571 | 0.429 | 0.429 | 0.571 | 0.019 | 0.019  | 0.177 | 0.405 | 2.061 | -->
<!--| 2   | 0.429 | 0.143 | 0.000 | 0.429 | 0.571 | 0.429 | 0.429 | 0.571 | 0.104 | -0.033 | 0.000 | 0.000 | 0.235 | -->
<!--| 3   | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000  | 0.000 | 0.000 | 0.000 | -->

Formula below computes mutual information.
It says how much information a bit contains.

    M' = p00 * log(p00 / (p00 + p01) / (p00 + p10)) +
         p01 * log(p01 / (p00 + p01) / (p01 + p11)) +
         p10 * log(p10 / (p10 + p11) / (p00 + p10)) +
         p11 * log(p11 / (p10 + p11) / (p01 + p11))

    M = M' / log(2)

| i | p00   | p01   | p10   | p11   | M     | 
| - | -     | -     | -     | -     | -     | 
| 0 | 0.000 | 0.571 | 0.429 | 0.000 | 0.699 | 
| 1 | 0.286 | 0.286 | 0.143 | 0.286 | 2.061 | 
| 2 | 0.429 | 0.143 | 0.000 | 0.429 | 0.235 | 
| 3 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 
													


## Install

    python3 -m pip install bitinformation

## Usage

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
