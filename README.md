# llm.swift

A Swift port of Andrej Karpathy‘s [llm.c](https://github.com/karpathy/llm.c). 

## Quick start
- Clone [llm.c](https://github.com/karpathy/llm.c), checkout commit [2346cdac](https://github.com/karpathy/llm.c/tree/2346cdac931f544d63ce816f7e3f5479a917eef5) (`git checkout 2346cdac`) this port is based on, and follow instructions given there in section [quick start (CPU)](https://github.com/otabuzzman/llm.c/blob/2346cdac931f544d63ce816f7e3f5479a917eef5/README.md#quick-start-cpu). This will get you the dataset, the tokens, the small GPT-2 model (124M) released by OpenAI, and two executables for testing and training.

- Clone this repository, cd into it, build and run the executables for testing and training. 

  ```
  git clone https://github.com/otabuzzman/llm.swift.git
  cd llm.swift
  
  xcodebuild # will build for production
  # create sym-links for convenience
  ln -s build/Release/llm.swift test_gpt2
  ln -s build/Release/llm.swift train_gpt2
  
  # usage:
  # test_gpt2 [ <llm.c folder> ]
  # train_gpt2 [ <llm.c folder> ]
  
  ./test_gpt2 ../llm.c   # llm.c data in sibling folder
  ./train_gpt2 ../llm.c
  ```

### Performance notes
Swift porting done one-to-one with existing parallelization tools is slower than C and probably worth further investigation (preliminary conclusion).

The sequential (`NO_OMP=1`) code from the C compiler runs roughly 3 times faster than the code produced by the Swift compiler (20/ 60 seconds), measured in step duration of `test_gpt2` on a Macbook Air (2015) (Intel, 4 cores, 8 GB).

Parallelization with OpenMP (`NO_OMP=0`) in C and built-in _Structured Concurrency_ in Swift (using `withTaskGroup`) cuts down execution times to a half on both sides (10/ 30 seconds).

Regarding Swift, almost all of the performance gain came from parallelizing `matmul_forward`. The contribution of parallelization of the remaining functions that use OpenMP in C was marginal.

Parallelization of `matmul_backward` was strange as it lengthens execution time (40 seconds); thus left out `matmul_backword` parallelization for later investigation.

Using the _Grand Central Dispatch_ API `concurrentPerform` instead of `withTaskGroup` for any function that uses OpenMP on the C side yields an execution time on 25 seconds. 

Execution times of setups with notable impact:

|#|Setup|Time (seconds)|
|---:|:---|---:|
|1|C, `-march=native`, OpenMP|10|
|2|C, `-march=native`, no OpenMP|20|
|3|C, no `-march=native`, OpenMP|15|
|4|C, no `-march=native`, no OpenMP|25|
|5|Swift, `-march=native`, `matmul_backward`|50|
|6|Swift, `-march=native`, no `matmul_backward`|30|
|7|Swift, `-march=native`, GCD|25|

## Output samples

Output of Swift's `train_gpt2`:
```
[GPT-2]
max_seq_len: 1024
vocab_size: 50257
padded_vocab_size: 50304
num_layers: 12
num_heads: 12
channels: 768
num_parameters: 124475904
train dataset num_batches: 1192
val dataset num_batches: 128
num_activations: 73347840
val loss 5.3254156
step 0: train loss 5.356085 (took 60110.31210422516 ms)
step 1: train loss 4.3006387 (took 58332.464933395386 ms)
step 2: train loss 4.623086 (took 58333.27293395996 ms)
step 3: train loss 4.5993624 (took 58257.12597370148 ms)
step 4: train loss 4.6166654 (took 58329.46801185608 ms)
step 5: train loss 4.2314286 (took 58158.995032310486 ms)
step 6: train loss 3.7531614 (took 59324.42593574524 ms)
step 7: train loss 3.650456 (took 58542.4120426178 ms)
step 8: train loss 4.182243 (took 58558.4659576416 ms)
step 9: train loss 4.1995816 (took 58372.3349571228 ms)
val loss 4.3237658
step 10: train loss 4.288661 (took 58260.937094688416 ms)
step 11: train loss 3.5606422 (took 59435.32598018646 ms)
step 12: train loss 3.7314336 (took 62170.435070991516 ms)
step 13: train loss 4.1585116 (took 58287.61005401611 ms)
step 14: train loss 3.8856335 (took 58255.18488883972 ms)
step 15: train loss 3.766486 (took 58274.41608905792 ms)
step 16: train loss 4.1440086 (took 58361.828088760376 ms)
step 17: train loss 3.9611669 (took 58290.65299034119 ms)
step 18: train loss 3.796045 (took 60170.04597187042 ms)
step 19: train loss 3.3710434 (took 60961.08305454254 ms)
val loss 4.187855
generating:
---
I was so frightened with your face: to come and though they would not do it any more than as
Let us; but who ever can turn
Against a world so full,
That there'll have been none of our fightmen but
Weaver-bats and tearing men, and stir them utterly;
---
step 20: train loss 3.8827896 (took 63009.78696346283 ms)
step 21: train loss 4.199985 (took 59306.17904663086 ms)
step 22: train loss 4.4284277 (took 61485.97192764282 ms)
step 23: train loss 3.6859236 (took 58441.08498096466 ms)
step 24: train loss 3.6432934 (took 58446.670055389404 ms)
step 25: train loss 3.729694 (took 58503.68797779083 ms)
step 26: train loss 3.5506494 (took 63852.79297828674 ms)
step 27: train loss 3.3386307 (took 63432.64198303223 ms)
step 28: train loss 4.3420167 (took 63263.363003730774 ms)
step 29: train loss 3.8147268 (took 59481.106996536255 ms)
val loss 4.0227003
step 30: train loss 4.032416 (took 58526.270031929016 ms)
step 31: train loss 4.1180716 (took 60656.75401687622 ms)
step 32: train loss 3.5770087 (took 59657.63700008392 ms)
step 33: train loss 4.369796 (took 59965.45505523682 ms)
step 34: train loss 4.524115 (took 65765.72799682617 ms)
step 35: train loss 4.4388156 (took 64983.98792743683 ms)
step 36: train loss 4.1010995 (took 62695.37305831909 ms)
step 37: train loss 3.740977 (took 63762.57002353668 ms)
step 38: train loss 4.618742 (took 65055.13501167297 ms)
step 39: train loss 3.9722583 (took 61024.436950683594 ms)
val loss 4.0173483
generating:
---
CLAUSE:
I cannot, sir, can; as I would
do @scended
da drawn breath
to love
Ferrante, the fourth Receiver: the king must leave this
matter for our own use,
who will
roll the first wine-tureen and
press the
---
step 40: train loss 4.378439 (took 60820.064067840576 ms)
```

Output of Swift's `test_gpt2`:
```
[GPT-2]
max_seq_len: 1024
vocab_size: 50257
padded_vocab_size: 50304
num_layers: 12
num_heads: 12
channels: 768
num_parameters: 124475904
[State]
batch_size: 4
seq_len: 64
num_activations: 73347840
-43.43169 -43.431736
-39.836414 -39.83645
-43.06595 -43.066032
-42.828117 -42.828136
-43.52961 -43.52965
-44.318455 -44.318516
-41.227463 -41.22754
-41.270805 -41.27087
-42.54143 -42.54153
-42.39506 -42.395123
OK (LOGITS), max_diff = 0.0010375977
LOSS OK: 5.269893 5.269998
dwte
OK -0.0023204603 -0.0023199492
OK 0.0020716386 0.002071693
OK 0.0037163584 0.0037171948
OK 0.0013074379 0.0013068196
OK 0.0006306543 0.000631563
TENSOR OK, maxdiff = 0.0011749268
dwpe
OK -0.005118038 -0.005111558
OK -1.0658841e-06 -9.327196e-06
OK -0.0032674088 -0.0032632265
OK 0.009909394 0.00991323
OK 0.0021548946 0.002145649
TENSOR OK, maxdiff = 3.6744867e-05
dln1w
OK -0.0075203087 -0.0075246324
OK 0.008623586 0.008638417
OK 0.005003491 0.0050241766
OK -0.011098319 -0.011095028
OK -0.0016665062 -0.0016648447
TENSOR OK, maxdiff = 0.0030422118
dln1b
OK -0.03849395 -0.03847474
OK -0.030547278 -0.030597843
OK 0.010188758 0.010218
OK 0.08013435 0.08018854
OK -0.06099027 -0.060926706
TENSOR OK, maxdiff = 0.0012842207
dqkvw
OK -3.1021205e-05 -3.0990035e-05
OK -2.5524003e-05 -2.5398767e-05
OK -6.436045e-05 -6.433582e-05
OK 7.427888e-05 7.4414296e-05
OK 1.9829093e-05 2.0034813e-05
TENSOR OK, maxdiff = 0.00047449023
dqkvb
OK -0.00041382186 -0.00041190616
OK -0.00040984616 -0.00041113945
OK 0.00011316869 0.00011331367
OK -0.0005638881 -0.00056477886
OK 0.0005741234 0.00057149713
TENSOR OK, maxdiff = 0.000256842
dattprojw
OK 8.057074e-05 8.0502905e-05
OK -5.3344797e-06 -4.8910492e-06
OK -1.8935682e-05 -1.9039395e-05
OK 4.6994746e-06 4.3724604e-06
OK 3.1259875e-05 3.14401e-05
TENSOR OK, maxdiff = 0.00020065159
dattprojb
OK 0.00045628706 0.0004668263
OK -0.009968744 -0.009975138
OK -0.0017937027 -0.0017999576
OK 0.037638217 0.03760945
OK -0.03128777 -0.03125162
TENSOR OK, maxdiff = 0.00017983094
dln2w
OK -0.0183719 -0.01831808
OK 0.0048115053 0.0048143384
OK 0.008084181 0.008092893
OK -0.0014647923 -0.0014690551
OK -0.0027395312 -0.0027370392
TENSOR OK, maxdiff = 0.009723067
dln2b
OK -0.026405131 -0.026363645
OK -0.016711498 -0.016694352
OK 0.0010668249 0.0010847792
OK 0.034754228 0.03473249
OK -0.02863019 -0.028592188
TENSOR OK, maxdiff = 0.00081983954
dfcw
OK 0.00043798445 0.00043941368
OK -1.5009559e-07 -5.0043816e-08
OK -0.00015316524 -0.00015360993
OK -0.00016470066 -0.00016475243
OK 0.00040368875 0.00040468978
TENSOR OK, maxdiff = 0.0007956773
dfcb
OK 0.0032822792 0.003288376
OK 0.0020379843 0.0020419345
OK -0.001385784 -0.0013858759
OK 0.00038110014 0.00038582497
OK 0.0016024122 0.0016040489
TENSOR OK, maxdiff = 0.00019360008
dfcprojw
OK 0.0006779426 0.00067962374
OK 7.346058e-05 7.343709e-05
OK -0.0004150932 -0.0004159588
OK -5.8958e-05 -6.031629e-05
OK -0.0006025975 -0.00060316984
TENSOR OK, maxdiff = 0.00038507365
dfcprojb
OK 0.0035725276 0.0035794298
OK -0.007147566 -0.007154937
OK -0.0019545457 -0.0019616296
OK 0.0014659498 0.0014630328
OK 0.0012187347 0.0012136659
TENSOR OK, maxdiff = 0.000117892516
dlnfw
OK -2.2002176e-05 -2.2250955e-05
OK 0.0008107438 0.0008106372
OK 0.0011611169 0.0011611973
OK -0.0029564737 -0.0029568998
OK 0.001146391 0.0011453481
TENSOR OK, maxdiff = 0.00036417693
dlnfb
OK -0.0111008845 -0.0111006675
OK 0.008006945 0.008006417
OK -0.004763235 -0.0047669318
OK -0.0021103222 -0.0021124498
OK -0.0059031383 -0.0059049977
TENSOR OK, maxdiff = 6.598537e-05
step 0: loss 5.269893 (took 54210.73400974274 ms) OK = true
step 1: loss 4.0593877 (took 55287.113070487976 ms) OK = true
step 2: loss 3.3742127 (took 59258.83495807648 ms) OK = true
step 3: loss 2.8001285 (took 58529.488921165466 ms) OK = true
step 4: loss 2.3153136 (took 54386.79397106171 ms) OK = true
step 5: loss 1.849349 (took 54115.27609825134 ms) OK = true
step 6: loss 1.3952194 (took 60949.86701011658 ms) OK = true
step 7: loss 0.9986158 (took 55763.24498653412 ms) OK = true
step 8: loss 0.6255399 (took 55778.178095817566 ms) OK = true
step 9: loss 0.37801343 (took 53648.396015167236 ms) OK = true
overall okay: true
```

Output of C's `train_gpt2`:
```
[GPT-2]
max_seq_len: 1024
vocab_size: 50257
padded_vocab_size: 50304
num_layers: 12
num_heads: 12
channels: 768
num_parameters: 124475904
train dataset num_batches: 1192
val dataset num_batches: 128
num_activations: 73347840
val loss 5.325531
step 0: train loss 5.356193 (took 17168.833000 ms)
step 1: train loss 4.301083 (took 14565.950000 ms)
step 2: train loss 4.623323 (took 13462.665000 ms)
step 3: train loss 4.600489 (took 17875.864000 ms)
step 4: train loss 4.616791 (took 23968.945000 ms)
step 5: train loss 4.231491 (took 19155.457000 ms)
step 6: train loss 3.754269 (took 21874.913000 ms)
step 7: train loss 3.652406 (took 20047.310000 ms)
step 8: train loss 4.183630 (took 15435.924000 ms)
step 9: train loss 4.199314 (took 14743.075000 ms)
val loss 4.323425
step 10: train loss 4.288379 (took 13310.036000 ms)
step 11: train loss 3.558854 (took 13967.735000 ms)
step 12: train loss 3.730748 (took 15885.448000 ms)
step 13: train loss 4.159210 (took 16823.004000 ms)
step 14: train loss 3.886536 (took 18328.178000 ms)
step 15: train loss 3.764807 (took 12157.076000 ms)
step 16: train loss 4.142972 (took 11270.109000 ms)
step 17: train loss 3.962871 (took 11328.664000 ms)
step 18: train loss 3.796138 (took 11029.651000 ms)
step 19: train loss 3.371690 (took 17496.030000 ms)
val loss 4.186526
generating:
---
I was so upright that I would have never heard you had any talk. I have heard you talking the Crows cawing as upon a mountain, Fist,
Of will'er than these laughs we follow'd.

<|endoftext|>Second Servingman:
I salute him one man, and respond'd thee with
---
step 20: train loss 3.880784 (took 11758.028000 ms)
step 21: train loss 4.198482 (took 15723.884000 ms)
step 22: train loss 4.425916 (took 11974.415000 ms)
step 23: train loss 3.685766 (took 16036.071000 ms)
step 24: train loss 3.642242 (took 14144.053000 ms)
step 25: train loss 3.729666 (took 11957.165000 ms)
step 26: train loss 3.549570 (took 13603.906000 ms)
step 27: train loss 3.339429 (took 14257.500000 ms)
step 28: train loss 4.338738 (took 14916.327000 ms)
step 29: train loss 3.812686 (took 14934.599000 ms)
val loss 4.020240
step 30: train loss 4.027640 (took 11407.487000 ms)
step 31: train loss 4.114108 (took 11374.677000 ms)
step 32: train loss 3.574935 (took 11606.922000 ms)
step 33: train loss 4.365807 (took 10847.886000 ms)
step 34: train loss 4.515867 (took 10943.073000 ms)
step 35: train loss 4.433772 (took 11734.479000 ms)
step 36: train loss 4.097108 (took 12839.231000 ms)
step 37: train loss 3.739621 (took 12013.618000 ms)
step 38: train loss 4.611548 (took 16237.573000 ms)
step 39: train loss 3.970719 (took 11972.036000 ms)
val loss 4.016654
generating:
---
Come happily forth,
Where a lordsen cludges you;
That impegraces and immunities race for your fellow-citizens,
The men of your kingdoms, or those castrest for your substitutes; and
As a friend you may
Be the most frank, the virtuous;
That are
---
step 40: train loss 4.377730 (took 11135.249000 ms)
```

Output of C's `test_gpt2`:
```
[GPT-2]
max_seq_len: 1024
vocab_size: 50257
padded_vocab_size: 50304
num_layers: 12
num_heads: 12
channels: 768
num_parameters: 124475904
[State]
batch_size: 4
seq_len: 64
num_activations: 73347840
-43.431690, -43.431690
-39.836414, -39.836407
-43.065948, -43.065945
-42.828117, -42.828121
-43.529610, -43.529606
-44.318455, -44.318451
-41.227463, -41.227459
-41.270805, -41.270809
-42.541431, -42.541435
-42.395061, -42.395065
OK (LOGITS), max_diff = 8.544922e-04
LOSS OK: 5.270007 5.269998
dwte
OK -0.002320 -0.002320
OK 0.002072 0.002072
OK 0.003717 0.003717
OK 0.001307 0.001307
OK 0.000632 0.000632
TENSOR OK, maxdiff = 1.754761e-04
dwpe
OK -0.005110 -0.005112
OK -0.000012 -0.000009
OK -0.003261 -0.003263
OK 0.009908 0.009913
OK 0.002145 0.002146
TENSOR OK, maxdiff = 3.322959e-05
dln1w
OK -0.007523 -0.007525
OK 0.008643 0.008638
OK 0.005028 0.005024
OK -0.011094 -0.011095
OK -0.001663 -0.001665
TENSOR OK, maxdiff = 1.166821e-03
dln1b
OK -0.038457 -0.038475
OK -0.030594 -0.030598
OK 0.010218 0.010218
OK 0.080176 0.080189
OK -0.060899 -0.060927
TENSOR OK, maxdiff = 2.463497e-04
dqkvw
OK -0.000031 -0.000031
OK -0.000025 -0.000025
OK -0.000064 -0.000064
OK 0.000074 0.000074
OK 0.000020 0.000020
TENSOR OK, maxdiff = 1.879632e-04
dqkvb
OK -0.000411 -0.000412
OK -0.000412 -0.000411
OK 0.000114 0.000113
OK -0.000565 -0.000565
OK 0.000570 0.000571
TENSOR OK, maxdiff = 1.053312e-04
dattprojw
OK 0.000080 0.000081
OK -0.000005 -0.000005
OK -0.000019 -0.000019
OK 0.000004 0.000004
OK 0.000032 0.000031
TENSOR OK, maxdiff = 5.920976e-05
dattprojb
OK 0.000471 0.000467
OK -0.009981 -0.009975
OK -0.001804 -0.001800
OK 0.037575 0.037609
OK -0.031233 -0.031252
TENSOR OK, maxdiff = 4.418450e-05
dln2w
OK -0.018314 -0.018318
OK 0.004812 0.004814
OK 0.008089 0.008093
OK -0.001470 -0.001469
OK -0.002737 -0.002737
TENSOR OK, maxdiff = 1.741350e-03
dln2b
OK -0.026374 -0.026364
OK -0.016702 -0.016694
OK 0.001070 0.001085
OK 0.034703 0.034732
OK -0.028579 -0.028592
TENSOR OK, maxdiff = 1.475960e-04
dfcw
OK 0.000440 0.000439
OK -0.000000 -0.000000
OK -0.000154 -0.000154
OK -0.000165 -0.000165
OK 0.000405 0.000405
TENSOR OK, maxdiff = 1.643896e-04
dfcb
OK 0.003293 0.003288
OK 0.002043 0.002042
OK -0.001386 -0.001386
OK 0.000386 0.000386
OK 0.001603 0.001604
TENSOR OK, maxdiff = 5.320832e-05
dfcprojw
OK 0.000681 0.000680
OK 0.000073 0.000073
OK -0.000416 -0.000416
OK -0.000061 -0.000060
OK -0.000604 -0.000603
TENSOR OK, maxdiff = 7.372192e-05
dfcprojb
OK 0.003585 0.003579
OK -0.007159 -0.007155
OK -0.001963 -0.001962
OK 0.001462 0.001463
OK 0.001218 0.001214
TENSOR OK, maxdiff = 2.294645e-05
dlnfw
OK -0.000022 -0.000022
OK 0.000810 0.000811
OK 0.001161 0.001161
OK -0.002957 -0.002957
OK 0.001145 0.001145
TENSOR OK, maxdiff = 1.777709e-04
dlnfb
OK -0.011100 -0.011101
OK 0.008009 0.008006
OK -0.004771 -0.004767
OK -0.002112 -0.002112
OK -0.005905 -0.005905
TENSOR OK, maxdiff = 4.637241e-05
step 0: loss 5.270007 (took 12206.436000 ms) OK = 1
step 1: loss 4.059719 (took 11219.729000 ms) OK = 1
step 2: loss 3.375097 (took 10799.359000 ms) OK = 1
step 3: loss 2.800843 (took 10800.955000 ms) OK = 1
step 4: loss 2.315461 (took 12180.481000 ms) OK = 1
step 5: loss 1.849123 (took 11101.643000 ms) OK = 1
step 6: loss 1.394796 (took 11346.653000 ms) OK = 1
step 7: loss 0.999234 (took 11128.530000 ms) OK = 1
step 8: loss 0.624181 (took 10970.779000 ms) OK = 1
step 9: loss 0.376572 (took 10836.680000 ms) OK = 1
overall okay: 1
```
