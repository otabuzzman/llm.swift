# llm.swift

A Swift port of Andrej Karpathy‘s [llm.c](https://github.com/karpathy/llm.c). The C version was ported with the necessary changes due to differences in the ecosystems, mainly file I/O and parallelization, with the latter using Grand Central Dispatch instead of OpenMP.

## Quick start

- Clone [llm.c](https://github.com/karpathy/llm.c) and follow instructions given there in README, section [quick start (CPU)](https://github.com/otabuzzman/llm.c/blob/2346cdac931f544d63ce816f7e3f5479a917eef5/README.md#quick-start-cpu). This will get you the dataset, the tokens, the small GPT-2 model (124M) released by OpenAI, and two executables for testing and training.

- Clone this repository, cd into it, build and run the executables for testing and training. 

  ```
  git clone https://github.com/otabuzzman/llm.swift.git
  cd llm.swift
  
  # build for production
  xcodebuild -scheme llm.swift -configuration Release \
    SWIFT_ACTIVE_COMPILATION_CONDITIONS="$SWIFT_ACTIVE_COMPILATION_CONDITIONS LLMSWIFT_STANDALONE"
  
  # usage:
  #   test_gpt2 [ <llm.c folder> ]
  #   train_gpt2 [ <llm.c folder> ]
  
  ./test_gpt2 ../llm.c # assuming llm.c in sibling folder
  ./train_gpt2 ../llm.c
  ```

The [samples.md](samples.md) file provides the output of llm.swift captured from the first working version (without Metal) on a MacBook Air 2015. The output of llm.c (with OpenMP) is also provided for comparison.

## Acknowledgements

**Metal implementation**

James Thompson - (Metal implementation)
<br>Copyright (c) 2024 James Thomson - MIT License

Adopted a concept shared by [@regrettable-username](https://github.com/regrettable-username) in his Metal port [llm.metal](https://github.com/regrettable-username/llm.metal).
