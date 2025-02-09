checkpoint: gpt2_124M.bin gpt2_124M_debug_state.bin gpt2_tokenizer.bin

gpt2_124M.bin:
	wget -q -O $@ https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main/gpt2_124M.bin
gpt2_124M_debug_state.bin:
	wget -q -O $@ https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main/gpt2_124M_debug_state.bin
gpt2_tokenizer.bin:
	wget -q -O $@ https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main/gpt2_tokenizer.bin
