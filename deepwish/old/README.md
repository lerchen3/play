tokens/: an attempt to run distributed bpe for no reason at all. fails miserably because gh codespace ssh command has significant startup overhead (sending 114 bytes via ssh takes 5.77s) and gh codespace cp is wonky and doesn't seem to work. given that each chunk of work takes ~1s there's no way for this to be hidden :')
keeping in case i want to run distributed things later where the 5s latency is acceptable
(full bpe ended up running in 4 minutes with 48-threaded cpp code on the 3gb dataset. lmfao? oops)
triton/: some toy triton kernels; learning the language.