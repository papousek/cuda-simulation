SRCDIR=c/src
SRCFILE=num_sim_kernel.cu
BUILDDIR=c/build
OUTFILE=num_sim_kernel.cubin
DEBUG_FLAGS=-G -g
FLAGS=-O2 -gencode=arch=compute_20,code=\"sm_20\" -gencode=arch=compute_20,code=\"sm_20\"
CUBIN_FLAGS=-cubin
CC=nvcc

.PHONY:
compile:
	$(CC) $(FLAGS) -o $(BUILDDIR)/$(OUTFILE) $(SRCDIR)/$(SRCFILE);

.PHONY: cubin
cubin:
	$(CC) $(CUBIN_FLAGS) $(FLAGS) -o $(BUILDDIR)/$(OUTFILE) $(SRCDIR)/$(SRCFILE);

.PHONY: debug
debug:
	$(CC) $(DEBUG_FLAGS) $(FLAGS) -o $(BUILDDIR)/$(OUTFILE) $(SRCDIR)/$(SRCFILE);

