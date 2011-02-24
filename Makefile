SRCDIR=c/src
SRCFILE=ode_solver.cu
BUILDDIR=c/build
OUTFILE=ode_solver
DEBUG_FLAGS=-G -g
FLAGS=-O2 -gencode=arch=compute_20,code=\"sm_20\" -gencode=arch=compute_20,code=\"sm_20\"
CUBIN_FLAGS=-cubin
CC=nvcc

.PHONY:
compile:
	$(CC) $(FLAGS) -o $(BUILDDIR)/$(OUTFILE) $(SRCDIR)/$(SRCFILE)

.PHONY: cubin
cubin:
	$(CC) $(CUBIN_FLAGS) $(FLAGS) -o kernel_runge_kutta kernel_runge_kutta.cu;

.PHONY: debug
debug:
	$(CC) $(DEBUG_FLAGS) $(FLAGS) -o ode_solver ode_solver.cu;

