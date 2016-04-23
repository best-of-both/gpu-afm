# Subdirs to search for additional source files
space :=
space +=
depth := [$(MAKELEVEL)]

NVCC := /usr/local/cuda/bin/nvcc
CC := cc

ROOTDIR := $(shell git rev-parse --show-cdup)
DIRS := ./ $(patsubst %, ./%, $(SUBDIRS))
SOURCE_FILES := $(foreach d, $(DIRS), $(wildcard $(d)*.cu) )
LIBS := -lcublas -lafm
COMPILE_OPTIONS := -arch=compute_30 -code=sm_30

#Include
INCLUDES = -I$(ROOTDIR)include -I/usr/local/cuda/include

# Test units
TEST_SRCS := $(shell ls *.cu)
TEST_EXES := $(patsubst %.cu, %.exe, $(TEST_SRCS))

# Make $(PROJECT) the default target
all: $(TEST_EXES)
	
$(ROOTDIR)obj/%.o: %.cu
	@echo "$(depth) \033[32;1mCC \033[0m $(lastword $(subst /, $(space), $@))"
	@$(NVCC) -ccbin=$(CC) $(COMPILE_OPTIONS) -I./ $(INCLUDES) -dc $< -o $@

%.exe: $(ROOTDIR)obj/%.o
	@echo "$(depth) \033[32;1mCC TEST\033[0m $(lastword $(subst /, $(space), $@))"
	@$(NVCC) -ccbin=$(CC) $(COMPILE_OPTIONS) -L$(ROOTDIR)lib/ -I./ $(INCLUDES) $(LIBS) $< -o $@


# Clean & Debug
.PHONY: makefile-debug
makefile-debug:

.PHONY: clean
clean:
	@echo "$(depth) \033[32;1mRM\033[0m *.o *.exe"
	@rm -f $(TEST_EXES) *.o *.plist

clean-all: clean
