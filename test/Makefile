# Subdirs to search for additional source files
space :=
space +=
depth := [$(MAKELEVEL)]

NVCC := /usr/local/cuda/bin/nvcc
CC := cc

ROOTDIR := $(shell git rev-parse --show-cdup)
DIRS := ./ $(patsubst %, ./%, $(SUBDIRS))
SOURCE_FILES := $(foreach d, $(DIRS), $(wildcard $(d)*.cu) )

#Include
INCLUDES = -I$(ROOTDIR)include -I/usr/local/cuda/include

# Create an object file of every cu file
OBJECTS = $(patsubst ./%, $(ROOTDIR)obj/%, $(patsubst %.cu, %.o, $(SOURCE_FILES)))

# Dependencies
DEPENDENCIES = $(patsubst ./%, $(ROOTDIR)obj/%, $(patsubst %.cu, %.d, $(SOURCE_FILES)))

# Test units
TEST_SRCS := $(shell ls *.cu)
TEST_EXES := $(patsubst %.cu, %.exe, $(TEST_SRCS))

# Make $(PROJECT) the default target
all: $(TEST_EXES)

%.exe: %.cu
	@echo "$(depth) \033[32;1mCC TEST\033[0m $(lastword $(subst /, $(space), $@))"
	@$(NVCC) -ccbin=$(CC) $(COMPILE_OPTIONS) -o $@ $< -L$(ROOTDIR)lib/ -lafm -I./ $(INCLUDES) $(LIBS)

# Clean & Debug
.PHONY: makefile-debug
makefile-debug:

.PHONY: clean
clean:
	@rm -f $(TEST_EXES) *.plist

clean-all: clean
