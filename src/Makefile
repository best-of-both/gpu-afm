# Subdirs to search for additional source files
space :=
space +=
depth := [$(MAKELEVEL)]

NVCC := /usr/local/cuda/bin/nvcc
CC := cc

ROOTDIR := $(shell git rev-parse --show-cdup)
PROJECT := afm
SUBDIRS := $(shell ls -F | grep "\/" | grep -v '^test\/$$' )
MODULES := #$(patsubst %/Makefile,%,$(shell ls */Makefile))
DIRS := ./ $(patsubst %, ./%, $(SUBDIRS))
OBJDIRS := $(patsubst ./%, $(ROOTDIR)obj/%,$(DIRS) )
SOURCE_FILES := $(foreach d, $(DIRS), $(wildcard $(d)*.cu) )
NEED_MODULES := 

LIBS = $(NEED_MODULES:%=-l%)
ARCHIVES = $(foreach lib,$(NEED_MODULES),$(ROOTDIR)lib/lib$(lib).a)
#Include
INCLUDES = -I$(ROOTDIR)include -I/usr/local/cuda/include $(NEED_MODULES:%=-I../%/)

# Create an object file of every cu file
OBJECTS = $(patsubst ./%, $(ROOTDIR)obj/%, $(patsubst %.cu, %.o, $(SOURCE_FILES)))

# Dependencies
DEPENDENCIES = $(patsubst ./%.cu, $(ROOTDIR)deps/%.d, $(SOURCE_FILES))

# Target module library
TARGET := $(ROOTDIR)lib/lib$(PROJECT).a

# Create directories
%/:
	@echo "$(depth) \033[32;1mMKDIR\033[0m $(patsubst $(ROOTDIR)%,%,$@)"
	@mkdir -p $@

# Create .d files
$(ROOTDIR)deps/%.d: %.cu
	@echo "$(depth) \033[32;1mCC -MM\033[0m $(lastword $(subst /, $(space), $@))"
	$(NVCC) -E -ccbin=$(CC) --compiler-options=-MM,-MT,$(ROOTDIR)obj/$*.o,-MF,$@ -I./ $(INCLUDES) -L$(ROOTDIR)lib/ $(LIBS) $<

# Make $(PROJECT) the default target
all: $(OBJDIRS) $(DEPENDENCIES) $(ARCHIVES) $(TARGET)

$(ARCHIVES):
	$(eval lib := $(patsubst $(ROOTDIR)lib/lib%.a,%,$@))
	@echo "$(depth) \033[32;1mMAKE\033[0m ../$(lib) all"
	@$(MAKE) --quiet --directory=../$(lib) all

$(TARGET): $(OBJECTS)
	@echo "$(depth) \033[32;1mAR\033[0m $(lastword $(subst /, $(space), $(TARGET)))"
	@ar cr $(TARGET) $(OBJECTS)

%.exe: %.cu $(TARGET)
	@echo "$(depth) \033[32;1mCC TEST\033[0m $(lastword $(subst /, $(space), $@))"
	@$(NVCC) -ccbin=$(CC) $(COMPILE_OPTIONS) -o $@ $< -L$(ROOTDIR)lib/ -l$(PROJECT) -I./ $(INCLUDES) $(LIBS)

# Include dependencies (if there are any)
ifneq "$(strip $(DEPENDENCIES))" ""
    -include $(DEPENDENCIES)
endif

# Compile every cu file to an object
$(ROOTDIR)obj/%.o: %.cu
	@echo "$(depth) \033[32;1mCC\033[0m $(lastword $(subst /, $(space), $@))"
	@$(NVCC) -ccbin=$(CC) -c $(COMPILE_OPTIONS) -o $@ $< $(HEADERS) -I./ $(INCLUDES) -L$(ROOTDIR)lib/ $(LIBS)

# Clean & Debug
.PHONY: makefile-debug
makefile-debug:

.PHONY: clean
clean:
	@echo "$(depth) \033[32;1mRM\033[0m $(lastword $(subst /, $(space), $(TARGET)))"
	@echo "$(depth) \033[32;1mRM\033[0m $(patsubst $(ROOTDIR)obj/%,%,$(OBJECTS))"
	@rm -f $(TARGET) $(OBJECTS) *.plist

.PHONY: depclean
depclean:
	@echo "$(depth) \033[32;1mRM\033[0m $(DEPENDENCIES)"
	@rm -f $(DEPENDENCIES)

clean-all: clean depclean
