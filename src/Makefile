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
OBJDIRS := $(patsubst %, $(ROOTDIR)obj/%, $(SUBDIRS))
SOURCE_FILES := $(foreach d, $(DIRS), $(wildcard $(d)*.cu) )
NEED_MODULES := 
COMPILE_OPTIONS := -arch=compute_30 -code=sm_30

LIBS = $(NEED_MODULES:%=-l%)
ARCHIVES = $(foreach lib,$(NEED_MODULES),$(ROOTDIR)lib/lib$(lib).a)
#Include
INCLUDES = -I$(ROOTDIR)include -I/usr/local/cuda/include $(NEED_MODULES:%=-I../%/)

# Create an object file of every cu file
OBJECTS = $(patsubst ./%.cu, $(ROOTDIR)obj/%.o, $(SOURCE_FILES))

# Dependencies
DEPENDENCIES = $(patsubst ./%.cu, $(ROOTDIR)deps/%.d, $(SOURCE_FILES))

# Target module library
TARGET := $(ROOTDIR)lib/lib$(PROJECT).a

# Create directories
%/:
	@echo "$(depth) \033[32;1mMKDIR\033[0m $(patsubst $(ROOTDIR)%,%,$@)"
	@mkdir -p $@

# Create .d files
$(ROOTDIR)deps/%.d: %.cu $(ROOTDIR)deps/
	@echo "$(depth) \033[32;1mCC -MM\033[0m $(lastword $(subst /, $(space), $@))"
	@$(NVCC) -E -ccbin=$(CC) --compiler-options=-MM,-MT,$(ROOTDIR)obj/$*.o,-MF,$@ -I./ $(INCLUDES) -L$(ROOTDIR)lib/ $(LIBS) $<

# Make $(PROJECT) the default target
all: $(DEPENDENCIES) $(ARCHIVES) $(TARGET)

$(ROOTDIR)lib/lib%.a: $(ROOTDIR)lib/
	$(eval lib := $(patsubst $(ROOTDIR)lib/lib%.a,%,$@))
	@echo "$(depth) \033[32;1mMAKE\033[0m $(ROOTDIR)$(lib) all"
	@$(MAKE) --quiet --directory=$(ROOTDIR)$(lib) all

# Include dependencies (if there are any)
ifneq "$(strip $(DEPENDENCIES))" ""
    -include $(DEPENDENCIES)
endif

# Compile every cu file to an object
$(ROOTDIR)obj/%.o: %.cu
	@echo "$(depth) \033[32;1mCC\033[0m $(lastword $(subst /, $(space), $@))"
	@$(NVCC) -ccbin=$(CC) -dc $(COMPILE_OPTIONS) -o $@ $< $(HEADERS) -I./ $(INCLUDES) -L$(ROOTDIR)lib/ $(LIBS)

$(TARGET): $(OBJECTS)
	@echo "$(depth) \033[32;1mAR\033[0m $(lastword $(subst /, $(space), $(TARGET)))"
	@ar cr $(TARGET) $(OBJECTS)

# Clean & Debug
.PHONY: makefile-debug
makefile-debug:

.PHONY: clean
clean:
	@echo "$(depth) \033[32;1mRM\033[0m $(lastword $(subst /, $(space), $(TARGET)))"
	@echo "$(depth) \033[32;1mRM\033[0m $(patsubst $(ROOTDIR)obj/%, %, $(OBJECTS))"
	@rm -f $(TARGET) $(OBJECTS) *.plist

.PHONY: depclean
depclean:
	@echo "$(depth) \033[32;1mRM\033[0m $(DEPENDENCIES)"
	@rm -f $(DEPENDENCIES)

clean-all: clean depclean
