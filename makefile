SUBDIRS := $(shell ls -F */Makefile)
MODULES := src #$(patsubst %/Makefile, %, $(SUBDIRS))
CLEAN := $(patsubst %, %-clean, $(MODULES))
depth := [$(MAKELEVEL)]

# Make modules
all: $(MODULES)
	
.PHONY: $(MODULES)
$(MODULES):
	@echo "$(depth) \033[32;1mMAKE\033[0m $@/ all"
	@${MAKE} --quiet --directory=$@ all
	
.DEFAULT:
	@for m in $(MODULES); do \
		echo "$(depth) \033[32;1mMAKE\033[0m $$m/ $@"; \
		${MAKE} --quiet --directory=$$m $@; \
	done
	
test: $(MODULES)
	@echo "$(depth) \033[32;1mMAKE\033[0m $@/ all"
	@${MAKE} --quiet --directory=test all
	
