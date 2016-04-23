SUBDIRS := $(shell ls -F */Makefile)
MODULES := src #$(patsubst %/Makefile, %, $(SUBDIRS))
CLEAN := $(patsubst %, %-clean, $(MODULES))
depth := [$(MAKELEVEL)]

# Make modules
all: $(MODULES) test
	
.PHONY: $(MODULES)
$(MODULES):
	@echo "$(depth) \033[32;1mMAKE\033[0m $@/ all"
	@${MAKE} --directory=$@ all
	
clean:
	@for m in $(MODULES); do \
		echo "$(depth) \033[32;1mMAKE\033[0m $$m/ $@"; \
		${MAKE} --directory=$$m $@; \
	done
	@echo "$(depth) \033[32;1mMAKE\033[0m test/ $@";
	@${MAKE} --directory=test $@; \

.DEFAULT:
	@for m in $(MODULES); do \
		echo "$(depth) \033[32;1mMAKE\033[0m $$m/ $@"; \
		${MAKE} --directory=$$m $@; \
	done
	
test: $(MODULES)
	@echo "$(depth) \033[32;1mMAKE\033[0m $@/ all"
	@${MAKE} --directory=test all
	
