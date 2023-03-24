## a quick make file for test

BASE = $(shell /bin/pwd)
SRCD = $(BASE)/src
OBJD = $(BASE)/obj

SRC += $(SRCD)/Chebyshev.cc
#SRC += $(SRCD)/Legendre.cc
#SRC += $(SRCD)/sYlm.cc
#SRC += $(SRCD)/matrix.cc
#SRC += $(SRCD)/h5read.cc
#SRC += $(SRCD)/decomp.cc
#SRC += $(SRCD)/alias.cc
#SRC += $(SRCD)/ExtractMetric.cc

OBJ = $(patsubst $(SRCD)/%.cc,$(OBJD)/%.o,$(SRC))

INC_PARAMS=-I$(SRCD)

CXX ?= g++
CXXFLAGS ?= 


.PHONY: all clean

all: $(OBJ)
	@echo 'compiling done!'

$(OBJD)/%.o: $(SRCD)/%.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INC_PARAMS) -c $< -o $@

clean:
	@rm -rvf $(OBJD)

