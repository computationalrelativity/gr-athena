## a quick make file for test

## conventions
LIBNAME = CCE
EXE = CCE

CXX ?= g++
CXXFLAGS ?= -Wfatal-errors
DFLAGS += -DTEST_DECOMP
#DFLAGS += -DUSE_LEGENDRE 

## external libs and includes
EX_LIBS += -lgsl 
EX_LIBS += -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5
EX_LIBS += -lm
EX_INCD = /usr/lib/x86_64-linux-gnu/hdf5/serial/include

BASE = $(shell /bin/pwd)
SRCD = $(BASE)/src
OBJD = $(BASE)/obj
LIBD = $(BASE)/lib

## compile only these files
SRC += $(SRCD)/Chebyshev.cc
SRC += $(SRCD)/Legendre.cc
SRC += $(SRCD)/sYlm.cc
SRC += $(SRCD)/matrix.cc
SRC += $(SRCD)/h5read.cc
SRC += $(SRCD)/decomp.cc
SRC += $(SRCD)/ExtractMetric.cc
SRC += $(SRCD)/main.cc
#SRC += $(SRCD)/alias.cc

OBJ = $(patsubst $(SRCD)/%.cc,$(OBJD)/%.o,$(SRC))
LIB = $(LIBD)/lib$(LIBNAME).a

INC_PARAMS += -I$(SRCD)
INC_PARAMS += -I$(EX_INCD)

AR = ar

.PHONY: install clean

install: $(EXE)
	@echo '---'
	@echo 'Compiling done!'
	@echo '---'

$(EXE): $(LIB)
	@echo '---'
	@echo 'Linking ...'
	@echo '---'
	$(CXX) $(CXXFLAGS) -o $(EXE) $(LIB) $(EX_LIBS)

$(LIB): $(OBJ)
	@echo '---'
	@echo "Making libraries ... "
	@echo '---'
	@mkdir -p $(LIBD)
## static lib
	$(AR) rcs $(LIB) $(OBJ)

$(OBJD)/%.o: $(SRCD)/%.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(DFLAGS) $(INC_PARAMS) -c $< -o $@

clean:
	@rm -rvf $(OBJD) $(LIBD) $(EXE) *.h5


