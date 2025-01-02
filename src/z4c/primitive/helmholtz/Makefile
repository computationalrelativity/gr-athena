FC       := gfortran
FFLAGS   += -O0 -g -Wall -Wpedantic
LIB      := ar crs
FLAGS    :=

INCLUDE = -I/usr/local/include

ATHENA_ROOT = "../../../.."
SRC_FILES = quintic_hermite.f90 helmholtz.f90
OBJ_DIR := $(ATHENA_ROOT)/obj/
LIB_DIR := $(ATHENA_ROOT)/lib/
OBJ_FILES := $(addprefix $(OBJ_DIR),$(notdir $(SRC_FILES:.f90=.o)))

LIBRARY_NAME := libhelmholtz.a

LIBRARY := $(LIB_DIR)$(LIBRARY_NAME)

.PHONY: all dirs clean

all : dirs $(LIBRARY)

objs : dirs $(OBJ_FILES)

dirs : $(LIB_DIR) $(OBJ_DIR)

$(LIB_DIR):
	mkdir -p $(LIB_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(LIBRARY) : $(OBJ_FILES)
	$(LIB) $(LIBRARY) $(OBJ_FILES)

$(OBJ_DIR)helmhotz.o : $(OBJ_DIR)quintic_hermite.o

$(OBJ_DIR)%.o : %.f90
	$(FC) $(FFLAGS) $(FLAGS) $(INCLUDE) -c $< -o $@ -J $(OBJ_DIR)

clean :
	rm -rf $(OBJ_DIR)*
	rm -rf $(LIBRARY)

