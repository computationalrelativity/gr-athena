#!/bin/bash
###############################################################################

###############################################################################
# Short script for taking care of (optional) compilation and running a problem
###############################################################################
export FN=$(readlink -f "$0"); export DIR_SCRIPTS=$(dirname "${FN}")

###############################################################################
# configure here
export RUN_NAME=minkowski_vc
export BIN_NAME=z4c
export REL_OUTPUT=outputs/z4c_vc
export REL_INPUT=scripts/problems
export INPUT_NAME=z4c_minkowski.inp

# if compilation is chosen
export DIR_HDF5=$(spack location -i hdf5)
export DIR_GSL=$(spack location -i gsl)

export COMPILE_STR="--prob=z4c_awa_tests
                    -z -z_vc
                    --cxx g++ -omp
                    --nghost=4
                    --ncghost=5
                    --ncghost_cx=5
                    --nextrapolate=4"

# apply caching compiler together with gold linker
export COMPILE_STR="${COMPILE_STR} -ccache -link_gold"

# hdf5 compile str
export COMPILE_STR="${COMPILE_STR} -hdf5 -h5double"
export COMPILE_STR="${COMPILE_STR} -gsl"

export COMPILE_STR="${COMPILE_STR} --lib_path=${DIR_HDF5}/lib"
export COMPILE_STR="${COMPILE_STR} --include=${DIR_HDF5}/include"

export COMPILE_STR="${COMPILE_STR} --lib_path=${DIR_GSL}/lib"
export COMPILE_STR="${COMPILE_STR} --include=${DIR_GSL}/include"
###############################################################################

echo "COMPILE_STR"
echo ${COMPILE_STR}

###############################################################################
# ensure paths are adjusted and directory structure exists
source ${DIR_SCRIPTS}/utils/provide_library_paths.sh ${DIR_HDF5}
source ${DIR_SCRIPTS}/utils/provide_library_paths.sh ${DIR_GSL}
source ${DIR_SCRIPTS}/utils/provide_compile_paths.sh
###############################################################################

###############################################################################
# prepare external
# ...
###############################################################################

###############################################################################
# compile
source ${DIR_SCRIPTS}/utils/compile_athena.sh
###############################################################################

###############################################################################
# dump information
source ${DIR_SCRIPTS}/utils/dump_info.sh
###############################################################################

###############################################################################
# execute
source utils/exec.sh
###############################################################################

tail -n5 ${DIR_OUTPUT}/minkowski.hst

# >:D
