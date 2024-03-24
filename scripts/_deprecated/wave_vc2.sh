#!/bin/bash
###############################################################################

###############################################################################
# Short script for taking care of (optional) compilation and running a problem
###############################################################################
export FN=$(readlink -f "$0"); export DIR_SCRIPTS=$(dirname "${FN}")

# export DIR_INSTALL_MUST=/mnt/nimbus/_Installed/MUST
# export PATH=${DIR_INSTALL_MUST}/bin:$PATH

###############################################################################
# configure here
export RUN_NAME=vc
export BIN_NAME=wave_vc
export REL_OUTPUT=outputs/wave_2d
export REL_INPUT=scripts/problems

# Will be populated with defaults instead.
export INPUT_NAME=cvg_wave_2d.inp

# if compilation is chosen
export DIR_HDF5=$(spack location -i hdf5)

export COMPILE_STR="--prob=wave_2d_cvg_trig -w -w_vc
                    --cxx g++-simd
                    --nghost=4
                    --ncghost=5
                    --ncghost_cx=5
                    --nextrapolate=4"

# debug
# export COMPILE_STR="${COMPILE_STR} -debug"

# apply caching compiler together with gold linker
export COMPILE_STR="${COMPILE_STR} -ccache -link_gold"

# hdf5 compile str
export COMPILE_STR="${COMPILE_STR} -hdf5 -h5double"
export COMPILE_STR="${COMPILE_STR} --lib_path=${DIR_HDF5}/lib"
export COMPILE_STR="${COMPILE_STR} --include=${DIR_HDF5}/include"

echo "COMPILE_STR"
echo ${COMPILE_STR}
###############################################################################

###############################################################################
# ensure paths are adjusted and directory structure exists
source ${DIR_SCRIPTS}/utils/provide_library_paths.sh ${DIR_HDF5}

source ${DIR_SCRIPTS}/utils/provide_compile_paths.sh
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

tail -n5 ${DIR_OUTPUT}/wave_2d.hst

# >:D
