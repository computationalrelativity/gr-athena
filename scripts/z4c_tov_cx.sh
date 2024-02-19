#!/bin/bash
###############################################################################

###############################################################################
# Short script for taking care of (optional) compilation and running a problem
###############################################################################
export FN=$(readlink -f "$0"); export DIR_SCRIPTS=$(dirname "${FN}")

###############################################################################
# configure here
export NINTERP=1
export USE_DIRECT=0
export TAG="64"

export RIEMANN_SOLVER=llftaudyn
# export RIEMANN_SOLVER=hlletaudyn
# export RIEMANN_SOLVER=marquinataudyn

if [ $USE_DIRECT == 1 ]
then
  export RUN_NAME=gra_tov_cx_ninterp${NINTERP}_dir_${TAG}
else
  export RUN_NAME=gra_tov_cx_ninterp${NINTERP}_${RIEMANN_SOLVER}_${TAG}
fi

export BIN_NAME=z4c
export REL_OUTPUT=outputs/z4c_cx
export REL_INPUT=scripts/problems
export INPUT_NAME=z4c_tov.inp

# if compilation is chosen
export DIR_HDF5=$(spack location -i hdf5)
export DIR_GSL=$(spack location -i gsl)
export DIR_REP=$(spack location -i reprimand)
export DIR_BOS=$(spack location -i boost)
# this is needed...
export DIR_GCC=$(spack location -i gcc)

export COMPILE_STR="--prob=gr_tov
                    --coord=gr_dynamical
                    --eos=adiabatictaudyn_rep
                    --flux=${RIEMANN_SOLVER}
                    -z -g -f -z_cx
                    --cxx g++ -omp
                    --nghost=4
                    --ncghost=4
                    --ncghost_cx=4
                    --nextrapolate=4
                    -hybridinterp
                    -DBG_MA_SOURCES
                    --ninterp=${NINTERP}"

# apply caching compiler together with gold linker
# export COMPILE_STR="${COMPILE_STR} -ccache -link_gold"
export COMPILE_STR="${COMPILE_STR} -ccache"

# hdf5 compile str
export COMPILE_STR="${COMPILE_STR} -hdf5 -h5double"
export COMPILE_STR="${COMPILE_STR} -gsl"

export COMPILE_STR="${COMPILE_STR} --lib_path=${DIR_HDF5}/lib"
export COMPILE_STR="${COMPILE_STR} --include=${DIR_HDF5}/include"

export COMPILE_STR="${COMPILE_STR} --lib_path=${DIR_GSL}/lib"
export COMPILE_STR="${COMPILE_STR} --include=${DIR_GSL}/include"

export COMPILE_STR="${COMPILE_STR} --lib_path=${DIR_REP}/lib"
export COMPILE_STR="${COMPILE_STR} --include=${DIR_REP}/include"

export COMPILE_STR="${COMPILE_STR} --lib_path=${DIR_BOS}/lib"
export COMPILE_STR="${COMPILE_STR} --include=${DIR_BOS}/include"

###############################################################################

echo "COMPILE_STR"
echo ${COMPILE_STR}

###############################################################################
# ensure paths are adjusted and directory structure exists
source ${DIR_SCRIPTS}/utils/provide_library_paths.sh ${DIR_GCC}
source ${DIR_SCRIPTS}/utils/provide_library_paths.sh ${DIR_HDF5}
source ${DIR_SCRIPTS}/utils/provide_library_paths.sh ${DIR_GSL}
source ${DIR_SCRIPTS}/utils/provide_library_paths.sh ${DIR_REP}
source ${DIR_SCRIPTS}/utils/provide_library_paths.sh ${DIR_BOS}
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


# >:D
