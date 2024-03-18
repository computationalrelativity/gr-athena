#!/bin/bash
###############################################################################

###############################################################################
# Short script for taking care of (optional) compilation and running a problem
###############################################################################
export FN=$(readlink -f "$0"); export DIR_SCRIPTS=$(dirname "${FN}")

###############################################################################
# configure here

# 0 - normal, 1 - valgrind, 2 - gdb, ...
export RUN_MODE=0
export USE_MPI=0

export NINTERP=1
export USE_HYBRIDINTERP=1
export DIR_TAG="MPI${USE_MPI}_HYB${USE_HYBRIDINTERP}_NI${NINTERP}"

export RIEMANN_SOLVER=llftaudyn
# export RIEMANN_SOLVER=hlletaudyn
# export RIEMANN_SOLVER=marquinataudyn

export BIN_NAME=z4c
export REL_OUTPUT=outputs/z4c_cx
export REL_INPUT=scripts/run/inputs
export INPUT_NAME=z4c_tov.inp
export RUN_NAME=gr_tov_cx_${DIR_TAG}

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
                    --ninterp=${NINTERP}"

if [ $USE_HYBRIDINTERP == 1 ]
then
  export COMPILE_STR="${COMPILE_STR} -hybridinterp"
fi

# apply caching compiler together with gold linker
export COMPILE_STR="${COMPILE_STR} -ccache -link_gold"

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

if [ $RUN_MODE -gt 0 ]
then
  export COMPILE_STR="${COMPILE_STR} -debug"
fi

if [ $USE_MPI == 1 ]
then
  export COMPILE_STR="${COMPILE_STR} -mpi"
fi

###############################################################################

echo "COMPILE_STR:"
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

if [ $USE_MPI == 1 ]
then
  source utils/mpi_exec.sh
else
  source utils/exec.sh
fi

###############################################################################


# >:D
