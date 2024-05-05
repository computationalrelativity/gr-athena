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

export USE_CX=1
export NINTERP=1
export USE_HYBRIDINTERP=1
export DIR_TAG="MPI${USE_MPI}_HYB${USE_HYBRIDINTERP}_NI${NINTERP}"

export RIEMANN_SOLVER=llftaudyn
# export RIEMANN_SOLVER=hlletaudyn
# export RIEMANN_SOLVER=marquinataudyn

export BIN_NAME=z4c_tov
export REL_INPUT=scripts/run/inputs
export INPUT_NAME=grhd/z4c_tov.inp
export RUN_NAME=grhd_tov_${DIR_TAG}
# pass to executable on cmdline
export GRA_CMD=""

# uncomment to use this restart segment
# export USE_RESTART="00000"

export FIELD_VAR="z"

export REL_OUTPUT=outputs/${FIELD_VAR}_
if [ $USE_CX == 1 ]
then
  export REL_OUTPUT="${REL_OUTPUT}cx"
else
  export REL_OUTPUT="${REL_OUTPUT}vc"
fi

export COMPILE_STR="--prob=gr_tov
                    --coord=gr_dynamical
                    --eos=eostaudyn_ps
                    --eospolicy=idealgas
                    --errorpolicy=reset_floor
                    --flux=${RIEMANN_SOLVER}
                    -${FIELD_VAR}
                    -g -f
                    --cxx g++ -omp
                    --nghost=4
                    --ncghost=4
                    --ncghost_cx=4
                    --nextrapolate=4
                    --ninterp=${NINTERP}"

# complete COMPILE_STR specification
export USE_REPRIMAND=1
export USE_BOOST=1

source ${DIR_SCRIPTS}/utils/provide_compile_str_libs.sh
###############################################################################

echo "COMPILE_STR:"
echo ${COMPILE_STR}

###############################################################################
# ensure directory structure exists
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
source ${DIR_SCRIPTS}/utils/call_exec.sh
###############################################################################

# >:D
