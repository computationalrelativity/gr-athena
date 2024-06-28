#!/bin/bash
###############################################################################

###############################################################################
# execute
cd ${DIR_OUTPUT}

export TOTAL_TASKS=5
export TASKS_PER_NODE=1
export OMP_NUM_THREADS=2

# export TOTAL_TASKS=1
# export TASKS_PER_NODE=1
# export OMP_NUM_THREADS=1


if [ $RUN_MODE == 1 ]
then
  export dir_ompi=$(spack location -i openmpi)
  export dir_valg=$(spack location -i valgrind)
  LD_PRELOAD=${dir_valg}/lib/valgrind/libmpiwrap-amd64-linux.so \
  mpirun \
    -np ${TOTAL_TASKS} \
    --bind-to none \
    --oversubscribe \
    -x OMP_NUM_THREADS=${OMP_NUM_THREADS} \
    valgrind \
     --leak-check=full --show-reachable=yes --track-origins=yes \
     --log-file=vg.log.%p \
     --suppressions=${dir_ompi}/share/openmpi/openmpi-valgrind.supp \
    ./${EXEC_NAME}.x  -r ${1} ${2} \
    mesh/num_threads=${OMP_NUM_THREADS} \
    ${1}
else
  mpirun \
    -np ${TOTAL_TASKS} \
    --bind-to none \
    --oversubscribe \
    -x OMP_NUM_THREADS=${OMP_NUM_THREADS} \
    ./${EXEC_NAME}.x  -r ${1} ${2} \
    mesh/num_threads=${OMP_NUM_THREADS} \
    ${1}
fi

echo "Done >:D"
cd ${DIR_SCRIPTS}
###############################################################################

# >:D
