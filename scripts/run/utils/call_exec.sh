#!/bin/bash
###############################################################################

if [ -z "${USE_RESTART}" ]
then
  # not using restarts
  if [ $USE_MPI == 1 ]
  then
    source utils/mpi_exec.sh ${GRA_CMD}
  else
    source utils/exec.sh ${GRA_CMD}
  fi
else
  # using restarts
  export FN_RESTART="${BIN_NAME}.${USE_RESTART}.rst"

  if [ $USE_MPI == 1 ]
  then
    source utils/mpi_exec_restart.sh ${FN_RESTART} ${GRA_CMD}
  else
    source utils/exec_restart.sh ${FN_RESTART} ${GRA_CMD}
  fi

fi

###############################################################################

# >:D
