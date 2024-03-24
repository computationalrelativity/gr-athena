#!/bin/bash
###############################################################################

###############################################################################
# execute
cd ${DIR_OUTPUT}

if [ $RUN_MODE == 1 ]
then
  # make call log [inspect with qcachegrind]
  # valgrind --tool=callgrind --log-file=callgrind.log \
  #   ./${EXEC_NAME}.x -i ${DIR_ATHENA}/${REL_INPUT}/${INPUT_NAME}
  valgrind                   \
    --leak-check=full -s     \
    --show-leak-kinds=all    \
    --track-origins=yes      \
    --log-file=valgrind.log  \
    ./${EXEC_NAME}.x -r ${1} ${2}
  # valgrind --leak-check=full -s \
  #   ./${EXEC_NAME}.x -i ${DIR_ATHENA}/${REL_INPUT}/${INPUT_NAME} -m 2

elif [ $RUN_MODE == 2 ]
then
  gdb \
    -ex 'break main.cpp:1' \
    -ex 'info b' \
    -ex 'set print pretty on' \
    -ex 'set output-radix 10' \
    -ex=r --args ./${EXEC_NAME}.x -r ${1} ${2}

else
  time ./${EXEC_NAME}.x -r ${1} ${2}
fi

echo "Done >:D"
cd ${DIR_SCRIPTS}
###############################################################################

# >:D
