#!/bin/bash
###############################################################################

###############################################################################
# execute
# cd ${DIR_ATHENA}/${REL_OUTPUT}/${RUN_NAME}
cd ${DIR_OUTPUT}

# ./${EXEC_NAME}.x -i ${DIR_ATHENA}/${REL_INPUT}/${INPUT_NAME}

# echo "> Executing: ${EXEC_NAME} in ${REL_OUTPUT}/${RUN_NAME} ..."
# echo "> Using input: ${REL_INPUT}/${INPUT_NAME} ..."


mustrun -np 4 --must:stacktrace dyninst --must:nocrash ${EXEC_NAME}.x -i ${DIR_ATHENA}/${REL_INPUT}/${INPUT_NAME}

# mustrun -np 4 --must:stacktrace dyninst --must:hybrid ${EXEC_NAME}.x -i ${DIR_ATHENA}/${REL_INPUT}/${INPUT_NAME}

# mustrun -np 4 --must:stacktrace dyninst --must:tsan --must:nocrash ${EXEC_NAME}.x -i ${DIR_ATHENA}/${REL_INPUT}/${INPUT_NAME}


# mpirun -np 4 ${EXEC_NAME}.x -r /mnt/nebula/_Repositories/NR/athena/development/outputs/z4c_c_tp/two_puncture/z4c.final.rst


# source /mnt/nebula/_Repositories/comp_environment/hephaestus_deployer/run_phoenix_gcc_mpi.sh
# LD_PRELOAD=/mnt/nebula/_Software/usr/gcc/valgrind/3.17.0/lib/valgrind/libmpiwrap-amd64-linux.so   \
#   mpirun -np 2 \
#     ./${EXEC_NAME}.x -i ${DIR_ATHENA}/${REL_INPUT}/${INPUT_NAME} \
#     time/tlim=0

# source /mnt/nebula/_Repositories/comp_environment/hephaestus_deployer/run_phoenix_gcc_mpi.sh
#  LD_PRELOAD=/mnt/nebula/_Software/usr/gcc/valgrind/3.17.0/lib/valgrind/libmpiwrap-amd64-linux.so   \
#    mpirun -np 2 \
#     valgrind \
#      --leak-check=full --show-reachable=yes --track-origins=yes \
#      --log-file=vg.log.%p \
#      --suppressions=/mnt/nebula/_Software/usr/gcc/openmpi/4.1.1/share/openmpi/openmpi-valgrind.supp \
#      ./${EXEC_NAME}.x -r /mnt/nebula/_Repositories/NR/athena/development/outputs/z4c_c_tp/two_puncture/z4c.00000.rst \
#      time/tlim=1 \
#      mesh/numlevel=2




# export OMPI_MCA_mpi_param_check=1
# export OMPI_MCA_mpi_show_handle_leaks=1

# mpirun -np 2 \
#   ./${EXEC_NAME}.x -i ${DIR_ATHENA}/${REL_INPUT}/${INPUT_NAME}




# perf record -g ./${EXEC_NAME}.x -i ${DIR_ATHENA}/${REL_INPUT}/${INPUT_NAME}
# perf report --children -g 'graph,0.5,caller' --dsos=${EXEC_NAME}.x -U

# gdb -ex 'info b' \
#     -ex 'set print pretty on' \
#     -ex=r --args ./${EXEC_NAME}.x -i ${DIR_ATHENA}/${REL_INPUT}/${INPUT_NAME}


# gdb -ex 'break wave_1d_cvg_trig.cpp:104' \
# gdb -q \
#     -ex 'break bvals_refine.cpp:227' \
#     -ex 'info b' \
#     -ex 'set print pretty on' \
#     -ex=r --args ./$EXEC_NAME.x -i $DIR_ATHENA/$REL_INPUT/$INPUT_NAME


# make call log [inspect with qcachegrind]
#valgrind --tool=callgrind --log-file=callgrind.log ./${EXEC_NAME}.x -i ${DIR_ATHENA}/${REL_INPUT}/${INPUT_NAME}
# valgrind --leak-check=full -s --show-leak-kinds=all ./${EXEC_NAME}.x -i ${DIR_ATHENA}/${REL_INPUT}/${INPUT_NAME} time/tlim=0
# valgrind --leak-check=full -s ./${EXEC_NAME}.x -i ${DIR_ATHENA}/${REL_INPUT}/${INPUT_NAME} -m 2

echo "Done >:D"
cd ${DIR_SCRIPTS}
###############################################################################

# >:D
