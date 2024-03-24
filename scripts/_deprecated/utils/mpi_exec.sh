#!/bin/bash
###############################################################################

###############################################################################
# execute
# cd ${DIR_ATHENA}/${REL_OUTPUT}/${RUN_NAME}
cd ${DIR_OUTPUT}

# ./${EXEC_NAME}.x -i ${DIR_ATHENA}/${REL_INPUT}/${INPUT_NAME}

# echo "> Executing: ${EXEC_NAME} in ${REL_OUTPUT}/${RUN_NAME} ..."
# echo "> Using input: ${REL_INPUT}/${INPUT_NAME} ..."



# mpirun -np 2 ${EXEC_NAME}.x -r /mnt/nebula/_Repositories/NR/athena/development/outputs/z4c_c_tp/two_puncture/z4c.final.rst


# OMP_NUM_THREADS=thds
# mpirun -np Total_Tasks --bind-to core
# --map-by ppr:Tasks_Per_Node:node:pe=thds
# -x OMP_NUM_THREADS

export TOTAL_TASKS=8
export TASKS_PER_NODE=1
export OMP_NUM_THREADS=1


# mpirun -np Total_Tasks --bind-to core
# --map-by ppr:Tasks_Per_Node:node:pe=thds
# -x OMP_NUM_THREADS

# mpirun \
#   -np ${TOTAL_TASKS} \
#   --bind-to core \
#   --map-by ppr:${TASKS_PER_NODE}:node:pe=${OMP_NUM_THREADS} \
#   -x OMP_NUM_THREADS=${OMP_NUM_THREADS} \
#   ./${EXEC_NAME}.x -i ${DIR_ATHENA}/${REL_INPUT}/${INPUT_NAME}

mpirun \
  -np ${TOTAL_TASKS} \
  --bind-to none \
  --oversubscribe \
  -x OMP_NUM_THREADS=${OMP_NUM_THREADS} \
  ./${EXEC_NAME}.x -i ${DIR_ATHENA}/${REL_INPUT}/${INPUT_NAME}

# # less aggressive
# mpirun \
#   -np ${TOTAL_TASKS} \
#   --bind-to none \
#   --oversubscribe \
#   -x OMP_NUM_THREADS=${OMP_NUM_THREADS} \
#   ./${EXEC_NAME}.x -r gr_Lor.00000.rst \
#   mesh/num_threads=${OMP_NUM_THREADS} \
#   time/cfl_number=0.25 \
#   problem/fatm=1e-20 \
#   problem/fthr=1 \
#   hydro/c2p_acc=1e-12 \
#   hydro/rho_strict=0 \
#   z4c/diss=0.1 \
#   z4c/shift_eta=0.3 \
#   z4c/chi_div_floor=1e-12 \
#   z4c/eps_floor=0 \
#   z4c/shift_Gamma=0.75 \
#   z4c/shift_advect=1.0 \
#   z4c/damp_kappa1=0.02 \
#   z4c/damp_kappa2=0.0 \
#   z4c/r_max_con=30 \
#   problem/max_z=100 \
#   time/xorder=weno5z \
#   time/integrator=rk3 \
#   time/tlim=15 \
#   psi4_extraction/filename=wav \
#   psi4_extraction/num_radii=0 \
#   task_triggers/dt_psi4_extraction=0.5 \
#   trackers_extrema/tol_ds=1e-13 \
#   trackers_extrema/iter_max=25 \
#   trackers_extrema/interp_ds_fac=10

# # aggressive
# mpirun \
#   -np ${TOTAL_TASKS} \
#   --bind-to none \
#   --oversubscribe \
#   -x OMP_NUM_THREADS=${OMP_NUM_THREADS} \
#   ./${EXEC_NAME}.x -r gr_Lor.00000.rst \
#   mesh/num_threads=${OMP_NUM_THREADS} \
#   time/cfl_number=0.25 \
#   problem/fatm=1e-18 \
#   problem/fthr=1e2 \
#   hydro/c2p_acc=1e-12 \
#   hydro/rho_strict=0 \
#   z4c/diss=0.02 \
#   z4c/shift_eta=0.3 \
#   z4c/shift_Gamma=0.75 \
#   z4c/shift_advect=1.0 \
#   z4c/r_max_con=30 \
#   problem/max_z=100 \
#   time/xorder=weno5d_si \
#   time/integrator=ssprk5_4 \
#   psi4_extraction/filename=wav \
#   psi4_extraction/num_radii=0 \
#   task_triggers/dt_psi4_extraction=0.5
  # \
  # trackers_extrema/ref_level_1=1 \
  # trackers_extrema/ref_level_2=1 \
  # trackers_extrema/ref_zone_radius_1=10 \
  # trackers_extrema/ref_zone_radius_2=10 \
  # mesh/refinement=adaptive


#  time/integrator=ssprk5_4 \

#  time/xorder=3

#   time/xorder_style=weno5z \


# valgrind
# export DIR_VG=$(spack location -i valgrind)

# # LD_PRELOAD=${DIR_VG}/lib/valgrind/libmpiwrap-amd64-linux.so
# mpirun -np 2 \
#   valgrind \
#     ./${EXEC_NAME}.x -i ${DIR_ATHENA}/${REL_INPUT}/${INPUT_NAME}


# --leak-check=full --show-reachable=yes --track-origins=yes \
# --log-file=vg.log.%p \

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
