#!/bin/bash
###############################################################################

export FN=$(readlink -f "$0"); export DIR_SCRIPTS=$(dirname "${FN}")

cd ${DIR_SCRIPTS}

export dir_out=/tmp/gra_debug/m1
export dir_data=../../outputs/m1_cx/m1_sphere_radabs_MPI0_HYB1_NI1/

ipython -i ${repos}/numerical_relativity/simtroller/cmd_vis_gra.py -- \
  --dir_data ${dir_data}     \
  --dir_out ${dir_out}       \
  --N_B 20                   \
  --sampling x1v             \
  --plot_range [-0.3,1.4]   \
  --range [-4,4]             \
  --parallel_pool 8          \
  --var lab_aux.chi lab.E lab.Fx --plot_show 0

#  --plot_scaling semilogy    \
#  --make_movie 1             \
#  --movie_fps 15

#
# :D
#
