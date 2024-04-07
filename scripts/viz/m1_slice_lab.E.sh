#!/bin/bash
###############################################################################

export FN=$(readlink -f "$0"); export DIR_SCRIPTS=$(dirname "${FN}")

cd ${DIR_SCRIPTS}

export dir_out=/tmp/gra_debug/m1
export dir_data=../../outputs/m1_cx/m1_diffusion_MPI0_HYB1_NI1/

ipython -i ${repos}/numerical_relativity/simtroller/cmd_vis_gra.py -- \
  --dir_data ${dir_data}     \
  --dir_out ${dir_out}       \
  --N_B 10                   \
  --sampling x1v             \
  --plot_range [-0.2, 1.2]   \
  --range [-3,3]             \
  --var lab.E --plot_show 0  \
  --make_movie 1             \
  --movie_fps 15

#
# :D
#
