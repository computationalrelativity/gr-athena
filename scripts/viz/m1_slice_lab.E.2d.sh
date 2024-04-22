#!/bin/bash
###############################################################################

export FN=$(readlink -f "$0"); export DIR_SCRIPTS=$(dirname "${FN}")

cd ${DIR_SCRIPTS}

export dir_out=/tmp/gra_debug/m1

export dir_data=../../outputs/m1_cx/m1_shadow_MPI0_HYB1_NI1/

ipython -i ${repos}/numerical_relativity/simtroller/cmd_vis_gra.py -- \
  --dir_data ${dir_data}     \
  --dir_out ${dir_out}       \
  --N_B 20 20                \
  --plot_range [-0.1, 1.2]   \
  --sampling x1v x2v         \
  --parallel_pool 16         \
  --var lab.E

#  --range [0,5] [0,4]      \

#  --make_movie 1             \
#  --movie_fps 15

# export dir_data=../../outputs/m1_cx/m1_shadow_MPI0_HYB1_NI1/

# ipython -i ${repos}/numerical_relativity/simtroller/cmd_vis_gra.py -- \
#  --dir_data ${dir_data}     \
#  --dir_out ${dir_out}       \
#  --N_B 20 20                \
#  --plot_range [-0.1, 1.1]   \
#  --sampling x1v x2v         \
#  --parallel_pool 8          \
#  --var lab.E


#
# :D
#
