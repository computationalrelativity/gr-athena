#!/bin/bash
###############################################################################

export FN=$(readlink -f "$0"); export DIR_SCRIPTS=$(dirname "${FN}")

cd ${DIR_SCRIPTS}

export dir_out=/tmp/gra_debug/m1
export dir_data=../../outputs/m1_cx/m1_diffusion_MPI0_HYB1_NI1/

ipython -i ${repos}/numerical_relativity/simtroller/cmd_vis_gra.py -- \
  --dir_data ${dir_data}     \
  --dir_out ${dir_out}       \
  --N_B 14                   \
  --sampling x1v             \
  --range [-1,1]             \
  --var lab.E --plot_show 1

#
# :D
#