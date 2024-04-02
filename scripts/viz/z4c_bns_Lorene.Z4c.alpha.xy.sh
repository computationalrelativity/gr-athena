#!/bin/bash
###############################################################################

export FN=$(readlink -f "$0"); export DIR_SCRIPTS=$(dirname "${FN}")

cd ${DIR_SCRIPTS}

export dir_out=/tmp/gra_debug/grhd_bns_Lorene
export dir_data=../../outputs/z_vc/grhd_bns_Lorene_MPI1_HYB1_NI1

ipython -i ${repos}/numerical_relativity/simtroller/cmd_vis_gra.py -- \
  --dir_data ${dir_data}      \
  --dir_out ${dir_out}        \
  --N_B 16 16                 \
  --sampling x1f x2f          \
  --range [-32,32]            \
  --var z4c.alpha

#
# :D
#
