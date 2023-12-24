#!/bin/bash
###############################################################################

###############################################################################
# execute
# cd ${DIR_ATHENA}/${REL_OUTPUT}/${RUN_NAME}
cd ${DIR_OUTPUT}

time ./${EXEC_NAME}.x -r z4c_grhd_tov_boost.final.rst time/tlim=50 mesh/refinement=adaptive

echo "Done >:D"
cd ${DIR_SCRIPTS}
###############################################################################

# >:D
