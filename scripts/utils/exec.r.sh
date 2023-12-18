#!/bin/bash
###############################################################################

###############################################################################
# execute
# cd ${DIR_ATHENA}/${REL_OUTPUT}/${RUN_NAME}
cd ${DIR_OUTPUT}

time ./${EXEC_NAME}.x -r gr_hydro_debug.final.rst time/tlim=50 mesh/refinement=adaptive

echo "Done >:D"
cd ${DIR_SCRIPTS}
###############################################################################

# >:D
