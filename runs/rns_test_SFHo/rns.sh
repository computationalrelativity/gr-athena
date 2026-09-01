rm -rf bin/athena_rns_SFHo

# Configure problem
export NGHOST=4
export NEXTR=5
export SAMP=cx
export NINTERP=3
export PGEN=gr_rns

# Define your library base directories clearly
export DIR_USR_HDF5=$HOME/local/hdf5
export DIR_USR_GSL=/usr

export COMPILE_STR="--prob=${PGEN}
                    -z -z_${SAMP}
                    -gsl
                    --eospolicy=eos_compose
                    --gsl_path=${DIR_USR_GSL}
                    -hdf5 -h5double
                    --hdf5_path=${DIR_USR_HDF5}
                    --cxx g++
                    -omp -mpi
                    --nghost=${NGHOST}
                    --ncghost_cx=${NGHOST}
                    --ncghost=${NGHOST}
                    --nextrapolate=${NEXTR}
                    -f -g
                    --nscalars=1
                    --coord=gr_dynamical
                    --errorpolicy=reset_floor"

python3 configure.py ${COMPILE_STR}

make clean && make -j${MAKE_THREADS}

mv bin/athena bin/athena_rns_SFHo