#!/bin/bash

# module load TACC

# intel
# module load intel/19.1.1

# gcc
# module load gcc
# module load mkl

# module load phdf5/1.10.4
# module load impi/19.0.9
# module load mvapich2-x/2.3
# module load cmake/3.24.2
# module load gsl/2.6
# module load fftw3

# gsl_p=${TACC_GSL_DIR}
# phdf5_p=${TACC_HDF5_DIR}
# fftw_p=${TACC_FFTW3_DIR}

# gcc only
# mkl_i="${MKLROOT}/include"
# mkl_l="${MKLROOT}/lib/intel64"

# module list

phdf5_d="/usr/lib/x86_64-linux-gnu/hdf5/openmpi"

# Configure athena_z4c for parallel evolution, HD, compose EoS, migration ID
python3 configure.py \
  --prob gr_tov -bgfz -z_cx --coord=gr_dynamical --flux=llftaudyn --nghost=4 --ncghost=4 --ncghost_cx=4 --eos=eostaudyn_ps --eospolicy=eos_compose --errorpolicy=reset_floor --nscalars=1\
  --nextrapolate=4 -hybridinterp --ninterp=1\
  -mpi -omp --cxx=g++ --mpiccmd=mpicxx.openmpi\
  -gsl \
  -fft \
  -hdf5 -h5double --hdf5_path=${phdf5_d} \
  --lib mkl_intel_lp64 --lib mkl_core --lib mkl_gnu_thread --lib pthread \
  -debug

# intel only
#  --cflag="-mkl"

# gcc only
#  --include=${mkl_i} --lib_path=${mkl_l} \
#  --lib mkl_intel_lp64 --lib mkl_core --lib mkl_gnu_thread --lib pthread

# clean whatever is currently there
make clean

# build configuration
make -j 4
