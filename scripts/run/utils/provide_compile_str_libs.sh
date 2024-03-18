#!/bin/bash
###############################################################################

# if compilation is chosen
export DIR_HDF5=$(spack location -i hdf5)
export DIR_GSL=$(spack location -i gsl)

# check variable not unset or empty
if ! [ -z "${USE_REPRIMAND}" ] && [ ${USE_REPRIMAND} == 1 ]
then
  export DIR_REP=$(spack location -i reprimand)
fi

if ! [ -z "${USE_BOOST}" ] && [ ${USE_BOOST} == 1 ]
then
  export DIR_BOS=$(spack location -i boost)
fi

if ! [ -z "${USE_FFTW}" ] && [ ${USE_FFTW} == 1 ]
then
  export DIR_FFTW=$(spack location -i fftw)
fi

if ! [ -z "${USE_LORENE}" ] && [ ${USE_LORENE} == 1 ]
then
  export DIR_LOR=$(spack location -i lorene)
fi

if ! [ -z "${USE_TWOPUNCTURESC}" ] && [ ${USE_TWOPUNCTURESC} == 1 ]
then
  export DIR_TP=$(spack location -i twopuncturesc)
fi

# this is needed...
export DIR_GCC=$(spack location -i gcc)

if [ $USE_HYBRIDINTERP == 1 ]
then
  export COMPILE_STR="${COMPILE_STR} -hybridinterp"
fi

if [ $USE_CX == 1 ]
then
  export COMPILE_STR="${COMPILE_STR} -z_cx"
else
  export COMPILE_STR="${COMPILE_STR} -z_vc"
fi

# apply caching compiler together with gold linker
export COMPILE_STR="${COMPILE_STR} -ccache -link_gold"

# hdf5 compile str
export COMPILE_STR="${COMPILE_STR} -hdf5 -h5double"
export COMPILE_STR="${COMPILE_STR} -gsl"

export COMPILE_STR="${COMPILE_STR} --lib_path=${DIR_HDF5}/lib"
export COMPILE_STR="${COMPILE_STR} --include=${DIR_HDF5}/include"

export COMPILE_STR="${COMPILE_STR} --lib_path=${DIR_GSL}/lib"
export COMPILE_STR="${COMPILE_STR} --include=${DIR_GSL}/include"

if ! [ -z "${USE_REPRIMAND}" ] && [ ${USE_REPRIMAND} == 1 ]
then
  export COMPILE_STR="${COMPILE_STR} --lib_path=${DIR_REP}/lib"
  export COMPILE_STR="${COMPILE_STR} --include=${DIR_REP}/include"
fi

if ! [ -z "${USE_BOOST}" ] && [ ${USE_BOOST} == 1 ]
then
  export COMPILE_STR="${COMPILE_STR} --lib_path=${DIR_BOS}/lib"
  export COMPILE_STR="${COMPILE_STR} --include=${DIR_BOS}/include"
fi

if ! [ -z "${USE_FFTW}" ] && [ ${USE_FFTW} == 1 ]
then
  export COMPILE_STR="${COMPILE_STR} --lib_path=${DIR_FFTW}/lib"
  export COMPILE_STR="${COMPILE_STR} --include=${DIR_FFTW}/include"
  export COMPILE_STR="${COMPILE_STR} --lib=fftw3"
fi

if ! [ -z "${USE_LORENE}" ] && [ ${USE_LORENE} == 1 ]
then
  export COMPILE_STR="${COMPILE_STR} -lorene"
  export COMPILE_STR="${COMPILE_STR} --lib_path=${DIR_LOR}/lib"
  export COMPILE_STR="${COMPILE_STR} --include=${DIR_LOR}/include"
fi

if ! [ -z "${USE_TWOPUNCTURESC}" ] && [ ${USE_TWOPUNCTURESC} == 1 ]
then
  export COMPILE_STR="${COMPILE_STR} --lib_path=${DIR_TP}/lib"
  export COMPILE_STR="${COMPILE_STR} --include=${DIR_TP}/include"
fi

if [ $RUN_MODE -gt 0 ]
then
  export COMPILE_STR="${COMPILE_STR} -debug"
fi

if [ $USE_MPI == 1 ]
then
  export COMPILE_STR="${COMPILE_STR} -mpi"
fi


###############################################################################
# ensure paths are adjusted

source ${DIR_SCRIPTS}/utils/provide_library_paths.sh ${DIR_GCC}
source ${DIR_SCRIPTS}/utils/provide_library_paths.sh ${DIR_HDF5}
source ${DIR_SCRIPTS}/utils/provide_library_paths.sh ${DIR_GSL}
source ${DIR_SCRIPTS}/utils/provide_library_paths.sh ${DIR_REP}
source ${DIR_SCRIPTS}/utils/provide_library_paths.sh ${DIR_BOS}


###############################################################################

# >:D
