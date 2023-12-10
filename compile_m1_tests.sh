#export GSL_PATH=/usr/include/gsl/
#export HDF5_PATH=/usr/lib/x86_64-linux-gnu/hdf5/serial

python3 configure.py \
	-z -vertex \
	-gsl --gsl_path=${GSL_PATH} \
	-hdf5 -h5double  --hdf5_path=${HDF5_PATH} \
	--nghost=2 --ncghost=2 \
	--prob=m1_tests \
	-m1 -omp

make clean
make -j 4
