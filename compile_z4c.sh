python3 configure.py \
    -z -vertex \
    -gsl --gsl_path=${GSL_PATH} \
    --nghost=4 --ncghost=4 \
    -hdf5 -h5double  --hdf5_path=${HDF5_PATH} \
    --prob=shock_tube \
    -m1

#make clean
#make -j
