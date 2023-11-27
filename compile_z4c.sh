python3 configure.py \
    -z -vertex \
    -gsl --gsl_path=${GSL_PATH} \
    --nghost=4 --ncghost=4 \
    -hdf5 -h5double  --hdf5_path=${HDF5_PATH} \
    --prob=m1_beam \
    -m1 -debug

#make clean
#make -j
