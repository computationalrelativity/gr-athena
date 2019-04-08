make clean
python configure.py --prob=wave_test -w --nghost=2
make -j4
mv bin/athena bin/athena_sinusoidal_profile_2D
