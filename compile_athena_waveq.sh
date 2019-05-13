if [ -z "$1" ]; then
    echo "Usage: $0 dimensions[1|2|3] nghost[1|2|3] parallelization[mpi|omp](optional)"
    echo "E.g. 'bash $0 3 2 mpi omp' will generate bin/athena_wave_3d_2g_mpi_omp"
    exit 1
fi

STR="--prob=wave_test_$1d -w --nghost=$2"
NEWNAME="athena_wave_"$1"d_"$2"g"
if [ -n "$3" ]; then
    STR=$STR" -$3"
    NEWNAME=$NEWNAME"_$3"
fi
if [ -n "$4" ]; then
    STR=$STR" -$4"
    NEWNAME=$NEWNAME"_$4"
fi

make clean
python configure.py $STR 
make -j4
mv bin/athena bin/$NEWNAME
