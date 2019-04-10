if [ -z "$1" ]; then
    echo "Usage: $0 name_athena_exe"
    exit 1
fi

make clean
python configure.py --prob=wave_test -w --nghost=2
make -j4
mv bin/athena bin/$1
