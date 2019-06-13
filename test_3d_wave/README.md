Test 3d wave equation with SMR:

* Configure athena running "python configure.py -w --prob=wave_test_3d -omp"
* Compile athena
* Link athena here "ln -s ../bin/athena ./"
* Link the input file here "ln -s ../input/wave/athinput.wave3d_3reflev". There are two different input files for this test:
'athinput.wave3d_2reflev' will refine with blocks of the same size while 'athinput.wave3d_3reflev' will refine in such a way that meshblocks with different size coexist. In the latter case issues arise.
