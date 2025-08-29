# GR-Athena++

Block-based adaptive mesh refinement code for 3+1 numerical relativity.

[GR-Athena++](https://computationalrelativity.github.io/grathenacode/) is a scalable code for 3+1 numerical relativity refactored from the [Athena++](https://www.athena-astro.app/index.html) magnetohydrodynamics code and adaptive mesh refinement (AMR) framework.
The new main features of the code are

 * Support for vertex-centered and cell-centered spacetime variables
 * Dynamical spacetime solver based on the Z4c formulation of 3+1 general relativity and high-order operators
 * Adaptive mesh refinement algorithms for black hole evolutions in the puncture framework
 * GRMHD solver on dynamical spacetimes based on the conservative 3+1 Eulerian “Valencia” formulation and Athena++ constrained transport algorithm
 * Radiation transport solver on dynamical spacetimes based on the M1 formulation
 * Support for various equations of state, including tabular microphysical models
 * Deleptonization scheme for stellar core collapse
 * Gravitational-wave extraction algorithms based on Newmann-Penrose-Weyl and Regge-Wheeler-Zerilli approaches, output for Cauchy Characteristic evolution
 * Apparent horizon finder based on fast flow algorithm
 * Support for various initial data solvers and readers, including Lorene, SGRID and RNS, TwoPunctures


## Terms of use

[GR-Athena++](https://computationalrelativity.github.io/grathenacode/) is distributed in the hope that it will be useful, but without any warranty or support. If you use the code for your research, please cite the relevant method papers listed below.

For reporting potential performance or correctness bugs in the code and algorithm, please open a detailed GitHub Issue. High quality pull requests for bugfixes or new features are accepted on a case-by-case basis. Please contact the relevant code maintainer(s) before expending too much effort on a lengthy PR.


## Documentation 

The code is built upon [Athena++](https://github.com/PrincetonUniversity/athena/wiki).
Documentation on general structure can be found in the [Athena++ Wiki](https://github.com/PrincetonUniversity/athena/wiki),
see for example the [Quick-start guide](https://github.com/PrincetonUniversity/athena/wiki/Quick-Start),
the [Input file nomenclature](https://github.com/PrincetonUniversity/athena/wiki/The-Input-File),
and the [General running details](https://github.com/PrincetonUniversity/athena/wiki/Running-the-Code).

The code is described in the method papers:

 * [GR-Athena++: Puncture Evolutions on Vertex-centered Oct-tree Adaptive Mesh Refinement](https://arxiv.org/abs/2101.08289) Daszuta B., Zappa F., Cook W., Radice D., Bernuzzi S., and Morozova V. Astrophys.J.Supp. 257 (2021) 2, 25 [(bib)](https://ui.adsabs.harvard.edu/abs/2021ApJS..257...25D/exportcitation)
 * [GR-Athena++: General-relativistic magnetohydrodynamics simulations of neutron star spacetimes](https://arxiv.org/abs/2311.04989) Cook W., Daszuta B., Fields J., Hammond P., Albanesi S., Zappa F., Bernuzzi S., and Radice D. Astrophys.J.Supp. 227 (2025) 1, 3 [(bib)](https://ui.adsabs.harvard.edu/abs/2023arXiv231104989C/exportcitation)

To cite [GR-Athena++](https://computationalrelativity.github.io/grathenacode/) in your publication, please use the the above BibTeX and add a reference to the code repo in a footnote.


## Example: Build and run apple-with-apple (AwA) test [TODO: update to working example!]

Build the code:

```bash
# Library requirements (ensure you have these):
#
# hdf5 (located in ${PINST_HDF5})
# gsl (located in ${PINST_GSL})

# Get the repo
git clone https://github.com/computationalrelativity/gr-athena.git
cd gr-athena

# Configure
python configure.py    \
  --prob=z4c_awa_tests \
  -z -z_vc             \
  --cxx g++ -omp       \
  --nghost=4           \
  -hdf5 -h5double -gsl \
  --lib_path=${PINST_HDF5}/lib     \
  --lib_path=${PINST_GSL}/lib      \
  --include=${PINST_HDF5}/include  \
  --include=${PINST_GSL}/include

make clean && make 
```

Brief explanation:

- Configuration script sets up the problem to run (here `z4c_awa_tests`)
- Compiler is selected as `g++` with `OpenMP` threading
- Vertex-centered sampling is chosen; ghost selection is made

Run the code:

```bash
# Point to an input file for the test that lives in `simulations`
export DIR_SIMULATIONS=$PATH_TO_SIMULATIONS_REPO
export DIR_INPUTS=${DIR_SIMULATIONS}/inputs/GR-Athena++/br_master/gr/
mkdir -p outputs/gr_AwA_gauge_wave_1s

# This assumes you are still in the `athena_z4c` folder

# show information about how we compiled
./bin/athena -h

# perform a run with input file from this repository
./bin/athena                                 \
  -i ${DIR_INPUTS}/z4c_AwA_gauge_wave_1s.inp \
  -d outputs/gr_AwA_gauge_wave_1s/
```

Brief explanation:

- Binary is executed with input file `z4c_AwA_gauge_wave_1s.inp` from this repository.
- Output of run is dumped in `outputs/gr_AwA_gauge_wave_1s`.


## Notes for developers (Internal use)

### Branching

 * `master` : production branch, protected - requires PR
 * `feature/*` : development
 * `project/*` : development for specific project
 * `bug/*` : bug fix, delete after merge!
 * `attic/*` : reference/outdated code

### Simulations

Please see https://github.com/computationalrelativity/grathena-simulations
for general information on how to setup simulations of standard problems.

