athena
======
<!-- Jenkins Status Badge in Markdown (with view), unprotected, flat style -->
<!-- In general, need to be on Princeton VPN, logged into Princeton CAS, with ViewStatus access to Jenkins instance to click on unprotected Build Status Badge, but server is configured to whitelist GitHub -->
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

<!--[![Public GitHub  issues](https://img.shields.io/github/issues/PrincetonUniversity/athena-public-version.svg)](https://github.com/PrincetonUniversity/athena-public-version/issues)
[![Public GitHub pull requests](https://img.shields.io/github/issues-pr/PrincetonUniversity/athena-public-version.svg)](https://github.com/PrincetonUniversity/athena-public-version/pulls) -->

Athena++ radiation MHD code


matter-new-ps-branch
==================
Branch for integrating the new primitive solver and EOS framework into GR-Athena++.

Flags for running with Z4c and hydrodynamics:
`-gfz`
`--coord=gr_dynamical`
`--flux=llftaudyn`
`-vertex`
`--nghost=4`
`--ncghost=4`

Flags for enabling the new EOS framework:
`--eos=eostaudyn_ps`
`--eospolicy=idealgas`
`--errorpolicy=reset_floor`

Compatible problem generators:
`--prob=gr_tov`
`--prob=gr_neutron_star`

Other configure flags:
`-gsl`
`-omp` (Optional)
`-mpi`
`--cmd=mpicxx`
`-hdf5` (Optional, requires two following arguments)
`-h5double`
`--hdf5_path=/path/to/hdf5`
`-debug` (Currently required due to bugs)
`--lorene_path=$HOME_LORENE` (Required for `gr_neutron_star`)

Compatible input files can be found in `inputs/z4c/stars` and `inputs/hydro_dyn`
