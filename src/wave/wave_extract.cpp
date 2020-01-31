//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file wave_extract.cpp
//  \brief implementation of functions in the WaveExtract classes

#include <cstdio>
#include <stdexcept>
#include <sstream>

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#include "wave_extract.hpp"
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../mesh/mesh.hpp"
#include "../mesh/spherical_grid.hpp"

WaveExtract::WaveExtract(Mesh * pmesh, ParameterInput * pin):
    pmesh(pmesh), pofile(NULL) {
  int nlev = pin->GetOrAddInteger("wave", "extraction_nlev", 3);
  Real rad = pin->GetOrAddReal("wave", "extraction_radius", 1.0);
  ofname = pin->GetOrAddString("wave", "extract_filename", "wave.txt");
  root = pin->GetOrAddInteger("wave", "mpi_root", 0);

  psphere = new SphericalGrid(nlev, rad);

#ifdef MPI_PARALLEL
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  io_proc = (root == rank);
#else
  ioproc = true;
#endif

  if (ioproc) {
    pofile = fopen(ofname.c_str(), "w");
    if (NULL == pofile) {
      std::stringstream msg;
      msg << "### FATAL ERROR in WaveExtract constructor" << std::endl;
      msg << "Could not open file '" << ofname << "' for writing!";
      throw std::runtime_error(msg.str().c_str());
    }
    fprintf(pofile, "# 1:iter 2:time 3:l=0\n");
  }
}

WaveExtract::~WaveExtract() {
  delete psphere;
  if (ioproc) {
    fclose(pofile);
  }
}

void WaveExtract::ReduceMonopole() {
  monopole = 0.;
  MeshBlock const * pmb = pmesh->pblock;
  while (pmb != NULL) {
    monopole += pmb->pwave_extr_loc->monopole;
    pmb = pmb->next;
  }
#ifdef MPI_PARALLEL
  int rank;
  MPI_comm_rank(MPI_COMM_WORLD, &rank);
  if (root == rank) {
    MPI_Reduce(MPI_IN_PLACE, &monopole, 1, MPI_ATHENA_REAL, MPI_SUM, root, MPI_COMM_WORLD);
  }
  else {
    MPI_Reduce(&monopole, &monopole, 1, MPI_ATHENA_REAL, MPI_SUM, root, MPI_COMM_WORLD);
  }
#endif
}

void WaveExtract::Write(int iter, Real time) const {
  if (ioproc) {
    fprintf(pofile, "%d %g %g\n", iter, time, monopole);
  }
}

WaveExtractLocal::WaveExtractLocal(SphericalGrid * psphere, MeshBlock * pmb, ParameterInput * pin) {
  ppatch = new SphericalPatch(psphere, pmb, SphericalPatch::cell);
  data.NewAthenaArray(ppatch->NumPoints());
  weight.NewAthenaArray(ppatch->NumPoints());
  for (int ip = 0; ip < ppatch->NumPoints(); ++ip) {
    weight(ip) = ppatch->psphere->ComputeWeight(ppatch->idxMap(ip));
  }
}

WaveExtractLocal::~WaveExtractLocal() {
  delete ppatch;
}

void WaveExtractLocal::Decompose(AthenaArray<Real> const & u) {
  ppatch->InterpToSpherical(u, &data);
  monopole = 0.0;
  for (int ip = 0; ip < ppatch->NumPoints(); ++ip) {
    monopole += data(ip)*weight(ip);
  }
}
