//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file wave_extract.cpp
//  \brief implementation of functions in the WaveExtract classes

#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../globals.hpp"
#include "../mesh/mesh.hpp"
#include "../mesh/spherical_grid.hpp"
#include "../parameter_input.hpp"
#include "../utils/spherical_harmonics.hpp"
#include "wave_extract.hpp"
#include "z4c.hpp"

WaveExtractHarmonics::WaveExtractHarmonics(SphericalGrid const* psphere,
                                           int lmax,
                                           bool bitant)
{
  num_vertices_ = psphere->NumVertices();
  const int nmodes = (lmax + 1) * (lmax + 1) - 4;
  ylm_.resize(2 * nmodes * num_vertices_);
  bitant_z_fac_.resize(num_vertices_);

  for (int ic = 0; ic < num_vertices_; ++ic)
  {
    Real theta, phi;
    psphere->GeodesicGrid::PositionPolar(ic, &theta, &phi);

    // For bitant reflection,
    // _sY_l^m(pi - theta, phi) =
    // (-1)^(l+s) conj(_sY_l^(-m)(theta, phi)).
    // PositionPolar returns theta in [0,pi]. The existing Weyl extraction
    // flips its imaginary contribution in the southern hemisphere; cache
    // that fixed per-vertex factor here.
    bitant_z_fac_[ic] = (bitant && theta > PI / 2) ? -1.0 : 1.0;

    for (int l = 2; l <= lmax; ++l)
    {
      for (int m = -l; m <= l; ++m)
      {
        const int mode = l*l - 4 + (m + l);
        const int idx = 2 * (mode * num_vertices_ + ic);
        gra::sph_harm::sYlm(
          -2, l, m, theta, phi, &ylm_[idx], &ylm_[idx + 1]);
      }
    }
  }
}

WaveExtract::WaveExtract(Mesh* pmesh, ParameterInput* pin, int n)
    : pmesh(pmesh), pofile(NULL)
{
  int nlev = pin->GetOrAddInteger("psi4_extraction", "nlev", 3);
  Real rad;
  std::string rad_parname;
  rad_parname       = "radius_";
  std::string n_str = std::to_string(n);
  rad_parname += n_str;
  rad    = pin->GetOrAddReal("psi4_extraction", rad_parname, 10.0);
  rad_id = n;
  ofname = pin->GetOrAddString("psi4_extraction", "filename", "wave");
  lmax   = pin->GetOrAddInteger("psi4_extraction", "lmax", 2);
  psi.NewAthenaArray(lmax - 1, 2 * (lmax) + 1, 2);
  psi.ZeroClear();
  bool bitant = pin->GetOrAddBoolean("mesh", "bitant", false);
  psphere     = new SphericalGrid(nlev, rad, bitant);
  ofname += "_r";
  std::stringstream strObj3;
  strObj3 << std::setfill('0') << std::setw(5) << std::fixed
          << std::setprecision(2) << rad;
  ofname += strObj3.str();
  ofname += ".txt";

  if (0 == Globals::my_rank)
  {
    // check if output file already exists
    if (access(ofname.c_str(), F_OK) == 0)
    {
      pofile = fopen(ofname.c_str(), "a");
    }
    else
    {
      pofile = fopen(ofname.c_str(), "w");
      if (NULL == pofile)
      {
        std::stringstream msg;
        msg << "### FATAL ERROR in WaveExtract constructor" << std::endl;
        msg << "Could not open file '" << ofname << "' for writing!";
        throw std::runtime_error(msg.str().c_str());
      }
      fprintf(pofile, "# 1:iter 2:time");
      int idx = 3;
      for (int l = 2; l <= lmax; ++l)
      {
        for (int m = -l; m <= l; ++m)
        {
          fprintf(pofile, " %d:l=%d-m=%d-Re", idx++, l, m);
          fprintf(pofile, " %d:l=%d-m=%d-Im", idx++, l, m);
        }
      }
      fprintf(pofile, "\n");
      fflush(pofile);
    }
  }
}

WaveExtract::~WaveExtract()
{
  delete psphere;
  if (0 == Globals::my_rank)
  {
    fclose(pofile);
  }
}

void WaveExtract::AccumulateMultipole()
{
  const auto& pmb_array = pmesh->GetMeshBlocksCached();
  psi.ZeroClear();
  for (const auto* pmb : pmb_array)
  {
    for (int l = 2; l < lmax + 1; ++l)
    {
      for (int m = -l; m < l + 1; ++m)
      {
        psi(l - 2, m + l, 0) +=
          pmb->pwave_extr_loc[rad_id]->psi(l - 2, m + l, 0);
        psi(l - 2, m + l, 1) +=
          pmb->pwave_extr_loc[rad_id]->psi(l - 2, m + l, 1);
      }
    }
  }
}

void WaveExtract::ReduceAll(std::vector<WaveExtract*>& wave_extractions)
{
  if (wave_extractions.empty())
  {
    return;
  }

  for (auto* pwextr : wave_extractions)
  {
    pwextr->AccumulateMultipole();
  }

#ifdef MPI_PARALLEL
  const int nrad       = static_cast<int>(wave_extractions.size());
  const int psi_size   = wave_extractions.front()->psi.GetSize();
  const int total_size = nrad * psi_size;
  std::vector<Real> psi_all(total_size);

  for (int r = 0; r < nrad; ++r)
  {
    std::memcpy(psi_all.data() + r * psi_size,
                wave_extractions[r]->psi.data(),
                psi_size * sizeof(Real));
  }

  if (0 == Globals::my_rank)
  {
    MPI_Reduce(MPI_IN_PLACE,
               psi_all.data(),
               total_size,
               MPI_ATHENA_REAL,
               MPI_SUM,
               0,
               MPI_COMM_WORLD);
  }
  else
  {
    MPI_Reduce(psi_all.data(),
               nullptr,
               total_size,
               MPI_ATHENA_REAL,
               MPI_SUM,
               0,
               MPI_COMM_WORLD);
  }

  if (0 == Globals::my_rank)
  {
    for (int r = 0; r < nrad; ++r)
    {
      std::memcpy(wave_extractions[r]->psi.data(),
                  psi_all.data() + r * psi_size,
                  psi_size * sizeof(Real));
    }
  }
#endif
}

void WaveExtract::Write(int iter, Real time) const
{
  if (0 == Globals::my_rank)
  {
    fprintf(pofile, "%d %.*g ", iter, FPRINTF_PREC, time);
    for (int l = 2; l < lmax + 1; ++l)
    {
      for (int m = -l; m < l + 1; ++m)
      {
        fprintf(pofile,
                "%.*g %.*g ",
                FPRINTF_PREC,
                psi(l - 2, m + l, 0),
                FPRINTF_PREC,
                psi(l - 2, m + l, 1));
      }
    }
    fprintf(pofile, "\n");
    fflush(pofile);
  }
}

WaveExtractLocal::WaveExtractLocal(SphericalGrid* psphere,
                                   MeshBlock* pmb,
                                   ParameterInput* pin,
                                   int n,
                                   WaveExtractHarmonics const* pwave_harmonics)
    : pwave_harmonics(pwave_harmonics)
{
  std::string rad_parname;
  rad_parname       = "radius_";
  std::string n_str = std::to_string(n);
  rad_parname += n_str;
  rad  = pin->GetOrAddReal("psi4_extraction", rad_parname.c_str(), 10.0);
  lmax = pin->GetOrAddInteger("psi4_extraction", "lmax", 2);
  psi.NewAthenaArray(lmax - 1, 2 * (lmax) + 1, 2);
  psi.ZeroClear();
#if defined(Z4C_VC_ENABLED)
  ppatch = new SphericalPatch(psphere, pmb, SphericalPatch::vertex);
#else
  ppatch = new SphericalPatch(psphere, pmb, SphericalPatch::cell);
#endif
  datareal.NewAthenaArray(ppatch->NumPoints());
  dataim.NewAthenaArray(ppatch->NumPoints());
  weight.NewAthenaArray(ppatch->NumPoints());
  for (int ip = 0; ip < ppatch->NumPoints(); ++ip)
  {
    weight(ip) = ppatch->psphere->ComputeWeight(ppatch->idxMap(ip));
    weight(ip) /= rad * rad;
  }
}

WaveExtractLocal::~WaveExtractLocal()
{
  delete ppatch;
}

void WaveExtractLocal::Decompose_multipole(AthenaArray<Real> const& u_R,
                                           AthenaArray<Real> const& u_I)
{
  ppatch->InterpToSpherical(u_R, &datareal);
  ppatch->InterpToSpherical(u_I, &dataim);
  Real ylmR, ylmI;
  psi.ZeroClear();
  for (int l = 2; l < lmax + 1; ++l)
  {
    for (int m = -l; m < l + 1; ++m)
    {
      Real psilmR = 0.0;
      Real psilmI = 0.0;
      for (int ip = 0; ip < ppatch->NumPoints(); ++ip)
      {
        const int ic = ppatch->idxMap(ip);
        Real const* ylm = pwave_harmonics->Ylm(l, m, ic);
        ylmR = ylm[0];
        ylmI = ylm[1];
        Real bitant_z_fac = pwave_harmonics->BitantZFac(ic);
        psilmR += datareal(ip) * weight(ip) * ylmR +
                  bitant_z_fac * dataim(ip) * weight(ip) * ylmI;
        psilmI += bitant_z_fac * dataim(ip) * weight(ip) * ylmR -
                  datareal(ip) * weight(ip) * ylmI;
      }
      psi(l - 2, m + l, 0) = psilmR;
      psi(l - 2, m + l, 1) = psilmI;
    }
  }
}
