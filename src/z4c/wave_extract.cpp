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
#include <unistd.h>
#include <iomanip>
#include <iostream>

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#include "wave_extract.hpp"
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../globals.hpp"
#include "../parameter_input.hpp"
#include "../mesh/mesh.hpp"
#include "../mesh/spherical_grid.hpp"
#include "z4c.hpp"

WaveExtract::WaveExtract(Mesh * pmesh, ParameterInput * pin, int n):
    pmesh(pmesh), pofile(NULL) {
  int nlev = pin->GetOrAddInteger("psi4_extraction", "nlev", 3);
  Real rad;
  std::string rad_parname;
  rad_parname = "radius_";
  std::string n_str = std::to_string(n);
  rad_parname += n_str;
  rad = pin->GetOrAddReal("psi4_extraction", rad_parname, 10.0);
  rad_id = n;
  ofname = pin->GetOrAddString("psi4_extraction", "filename", "wave");
  lmax = pin->GetOrAddInteger("psi4_extraction", "lmax", 2);
  psi.NewAthenaArray(lmax-1,2*(lmax)+1,2);
  psi.ZeroClear();
  bool bitant = pin->GetOrAddBoolean("mesh", "bitant", false);
  psphere = new SphericalGrid(nlev, rad, bitant);
  ofname += "_r";
  std::stringstream strObj3;
  strObj3 << std::setfill('0') << std::setw(5) << std::fixed << std::setprecision(2) << rad;
  ofname += strObj3.str();
  ofname += ".txt";

  if (0 == Globals::my_rank) {
    // check if output file already exists
    if (access(ofname.c_str(), F_OK) == 0) {
      pofile = fopen(ofname.c_str(), "a");
    }
    else {
      pofile = fopen(ofname.c_str(), "w");
      if (NULL == pofile) {
        std::stringstream msg;
        msg << "### FATAL ERROR in WaveExtract constructor" << std::endl;
        msg << "Could not open file '" << ofname << "' for writing!";
        throw std::runtime_error(msg.str().c_str());
      }
      fprintf(pofile, "# 1:iter 2:time");
      int idx = 3;
      for (int l = 2; l <= lmax; ++l) {
        for (int m = -l; m <= l; ++m) {
          fprintf(pofile, " %d:l=%d-m=%d-Re", idx++, l, m);
          fprintf(pofile, " %d:l=%d-m=%d-Im", idx++, l, m);
        }
      }
      fprintf(pofile, "\n");
      fflush(pofile);
    }
  }
}

WaveExtract::~WaveExtract() {
  delete psphere;
  if (0 == Globals::my_rank) {
    fclose(pofile);
  }
}

void WaveExtract::ReduceMultipole() {
  MeshBlock const * pmb = pmesh->pblock;
  psi.ZeroClear();
  while (pmb != NULL) {
    for(int l=2;l<lmax+1;++l){
      for(int m=-l;m<l+1;++m){
        psi(l-2,m+l,0) +=pmb->pwave_extr_loc[rad_id]->psi(l-2,m+l,0);
        psi(l-2,m+l,1) +=pmb->pwave_extr_loc[rad_id]->psi(l-2,m+l,1);
      }
    }
    pmb = pmb->next;
  }
#ifdef MPI_PARALLEL
  if (0 == Globals::my_rank) {
    for(int l=2;l<lmax+1;++l){
      for(int m=-l;m<l+1;++m){
      MPI_Reduce(MPI_IN_PLACE, &psi(l-2,m+l,0), 1, MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(MPI_IN_PLACE, &psi(l-2,m+l,1), 1, MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
      }
    }
  }
  else {
    for(int l=2;l<lmax+1;++l){
      for(int m=-l;m<l+1;++m){
      MPI_Reduce(&psi(l-2,m+l,0), &psi(l-2,m+l,0), 1, MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(&psi(l-2,m+l,1), &psi(l-2,m+l,1), 1, MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
      }
    }
  }
#endif
}

void WaveExtract::Write(int iter, Real time) const {
  if (0 == Globals::my_rank) {
    fprintf(pofile, "%d %.*g ", iter, FPRINTF_PREC, time);
    for(int l=2;l<lmax+1;++l){
      for(int m=-l;m<l+1;++m){
        fprintf(pofile, "%.*g %.*g ",
                FPRINTF_PREC, psi(l-2,m+l,0),
                FPRINTF_PREC, psi(l-2,m+l,1));
      }
    }
    fprintf(pofile, "\n");
    fflush(pofile);
  }
}

WaveExtractLocal::WaveExtractLocal(SphericalGrid * psphere, MeshBlock * pmb, ParameterInput * pin, int n) {
  std::string rad_parname;
  rad_parname = "radius_";
  std::string n_str = std::to_string(n);
  rad_parname += n_str;
  rad = pin->GetOrAddReal("psi4_extraction", rad_parname.c_str(), 10.0);
  lmax = pin->GetOrAddInteger("psi4_extraction", "lmax", 2);
  psi.NewAthenaArray(lmax-1,2*(lmax)+1,2);
  psi.ZeroClear();
  bitant = pin->GetOrAddBoolean("mesh", "bitant", false);
#if defined(Z4C_VC_ENABLED)
  ppatch = new SphericalPatch(psphere, pmb, SphericalPatch::vertex);
#else
  ppatch = new SphericalPatch(psphere, pmb, SphericalPatch::cell);
#endif
  datareal.NewAthenaArray(ppatch->NumPoints());
  dataim.NewAthenaArray(ppatch->NumPoints());
  weight.NewAthenaArray(ppatch->NumPoints());
  for (int ip = 0; ip < ppatch->NumPoints(); ++ip) {
    weight(ip) = ppatch->psphere->ComputeWeight(ppatch->idxMap(ip));
    weight(ip) /= rad*rad;
  }
}

WaveExtractLocal::~WaveExtractLocal() {
  delete ppatch;
}

void WaveExtractLocal::Decompose_multipole(AthenaArray<Real> const & u_R, AthenaArray<Real> const & u_I) {
    ppatch->InterpToSpherical(u_R, &datareal);
    ppatch->InterpToSpherical(u_I, &dataim);
    Real theta, phi, ylmR, ylmI; //,x,y,z;
    psi.ZeroClear();
    for (int l = 2; l < lmax+1; ++l){
      for (int m = -l; m < l+1 ; ++m){
        Real psilmR = 0.0;
        Real psilmI = 0.0;
          for (int ip = 0; ip < ppatch->NumPoints(); ++ip) {
            ppatch->psphere->GeodesicGrid::PositionPolar(ppatch->idxMap(ip),&theta,&phi);
            swsh(&ylmR,&ylmI,l,m,theta,phi);
            // The spherical harmonics transform as Y^s_{l m}( Pi-th, ph ) = (-1)^{l+s} Y^s_{l -m}(th, ph)
            // but the PoisitionPolar function returns theta \in [0,\pi], so these are correct for bitant.
            // With bitant, under reflection the imaginary part of the weyl scalar should pick a - sign,
            // which is accounted for here.
            Real bitant_z_fac = (bitant && theta > PI/2) ? -1 : 1;
            psilmR += datareal(ip)*weight(ip)*ylmR + bitant_z_fac*dataim(ip)*weight(ip)*ylmI;
            psilmI += bitant_z_fac*dataim(ip)*weight(ip)*ylmR - datareal(ip)*weight(ip)*ylmI;
          }
        psi(l-2,m+l,0) = psilmR;
        psi(l-2,m+l,1) = psilmI;
      }
   }
}

//Factorial
Real WaveExtractLocal::fac(Real n){
 if(n==0 || n==1){
   return 1.0;
 }
 else{
   n=n*fac(n-1);
   return n;
 }
}

//Calculate spin weighted spherical harmonics sw=-2 using Wigner-d matrix notation see e.g. Eq II.7, II.8 in 0709.0093
void WaveExtractLocal::swsh(Real * ylmR, Real * ylmI, int l, int m, Real theta, Real phi){
  Real wignerd = 0;
  int k1,k2,k;
  k1 = std::max(0, m-2);
  k2 = std::min(l+m,l-2);
  for (k = k1; k<k2+1; ++k){
    wignerd += pow((-1),k)*sqrt(fac(l+m)*fac(l-m)*fac(l+2)*fac(l-2))*pow(std::cos(theta/2.0),2*l+m-2-2*k)*pow(std::sin(theta/2.0),2*k+2-m)/(fac(l+m-k)*fac(l-2-k)*fac(k)*fac(k+2-m));
  }
  *ylmR = sqrt((2*l+1)/(4*M_PI))*wignerd*std::cos(m*phi);
  *ylmI = sqrt((2*l+1)/(4*M_PI))*wignerd*std::sin(m*phi);
}

