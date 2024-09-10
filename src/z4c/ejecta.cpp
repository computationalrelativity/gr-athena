//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file ahf.cpp
//  \brief implementation of the apparent horizon finder class
// Developed from BAM's AHmod, see also
//  https://git.tpi.uni-jena.de/sbernuzzi/ahfpy

#include <cstdio>
#include <stdexcept>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <unistd.h>
#include <cmath> // NAN

#define DEBUG_OUTPUT 0 

#include "ejecta.hpp"
#include "../hydro/hydro.hpp"
#include "../field/field.hpp"
#include "z4c.hpp"
#include "z4c_macro.hpp"
#include "../globals.hpp"
#include "../parameter_input.hpp"
#include "../mesh/mesh.hpp"
#include "../utils/tensor.hpp"
#include "../coordinates/coordinates.hpp"
#include "../utils/interp_intergrid.hpp"
#include "../eos/eos.hpp"


using namespace utils::tensor;

std::string prim_names[NHYDRO] = {
  "rho", "vx", "vy", "vz", "press",
};

std::string cons_names[NHYDRO] = {
  "dens", "Mx", "My", "Mz", "tau",
};

std::string adm_names[Z4c::N_ADM] = {
  "gxx", "gxy", "gxz", "gyy", "gyz", "gzz",
  "Kxx", "Kxy", "Kxz", "Kyy", "Kyz", "Kzz",
  "psi4",
};

std::string z4c_names[Z4c::N_Z4c] = {
  "chi",
  "gxx", "gxy", "gxz", "gyy", "gyz", "gzz",
  "Khat",
  "Axx", "Axy", "Axz", "Ayy", "Ayz", "Azz",
  "Gamx", "Gamy", "Gamz",
  "Theta",
  "alpha",
  "betax", "betay", "betaz",
};

std::string other_names[Ejecta::NOTHER] = {
  "detg", "Mdot","bernoulli","enthalpy","entropy","lorentz","u_t","fD_r","v_mag","poynting"
};

//----------------------------------------------------------------------------------------
//! \fn Ejecta::Ejecta(Mesh * pmesh, ParameterInput * pin, int n)
//  \brief class for ejecta extraction class
Ejecta::Ejecta(Mesh * pmesh, ParameterInput * pin, int n):
  pmesh(pmesh) 
{
std::string int_names[n_int] = {
  "mass_", "entr_", "rho_", "temp_", "ye_", "vel_", "ber_","velinf_"
};

std::string hist_names[n_hist] = {
  "entr_","logrho_","temp_","ye_","vel_","ber_","theta_","velinf_"
};

std::string unbound_names[n_unbound] = {
  "bernoulli_", "bernoulli_outflow_", "geodesic_", "geodesic_outflow_"
};
  nrad = pin->GetOrAddInteger("ejecta", "num_rad", 1);
  nr = n;
  std::string parname;
  std::string n_str = std::to_string(nr);
  
  ntheta = pin->GetOrAddInteger("ejecta", "ntheta", 10);
  nphi = pin->GetOrAddInteger("ejecta", "nphi", 6);
  if ((nphi+1)%2==0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in Ejecta setup" << std::endl
        << "nphi must be even " << nphi << std::endl;
    ATHENA_ERROR(msg);
  }

  verbose = pin->GetOrAddBoolean("ejecta", "verbose", 0);
  root = pin->GetOrAddInteger("ejecta", "mpi_root", 0);
  bitant = pin->GetOrAddBoolean("z4c", "bitant", false);
  
  parname = "radius_";
  parname += n_str;
  radius = pin->GetOrAddReal("ejecta", parname, 300);

  //parname = "compute_every_iter_";
  //parname += n_str;
  //compute_every_iter = pin->GetOrAddInteger("ejecta", parname, 1);
  compute_every_iter = pin->GetOrAddInteger("ejecta", "compute_every_iter", 10);

  //parname = "start_time_";
  //parname += n_str;
  //start_time = pin->GetOrAddReal("ejecta", parname, 0.0);
  start_time = pin->GetOrAddReal("ejecta", "start_time", 0.0);
  
  //parname = "stop_time_";
  //parname += n_str;
  //stop_time = pin->GetOrAddReal("ejecta", parname, 10000.0);
  stop_time = pin->GetOrAddReal("ejecta", "stop_time", 10000.0);

  theta.NewAthenaArray(ntheta);
  phi.NewAthenaArray(nphi);
  mass_contained = 0.0;

  for (int n=0; n<NHYDRO; ++n) {
    prim[n].NewAthenaArray(ntheta, nphi);
    cons[n].NewAthenaArray(ntheta, nphi);
  }
  for (int n=0; n<NDIM; ++n) {
    Bcc[n].NewAthenaArray(ntheta, nphi);
  }
#if USETM
  for (int n=0; n<NSCALARS; ++n) {
    Y[n].NewAthenaArray(ntheta, nphi);
  }
  T.NewAthenaArray(ntheta, nphi);
#endif

  for (int n=0; n<Z4c::N_ADM; ++n) {
    adm[n].NewAthenaArray(ntheta, nphi);
  }
  for (int n=0; n<Z4c::N_Z4c; ++n) {
    z4c[n].NewAthenaArray(ntheta, nphi);
  }
  for (int n=0; n<NOTHER; ++n) {
    other[n].NewAthenaArray(ntheta, nphi);
  }

  for (int i=0; i<ntheta; ++i) {
    theta(i)   = th_grid(i);
  }

  for (int j=0; j<nphi; ++j) {
    phi(j)   = ph_grid(j);
  }

  // Flag points existing on this mesh
  havepoint.NewAthenaArray(ntheta, nphi);
// n iterates over unboundedness criteria, m over variables, l over bins in histogram
// i over theta j over phi

//  I_hist_entr,I_hist_logrho,I_hist_temp,I_hist_ye,I_hist_vel,I_hist_ber,I_hist_theta,I_hist_velinf
  Real  def_max[n_hist] = {200.0,-2.5,10.0,0.55,1.0,1.0,M_PI,1.0};
  Real  def_min[n_hist] = {0.0,-15.0,0.0,0.035,0.0,0.0,0.0,0.0};
  Real max_hist[n_hist], min_hist[n_hist];
  for (int m=0; m<n_hist; ++m) {
    parname = "hist_n_";
    parname += int_names[m];
    n_bins[m] = pin->GetOrAddInteger("ejecta", parname, 50);
    parname = "hist_max_";
    parname += int_names[m];
    max_hist[m] = pin->GetOrAddReal("ejecta", parname, def_max[m]);
    parname = "hist_min_";
    parname += int_names[m];
    min_hist[m] = pin->GetOrAddReal("ejecta", parname, def_min[m]);
  } 
  
  for (int m=0; m<n_hist; ++m) {
    hist_grid[m].NewAthenaArray(n_bins[m]);
  }
  for (int m=0; m<n_hist; ++m) {
    delta_hist[m] = (max_hist[m] - min_hist[m])/n_bins[m];
    for (int l=0; l<n_bins[m]; ++l) {
      hist_grid[m](l) = min_hist[m] + l * delta_hist[m]; 
    } 
  }

  
  integrals_unbound.NewAthenaArray(n_unbound,n_int);
  az_integrals_unbound.NewAthenaArray(n_unbound,n_int,ntheta);
  for (int m=0; m<n_hist; ++m) {
    hist[m].NewAthenaArray(n_unbound,n_bins[m]);
  }


  // Prepare output
  parname = "ejecta_file_summary_";
  parname += n_str;
  ofname_summary = pin->GetString("job", "problem_id") + ".";
  ofname_summary += pin->GetOrAddString("ejecta", parname, "ejecta_summary_"+n_str);
  ofname_summary += ".txt";

  for (int n = 0; n<n_unbound; ++n){
    parname = "ejecta_file_";
    parname += unbound_names[n];
    parname += n_str;
    ofname_unbound[n] = pin->GetString("job", "problem_id") + ".";
    ofname_unbound[n] += pin->GetOrAddString("ejecta", parname, parname);
    ofname_unbound[n] += ".txt";
  }
  for (int n = 0; n<n_unbound; ++n){
    parname = "ejecta_file_az_";
    parname += unbound_names[n];
    parname += n_str;
    ofname_az_unbound[n] = pin->GetString("job", "problem_id") + ".";
    ofname_az_unbound[n] += pin->GetOrAddString("ejecta", parname, parname);
    ofname_az_unbound[n] += ".txt";
  }
  for (int n = 0; n<n_unbound; ++n){
    for(int m = 0; m<n_hist; ++m){
      parname = "ejecta_file_histogram_";
      parname += unbound_names[n];
      parname += hist_names[m];
      parname += n_str;
      ofname_hist_unbound[n][m] = pin->GetString("job", "problem_id") + ".";
      ofname_hist_unbound[n][m] += pin->GetOrAddString("ejecta", parname, parname);
      ofname_hist_unbound[n][m] += ".txt";
    }
  }

#ifdef MPI_PARALLEL
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ioproc = (root == rank);
#else
  ioproc = true;
#endif
  if (ioproc) {
    // Summary file
    bool new_file = true;
    if (access(ofname_summary.c_str(), F_OK) == 0) {
      new_file = false;
    }
    pofile_summary = fopen(ofname_summary.c_str(), "a");
    if (NULL == pofile_summary) {
      std::stringstream msg;
      msg << "### FATAL ERROR in AHF constructor" << std::endl;
      msg << "Could not open file '" << pofile_summary << "' for writing!";
      throw std::runtime_error(msg.str().c_str());
    }
    if (new_file) {
      fprintf(pofile_summary, "# 1:iter 2:time 3:rho 4:press 5:Ye 6:vx 7:vy 8:vz 9:Bx 10:By 11:Bz\n");
      fflush(pofile_summary);
    }

    for (int n = 0; n < n_unbound; ++n){
      new_file = true;
      if (access(ofname_unbound[n].c_str(), F_OK) == 0) {
        new_file = false;
      }
      pofile_unbound[n] = fopen(ofname_unbound[n].c_str(), "a");
      if (NULL == pofile_unbound[n]) {
        std::stringstream msg;
        msg << "### FATAL ERROR in Ejecta constructor" << std::endl;
        msg << "Could not open file '" << pofile_unbound[n] << "' for writing!";
        throw std::runtime_error(msg.str().c_str());
      }
      if (new_file) {
        fprintf(pofile_unbound[n], "# 1:iter 2:time 3:mass 4:entropy 5:rho 6:temperature 7:Ye 8:velocity 9:bernoulli 10:velocity_inf \n");
        fflush(pofile_unbound[n]);
      }
    }

    for (int n = 0; n < n_unbound; ++n){
      new_file = true;
      if (access(ofname_az_unbound[n].c_str(), F_OK) == 0) {
        new_file = false;
      }
      pofile_az_unbound[n] = fopen(ofname_az_unbound[n].c_str(), "a");
      if (NULL == pofile_az_unbound[n]) {
        std::stringstream msg;
        msg << "### FATAL ERROR in Ejecta constructor" << std::endl;
        msg << "Could not open file '" << pofile_az_unbound[n] << "' for writing!";
        throw std::runtime_error(msg.str().c_str());
      }
      if (new_file) {
        fprintf(pofile_az_unbound[n], "# 1:theta 2:mass 3:entropy 4:rho 5:temperature 6:Ye 7:velocity 8:bernoulli 9:velocity_inf \n");
        fflush(pofile_az_unbound[n]);
      }
    }
    for (int n = 0; n < n_unbound; ++n){
      for (int m = 0; m < n_hist; ++m){
        new_file = true;
        if (access(ofname_hist_unbound[n][m].c_str(), F_OK) == 0) {
          new_file = false;
        }
        pofile_hist_unbound[n][m] = fopen(ofname_hist_unbound[n][m].c_str(), "a");
        if (NULL == pofile_hist_unbound[n][m]) {
          std::stringstream msg;
          msg << "### FATAL ERROR in Ejecta constructor" << std::endl;
          msg << "Could not open file '" << pofile_hist_unbound[n][m] << "' for writing!";
          throw std::runtime_error(msg.str().c_str());
        }
        if (new_file) {
          fprintf(pofile_hist_unbound[n][m], "# 1:bin 2:weight \n");
          fflush(pofile_hist_unbound[n][m]);
        }
      }
    }
  }
}

Ejecta::~Ejecta() {

  for (int n=0; n<NHYDRO; ++n) {
    prim[n].DeleteAthenaArray();
    cons[n].DeleteAthenaArray();
  }
#if USETM
  T.DeleteAthenaArray();
  for (int n=0; n<NSCALARS; ++n) {
    Y[n].DeleteAthenaArray();
  }
#endif
  for (int n=0; n<3; ++n) {
    Bcc[n].DeleteAthenaArray();
  }

  for (int n=0; n<Z4c::N_ADM; ++n) {
    adm[n].DeleteAthenaArray();
  }
  for (int n=0; n<Z4c::N_Z4c; ++n) {
    z4c[n].DeleteAthenaArray();
  }
  for (int n=0; n<NOTHER; ++n) {
    other[n].DeleteAthenaArray();
  }

  // Flag points existing on this mesh
  havepoint.DeleteAthenaArray();

  // Close files
  if (ioproc) {
    fclose(pofile_summary);
    for (int n = 0; n < n_unbound; ++n){
      fclose(pofile_az_unbound[n]);
      fclose(pofile_unbound[n]);
      for(int m = 0; m < n_hist; ++m){
        fclose(pofile_hist_unbound[n][m]);
      }
    }



  }
}

void Ejecta::Mass(MeshBlock * pmb)
{
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  for (int k=ks; k<=ke; k++) {
    Real z = pmb->pcoord->x3v(k);
    for (int j=js; j<=je; j++) {
      Real y = pmb->pcoord->x2v(j);
      for (int i=is; i<=ie; i++) {
        Real x = pmb->pcoord->x1v(i);
        Real vol = (pmb->pcoord->dx1v(i)) * (pmb->pcoord->dx2v(j)) * (pmb->pcoord->dx3v(k));
        Real rad2 = x*x + y*y + z*z;
        if (rad2 < SQR(radius)) {
          mass_contained += vol * pmb->phydro->u(IDN,k,j,i);
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
// \!fn void Ejecta::Interp(MeshBlock * pmb)
// \brief interpolate quantities on the surface n
// Flag here the surface points contained in the MB
void Ejecta::Interp(MeshBlock * pmb)
{

  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  LagrangeInterpND<2*NGHOST-1, 3> * pinterp3 = nullptr;
  LagrangeInterpND<2*NGHOST-1, 3> * pinterpvc3 = nullptr;
//  LagrangeInterpND<2*NGHOST-1, 3> * pinterp3_fx = nullptr;
//  LagrangeInterpND<2*NGHOST-1, 3> * pinterp3_fy = nullptr;
//  LagrangeInterpND<2*NGHOST-1, 3> * pinterp3_fz = nullptr;

  AthenaArray<Real> prim_[NHYDRO], cons_[NHYDRO], T_, Y_[NSCALARS], Bcc_[NDIM];
  AthenaArray<Real> vc_adm_[Z4c::N_ADM], vc_z4c_[Z4c::N_Z4c], adm_[Z4c::N_ADM], z4c_[Z4c::N_Z4c];
  AthenaArray<Real> flux_[NDIM];

  TensorPointwise<Real, Symmetries::SYM2, NDIM, 2> adm_g_dd;
  adm_g_dd.NewTensorPointwise();

  for (int n=0; n<NHYDRO; ++n) {
    prim_[n].InitWithShallowSlice(pmb->phydro->w, IDN+n, 1);
    cons_[n].InitWithShallowSlice(pmb->phydro->u, IDN+n, 1);
  }
#if USETM
  for(int n=0; n<NSCALARS; ++n){
    Y_[n].InitWithShallowSlice(pmb->pscalars->r, IYF+n, 1);
  }
  //T not in ghosts - rather than comm Temp, calculate from other prims on sphere
//  T_.InitWithShallowSlice(pmb->phydro->temperature, ITM, 1);
#endif

  for (int n=0; n<NDIM; ++n) {
    flux_[n].InitWithShallowSlice(pmb->phydro->flux[n], IDN, 1);
    Bcc_[n].InitWithShallowSlice(pmb->pfield->bcc, IB1+n, 1);
  }

  for (int n=0; n<Z4c::N_ADM; ++n) {
    vc_adm_[n].InitWithShallowSlice(pmb->pz4c->storage.adm, Z4c::I_ADM_gxx+n, 1);
    adm_[n].NewAthenaArray(pmb->ncells3, pmb->ncells2, pmb->ncells1);
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          adm_[n](k,j,i) = VCInterpolation(vc_adm_[n], k, j, i);
        }
      }
    }
  }

  for (int n=0; n<Z4c::N_Z4c; ++n) {
    vc_z4c_[n].InitWithShallowSlice(pmb->pz4c->storage.u, Z4c::I_Z4c_chi+n, 1);
    z4c_[n].NewAthenaArray(pmb->ncells3, pmb->ncells2, pmb->ncells1);
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          z4c_[n](k,j,i) = VCInterpolation(vc_z4c_[n], k, j, i);
        }
      }
    }
  }

  // For interp
  Real origin[NDIM] = {pmb->pcoord->x1v(0), pmb->pcoord->x2v(0), pmb->pcoord->x3v(0)};
  Real delta[NDIM] = {pmb->pcoord->dx1v(0), pmb->pcoord->dx2v(0), pmb->pcoord->dx3v(0)};
  int size[NDIM] = {pmb->ncells1, pmb->ncells2, pmb->ncells3};
  Real originvc[NDIM] = {pmb->pcoord->x1f(0), pmb->pcoord->x2f(0), pmb->pcoord->x3f(0)};
  Real deltavc[NDIM] = {pmb->pcoord->dx1f(0), pmb->pcoord->dx2f(0), pmb->pcoord->dx3f(0)};
  int sizevc[NDIM] = {pmb->nverts1, pmb->nverts2, pmb->nverts3};

  Real origin_f[NDIM] = {pmb->pcoord->x1f(0), pmb->pcoord->x2f(0), pmb->pcoord->x3f(0)};
  Real delta_f[NDIM] = {pmb->pcoord->dx1f(0), pmb->pcoord->dx2f(0), pmb->pcoord->dx3f(0)};
  int size_fx[NDIM] = {pmb->ncells1+1, pmb->ncells2, pmb->ncells3};
  int size_fy[NDIM] = {pmb->ncells1, pmb->ncells2+1, pmb->ncells3};
  int size_fz[NDIM] = {pmb->ncells1, pmb->ncells2, pmb->ncells3+1};
  Real coord[NDIM];
 
  for (int i=0; i<ntheta; i++) {
    Real sinth = std::sin(theta(i));
    Real costh = std::cos(theta(i));
    for (int j=0; j<nphi; j++) {
      Real sinph = std::sin(phi(j));
      Real cosph = std::cos(phi(j));
     
      coord[0]  = radius * sinth * cosph;
      coord[1]  = radius * sinth * sinph;
      coord[2]  = radius * costh;
      // Impose bitant symmetry below
      bool bitant_sym = ( bitant && coord[2] < 0 ) ? true : false;
      // Associate z -> -z if bitant
      if (bitant) coord[2] = std::abs(coord[2]);

      if (!pmb->PointContained(coord[0], coord[1], coord[2])) continue;

      // this surface point is in this MB
      havepoint(i,j) += 1;
      
      // Interpolate
      pinterp3 = new LagrangeInterpND<2*NGHOST-1, 3>(origin, delta, size, coord);
      pinterpvc3 = new LagrangeInterpND<2*NGHOST-1, 3>(originvc, deltavc, sizevc, coord);

      for (int n=0; n<NHYDRO; ++n) {
        prim[n](i,j) = pinterp3->eval(&(prim_[n](0,0,0)));
        cons[n](i,j) = pinterp3->eval(&(cons_[n](0,0,0)));
      }
#if USETM
      for (int n=0; n<NSCALARS; ++n) {
        Y[n](i,j) = pinterp3->eval(&(Y_[n](0,0,0)));
      }
//      T(i,j) = pinterp3->eval(&(T_(0,0,0)));
      Real Ypt[NSCALARS];
      for (int n=0; n<NSCALARS; ++n){
          Ypt[n] = Y[n](i,j);
      }
      Real npt= prim[IDN](i,j) / pmb->peos->GetEOS().GetBaryonMass();
      Real Wvu[3] = {prim[IVX](i,j),prim[IVY](i,j),prim[IVZ](i,j)}; 
      Real prpt = prim[IPR](i,j);
      pmb->peos->GetEOS().ApplyDensityLimits(npt); 
      pmb->peos->GetEOS().ApplySpeciesLimits(Ypt); 
      T(i,j) = pmb->peos->GetEOS().GetTemperatureFromP(npt, prpt, Ypt);
#endif
      if (MAGNETIC_FIELDS_ENABLED) {
        for (int n=0; n<NDIM; ++n) {
          Bcc[n](i,j) = pinterp3->eval(&(Bcc_[n](0,0,0)));
        }
      }

      for (int n=0; n<Z4c::N_ADM; ++n) {
        adm[n](i,j) = pinterpvc3->eval(&(vc_adm_[n](0,0,0)));
//        adm[n](i,j) = pinterp3->eval(&(adm_[n](0,0,0)));
      }
      for (int n=0; n<Z4c::N_Z4c; ++n) {
        z4c[n](i,j) = pinterpvc3->eval(&(vc_z4c_[n](0,0,0)));
//        z4c[n](i,j) = pinterp3->eval(&(z4c_[n](0,0,0)));
      }
      other[I_detg](i,j) = SpatialDet(adm[0](i,j), adm[1](i,j), adm[2](i,j),
                                 adm[3](i,j), adm[4](i,j), adm[5](i,j));

      /*
      pinterp3_fx = new LagrangeInterpND<2*NGHOST-1, 3>(origin_f, delta_f, size_fx, coord);
      pinterp3_fy = new LagrangeInterpND<2*NGHOST-1, 3>(origin_f, delta_f, size_fy, coord);
      pinterp3_fz = new LagrangeInterpND<2*NGHOST-1, 3>(origin_f, delta_f, size_fz, coord);

      Real fx = pinterp3->eval(&(flux_[0](0,0,0)));
      Real fy = pinterp3->eval(&(flux_[1](0,0,0)));
      Real fz = pinterp3->eval(&(flux_[2](0,0,0)));
*/
      adm_g_dd(0,0) = adm[Z4c::I_ADM_gxx](i,j);
      adm_g_dd(0,1) = adm[Z4c::I_ADM_gxy](i,j);
      adm_g_dd(0,2) = adm[Z4c::I_ADM_gxz](i,j);
      adm_g_dd(1,1) = adm[Z4c::I_ADM_gyy](i,j);
      adm_g_dd(1,2) = adm[Z4c::I_ADM_gyz](i,j);
      adm_g_dd(2,2) = adm[Z4c::I_ADM_gzz](i,j);
      Mdot_total += other[1](i,j);
      other[I_lorentz](i,j) = 1.0;
      for (int a = 0; a<NDIM; ++a){      
        for (int b = 0; b<NDIM; ++b){      
          other[I_lorentz](i,j) += adm_g_dd(a,b) * prim[IVX+a](i,j) * prim[IVX+b](i,j);
        }
      }
      other[I_lorentz](i,j) = sqrt(other[I_lorentz](i,j));


      Real vx = prim[IVX](i,j) / other[I_lorentz](i,j);
      Real vy = prim[IVY](i,j) / other[I_lorentz](i,j);
      Real vz = prim[IVZ](i,j) / other[I_lorentz](i,j);

      Real fx = cons[IDN](i,j) * (z4c[Z4c::I_Z4c_alpha](i,j) * vx - z4c[Z4c::I_Z4c_betax](i,j));
      Real fy = cons[IDN](i,j) * (z4c[Z4c::I_Z4c_alpha](i,j) * vy - z4c[Z4c::I_Z4c_betay](i,j));
      Real fz = cons[IDN](i,j) * (z4c[Z4c::I_Z4c_alpha](i,j) * vz - z4c[Z4c::I_Z4c_betaz](i,j));

      other[I_fD_r](i,j) = fx * cosph * sinth + fy * sinph * sinth + fz * costh;
      other[I_Mdot](i,j) = MassLossRate(fx, fy, fz, sinth, costh, sinph, cosph);
      other[I_v_mag](i,j) = sqrt(1.0 - 1.0/SQR(other[I_lorentz](i,j)));


      other[I_u_t](i,j) = - other[I_lorentz](i,j) * z4c[Z4c::I_Z4c_alpha](i,j);
      for (int a = 0; a<NDIM; ++a){
        for (int b = 0; b<NDIM; ++b){
          other[I_u_t](i,j) += adm_g_dd(a,b) * z4c[Z4c::I_Z4c_betax+a](i,j) * prim[IVX+b](i,j);
	}
      }
      Real temppt = T(i,j);
      pmb->peos->GetEOS().ApplyPrimitiveFloor(npt, Wvu, prpt, temppt,Ypt);
      other[I_enthalpy](i,j) = pmb->peos->GetEOS().GetEnthalpy(npt,temppt,Ypt); 
      other[I_entropy](i,j) = pmb->peos->GetEOS().GetEntropy(npt,temppt,Ypt); 
      other[I_bernoulli](i,j) = - other[I_enthalpy](i,j) * other[I_u_t](i,j) - 1.0; 
      Real bi_u[3];
      Real b0_u = 0.0;
      for (int a = 0; a<NDIM; ++a){
        for (int b = 0; b<NDIM; ++b){
            b0_u += Bcc[a](i,j)*prim[IVX+b](i,j)*adm_g_dd(a,b) / z4c[Z4c::I_Z4c_alpha](i,j);
	}
      }
      for (int a = 0; a<NDIM; ++a){
            bi_u[a] = Bcc[i](i,j)/other[I_lorentz](i,j) + z4c[Z4c::I_Z4c_alpha](i,j)* b0_u*prim[IVX+a](i,j) - b0_u*z4c[Z4c::I_Z4c_betax+a](i,j);
      }

      Real bsq = SQR(z4c[Z4c::I_Z4c_alpha](i,j)* b0_u);
      for (int a = 0; a<NDIM; ++a){
          for (int b = 0; b<NDIM; ++b){
              bsq += Bcc[a](i,j)*Bcc[b](i,j) * adm_g_dd(a,b);
          }
      }
      bsq /= SQR(other[I_lorentz](i,j));

      Real ui_u[3];
      for (int a = 0; a<NDIM; ++a){
        ui_u[a] = prim[IVX+a](i,j) - other[I_lorentz](i,j) * z4c[Z4c::I_Z4c_betax+a](i,j) / z4c[Z4c::I_Z4c_alpha](i,j);
      }
      Real coord[3];
      coord[0]  = radius * sinth * cosph;
      coord[1]  = radius * sinth * sinph;
      coord[2]  = radius * costh;

      Real ur_u = 0.0;
      for (int a = 0; a<NDIM; ++a){
        ur_u += coord[a]*ui_u[a]/radius;
      }
      Real br_u = 0.0;
      for (int a = 0; a<NDIM; ++a){
        br_u += coord[a]*bi_u[a]/radius;
      }

      Real betasq = 0.0;
      for (int a = 0; a<NDIM; ++a){
        for (int b = 0; b<NDIM; ++b){
          betasq += z4c[Z4c::I_Z4c_betax+a](i,j) * z4c[Z4c::I_Z4c_betax+b](i,j) * adm_g_dd(a,b);
        }
      }

      Real b0_d = (-SQR(z4c[Z4c::I_Z4c_alpha](i,j)) + betasq) * b0_u;
      for (int a = 0; a<NDIM; ++a){
        for (int b = 0; b<NDIM; ++b){
          b0_d += z4c[Z4c::I_Z4c_betax+a](i,j) * bi_u[b] * adm_g_dd(a,b); 
        }
      }

      other[I_poynting](i,j) = bsq*ur_u*other[I_u_t](i,j) - br_u * b0_d;     // T^r_t = b^2 u^r u_t - b^r b_t
      delete pinterp3;
    } // phi loop
  } // theta loop
  
  for (int n=0; n<NHYDRO; ++n) {
    prim_[n].DeleteAthenaArray();
    cons_[n].DeleteAthenaArray();
  }
  T_.DeleteAthenaArray();
  for (int n=0; n<NSCALARS; ++n) {
    Y_[n].DeleteAthenaArray();
  }
  for (int n=0; n<NDIM; ++n) {
    flux_[n].DeleteAthenaArray();
    Bcc_[n].DeleteAthenaArray();
  }
  for (int n=0; n<Z4c::N_ADM; ++n) {
    vc_adm_[n].DeleteAthenaArray();
    adm_[n].DeleteAthenaArray();
  }
  for (int n=0; n<Z4c::N_Z4c; ++n) {
    vc_z4c_[n].DeleteAthenaArray();
    z4c_[n].DeleteAthenaArray();
  }
  adm_g_dd.DeleteTensorPointwise();
}

//----------------------------------------------------------------------------------------
// \!fn void Ejecta::Calculate(int iter, Real time)
// \brief Calculate ejecta quantities
void Ejecta::Calculate(int iter, Real time)
{
  if((time < start_time) || (time > stop_time)) return;
  if (iter % compute_every_iter != 0) return;

  for (int n=0; n<NHYDRO; ++n) {
    prim[n].ZeroClear();
    cons[n].ZeroClear();
  }
#if USETM
  for (int n=0; n<NSCALARS; ++n) {
    Y[n].ZeroClear();
  }
  T.ZeroClear();
#endif
  for (int n=0; n<3; ++n) {
    Bcc[n].ZeroClear();
  }
  for (int n=0; n<Z4c::N_ADM; ++n) {
    adm[n].ZeroClear();
  }
  for (int n=0; n<Z4c::N_Z4c; ++n) {
    z4c[n].ZeroClear();
  }
  for (int n=0; n<NOTHER; ++n) {
    other[n].ZeroClear();
  }
  mass_contained = 0.0;
  Mdot_total = 0.0;

  MeshBlock * pmb = pmesh->pblock;
  while (pmb != nullptr) {
    Interp(pmb);
    Mass(pmb);
    pmb = pmb->next;
  }

  SphericalIntegrals();


//#ifdef MPI_PARALLEL
//  MPI_Allreduce(MPI_IN_PLACE, rho.data(), ntheta*nphi, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
//#endif
#ifdef MPI_PARALLEL
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == root) {
    for (int n=0; n<NHYDRO; ++n) {
      MPI_Reduce(MPI_IN_PLACE, prim[n].data(), ntheta*nphi, MPI_ATHENA_REAL, MPI_SUM, root, MPI_COMM_WORLD);
      MPI_Reduce(MPI_IN_PLACE, cons[n].data(), ntheta*nphi, MPI_ATHENA_REAL, MPI_SUM, root, MPI_COMM_WORLD);
    }
#if USETM
    MPI_Reduce(MPI_IN_PLACE, T.data(), ntheta*nphi, MPI_ATHENA_REAL, MPI_SUM, root, MPI_COMM_WORLD);
    for (int n=0; n<NSCALARS; ++n) {
      MPI_Reduce(MPI_IN_PLACE, Y[n].data(), ntheta*nphi, MPI_ATHENA_REAL, MPI_SUM, root, MPI_COMM_WORLD);
    }
#endif
    for (int n=0; n<3; ++n) {
      MPI_Reduce(MPI_IN_PLACE, Bcc[n].data(), ntheta*nphi, MPI_ATHENA_REAL, MPI_SUM, root, MPI_COMM_WORLD);
    }

    for (int n=0; n<Z4c::N_ADM; ++n) {
      MPI_Reduce(MPI_IN_PLACE, adm[n].data(), ntheta*nphi, MPI_ATHENA_REAL, MPI_SUM, root, MPI_COMM_WORLD);
    }
    for (int n=0; n<Z4c::N_Z4c; ++n) {
      MPI_Reduce(MPI_IN_PLACE, z4c[n].data(), ntheta*nphi, MPI_ATHENA_REAL, MPI_SUM, root, MPI_COMM_WORLD);
    }
    for (int n=0; n<NOTHER; ++n) {
      MPI_Reduce(MPI_IN_PLACE, other[n].data(), ntheta*nphi, MPI_ATHENA_REAL, MPI_SUM, root, MPI_COMM_WORLD);
    }

    MPI_Reduce(MPI_IN_PLACE, integrals_unbound.data(),n_unbound*n_int , MPI_ATHENA_REAL, MPI_SUM, root, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, az_integrals_unbound.data(),n_unbound*n_int*ntheta , MPI_ATHENA_REAL, MPI_SUM, root, MPI_COMM_WORLD);
    for (int m=0; m < n_hist; ++m){
       MPI_Reduce(MPI_IN_PLACE, hist[m].data(),n_unbound*n_bins[m] , MPI_ATHENA_REAL, MPI_SUM, root, MPI_COMM_WORLD);
    }


    MPI_Reduce(MPI_IN_PLACE, &Mdot_total, 1, MPI_ATHENA_REAL, MPI_SUM, root, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &mass_contained, 1, MPI_ATHENA_REAL, MPI_SUM, root, MPI_COMM_WORLD);
  } else {
    for (int n=0; n<NHYDRO; ++n) {
      MPI_Reduce(prim[n].data(), prim[n].data(), ntheta*nphi, MPI_ATHENA_REAL, MPI_SUM, root, MPI_COMM_WORLD);
      MPI_Reduce(cons[n].data(), cons[n].data(), ntheta*nphi, MPI_ATHENA_REAL, MPI_SUM, root, MPI_COMM_WORLD);
    }
#if USETM
    MPI_Reduce(T.data(), T.data(), ntheta*nphi, MPI_ATHENA_REAL, MPI_SUM, root, MPI_COMM_WORLD);
    for (int n=0; n<NSCALARS; ++n) {
      MPI_Reduce(Y[n].data(), Y[n].data(), ntheta*nphi, MPI_ATHENA_REAL, MPI_SUM, root, MPI_COMM_WORLD);
    }
#endif
    for (int n=0; n<3; ++n) {
      MPI_Reduce(Bcc[n].data(), Bcc[n].data(), ntheta*nphi, MPI_ATHENA_REAL, MPI_SUM, root, MPI_COMM_WORLD);
    }

    for (int n=0; n<Z4c::N_ADM; ++n) {
      MPI_Reduce(adm[n].data(), adm[n].data(), ntheta*nphi, MPI_ATHENA_REAL, MPI_SUM, root, MPI_COMM_WORLD);
    }
    for (int n=0; n<Z4c::N_Z4c; ++n) {
      MPI_Reduce(z4c[n].data(), z4c[n].data(), ntheta*nphi, MPI_ATHENA_REAL, MPI_SUM, root, MPI_COMM_WORLD);
    }
    for (int n=0; n<NOTHER; ++n) {
      MPI_Reduce(other[n].data(), other[n].data(), ntheta*nphi, MPI_ATHENA_REAL, MPI_SUM, root, MPI_COMM_WORLD);
    }

    MPI_Reduce(integrals_unbound.data(), integrals_unbound.data(),n_unbound*n_int , MPI_ATHENA_REAL, MPI_SUM, root, MPI_COMM_WORLD);
    MPI_Reduce(az_integrals_unbound.data(), az_integrals_unbound.data(),n_unbound*n_int*ntheta , MPI_ATHENA_REAL, MPI_SUM, root, MPI_COMM_WORLD);
    for (int m=0; m < n_hist; ++m){
       MPI_Reduce(hist[m].data(), hist[m].data(),n_unbound*n_bins[m] , MPI_ATHENA_REAL, MPI_SUM, root, MPI_COMM_WORLD);
    }
    MPI_Reduce(&Mdot_total, &Mdot_total, 1, MPI_ATHENA_REAL, MPI_SUM, root, MPI_COMM_WORLD);
    MPI_Reduce(&mass_contained, &mass_contained, 1, MPI_ATHENA_REAL, MPI_SUM, root, MPI_COMM_WORLD);
  }
#endif
  Write(iter, time);
}

//----------------------------------------------------------------------------------------
// \!fn void Ejecta::Write(int iter, Real time)
// \brief Output ejecta quantities
// void Ejecta::Write_txt(int iter, Real time)
// {
//   if((time < start_time) || (time > stop_time)) return;
//   if (iter % compute_every_iter != 0) return;
//   MeshBlock * pmb = pmesh->pblock;

//   for (int i=0; i<ntheta; i++) {
//     Real theta = th_grid(i);
//     Real sinth = std::sin(theta);
//     Real costh = std::cos(theta);
//     for (int j=0; j<nphi; j++) {
//       Real phi   = ph_grid(j);
      
//       std::stringstream ss_i, ss_j;
//       ss_i << std::setw(3) << std::setfill('0') << i;
//       ss_j << std::setw(3) << std::setfill('0') << j;
//       std::string s_i = ss_i.str();
//       std::string s_j = ss_j.str();
//       std::ofstream outfile;
//       outfile.open("quantities_" + s_i + "_" + s_j + ".dat", std::ios_base::app);
//       Real tt = pmb->pmy_mesh->time;
//       outfile << tt << " " << theta << " " << phi << " "
//                     << rho(i,j) << " "
//                     << press(i,j) << " "
//                     << vx(i, j) << " " 
//                     << vy(i, j) << " "
//                     << vz(i, j) << " "
//                     << Bx(i, j) << " "
//                     << By(i, j) << " "
//                     << Bz(i, j) << " " << std::endl;
//       outfile.close();
//     }
//   }
// }

//---------------------------------------------------------------------------------------

void Ejecta::SphericalIntegrals()
{
  Real integrals[n_int];
  Real histvals[n_hist];
//  AthenaArray<int> count, countaz;
  AthenaArray<bool> unbound;
//  count.NewAthenaArray(n_unbound);
//  countaz.NewAthenaArray(n_unbound,ntheta);
  unbound.NewAthenaArray(n_unbound);
  unbound.Fill(false);
//  count.ZeroClear();
//  countaz.ZeroClear();

  integrals_unbound.ZeroClear();
  az_integrals_unbound.ZeroClear();
  for (int m = 0; m<n_hist; ++m ){
    hist[m].ZeroClear();
  }


  for (int i=0; i<ntheta; i++) {
    Real sinth = std::sin(theta(i));
    Real costh = std::cos(theta(i));
    for (int j=0; j<nphi; j++) {
      Real sinph = std::sin(phi(j));
      Real cosph = std::cos(phi(j));
	
      if(!havepoint(i,j)) continue; //continue if this process doesnt have this point

      if(!(other[I_bernoulli](i,j) > 0.0) && !(other[I_u_t](i,j) < -1.0)) continue; // continue if point bound
      
      Real x = radius * cosph * sinth;
      Real y = radius * sinph * sinth;
      Real z = radius * costh;
      Real vrad = x * prim[IVX](i,j) + y * prim[IVY](i,j) + z * prim[IVZ](i,j);
      Real weight = other[I_fD_r](i,j) * SQR(radius) * sinth * dth_grid() * dph_grid(); // radial mass flux * area elt
      
      // values for integrals over spheres
      integrals[I_int_mass] = weight; //mass flux      
      integrals[I_int_rho] = prim[IDN](i,j)*weight; //mass weighted density
      integrals[I_int_entr] = other[I_entropy](i,j)*weight;  //mass weighted entropy per baryon
      integrals[I_int_temp] = T(i,j)*weight; //mass weighted Temp
      integrals[I_int_ye]   = Y[0](i,j)*weight; //Mass weighted Ye
      integrals[I_int_vel]  = other[I_v_mag](i,j)*weight;  //mass weighted 3-velocity
      integrals[I_int_ber]  = other[I_bernoulli](i,j)*weight; //mass weighted bernoulli
      integrals[I_int_velinf]  = sqrt(2.0*(-other[I_u_t](i,j)-1.0))*weight; //mass weighted v_inf

      // values for histograms
      histvals[I_hist_entr] = other[I_entropy](i,j);
      histvals[I_hist_logrho] = std::log10(prim[IDN](i,j));
      histvals[I_hist_temp] = T(i,j);
      histvals[I_hist_ye] = Y[0](i,j);
      histvals[I_hist_vel] = other[I_v_mag](i,j);
      histvals[I_hist_ber] = other[I_bernoulli](i,j);
      histvals[I_hist_velinf] = sqrt(2.0*(-other[I_u_t](i,j)-1.0));
      histvals[I_hist_theta] = theta(i);

      //unboundedeness criteria
      unbound(I_unbound_bernoulli) = (other[I_bernoulli](i,j) > 0.0) ? true : false;
      unbound(I_unbound_bernoulli_outflow) = ((other[I_bernoulli](i,j) > 0.0) && (vrad > 0.0)) ? true : false ;
      unbound(I_unbound_geodesic) = (other[I_u_t](i,j) < -1.0) ? true : false;
      unbound(I_unbound_geodesic_outflow) = ((other[I_u_t](i,j) < -1.0) && (vrad > 0.0)) ? true : false;



      for(int n=0; n < n_unbound; ++n){ //unboundedness criteria
        if(!unbound(n)) continue;
//          ++count(n);
//       	  ++countaz(n,i);
        for (int m = 0; m<n_int; ++m ){ // integrated variables
          integrals_unbound(n,m) += integrals[m];
          az_integrals_unbound(n,m,i) += integrals[m];
	}
	for (int m = 0; m<n_hist; ++m ){ //loop over histogram variables 
	  if(histvals[m] < hist_grid[m](0)){ //if value is below lower limit of histogram add to lowest bin
            hist[m](n,0) += weight; 
	    continue;
	  }
          for(int l = 1; l < n_bins[m]; l++){ // loop over histogram bins for variable m
            if(histvals[m] < hist_grid[m](l)) {
              hist[m](n,l-1) += weight;
              break;
	    }
	    if(histvals[m] > hist_grid[m](n_bins[m]-1)){ //if value is above largest bin add to largest bin
	      hist[m](n,n_bins[m]-1) += weight; 
	    }
	  } //bin loop l
        } // variable loop m
      } //unboundedness loop n

    } //end phi loop
  } //end theta loop

//     count.DeleteAthenaArray();
//     countaz.DeleteAthenaArray();
     unbound.DeleteAthenaArray();
}




//----------------------------------------------------------------------------------------
// \!fn void Ejecta::Write(int iter, Real time)
// \brief Output ejecta quantities
void Ejecta::Write(int iter, Real time)
{
  if (ioproc) {

    std::stringstream ss_i;
    ss_i << std::setw(6) << std::setfill('0') << iter / compute_every_iter;
    std::string s_i = ss_i.str();
    std::string filename = "ejecta" + std::to_string(nr) + "_" + s_i + ".h5";
    hid_t file = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hid_t prim_id = H5Gcreate1(file, "prim", H5P_DEFAULT);
    hid_t cons_id = H5Gcreate1(file, "cons", H5P_DEFAULT);
    hid_t Bcc_id = H5Gcreate1(file, "Bcc", H5P_DEFAULT);
    hid_t adm_id = H5Gcreate1(file, "adm", H5P_DEFAULT);
    hid_t z4c_id = H5Gcreate1(file, "z4c", H5P_DEFAULT);
    hid_t other_id = H5Gcreate1(file, "other", H5P_DEFAULT);

    //Create the data space for the data set//
    hsize_t dim_1[1], dim_2[2];
    hid_t dataset, dataspace;

    // Time
    dim_1[0] = 1;
    dataspace = H5Screate_simple(1, dim_1, NULL);
    Real tvec[1] = {time};
    dataset = H5Dcreate(file, "time", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, tvec);
    H5Dclose(dataset);

    // R
    dim_1[0] = 1;
    dataspace = H5Screate_simple(1, dim_1, NULL);
    Real rvec[1] = {radius};
    dataset = H5Dcreate(file, "radius", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, rvec);
    H5Dclose(dataset);

    // Theta
    dim_1[0] = ntheta;
    dataspace = H5Screate_simple(1, dim_1, NULL);
    dataset = H5Dcreate(file, "theta", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, theta.data());
    H5Dclose(dataset);

    // Phi
    dim_1[0] = nphi;
    dataspace = H5Screate_simple(1, dim_1, NULL);
    dataset = H5Dcreate(file, "phi", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, phi.data());
    H5Dclose(dataset);

    // Primitive variables
    dim_2[0] = ntheta;
    dim_2[1] = nphi;
    dataspace = H5Screate_simple(2, dim_2, NULL);

    for (int n=0; n<NHYDRO; ++n) {
      dataset = H5Dcreate(file, ("prim/" + prim_names[n] + "/").c_str(), H5T_NATIVE_DOUBLE,
          dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, prim[n].data());
      H5Dclose(dataset);

      dataset = H5Dcreate(file, ("cons/" + cons_names[n] + "/").c_str(), H5T_NATIVE_DOUBLE,
          dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, cons[n].data());
      H5Dclose(dataset);
    }


#if USETM

    // T
    dataset = H5Dcreate(file, "prim/T", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, T.data());
    H5Dclose(dataset);

    // Y
    for (int n=0; n<NSCALARS; ++n) {
      dataset = H5Dcreate(file, ("prim/Y_" + std::to_string(n) + "/").c_str(), H5T_NATIVE_DOUBLE,
          dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, Y[n].data());
      H5Dclose(dataset);
    }
#endif

    for (int n=0; n<3; ++n) {
      dataset = H5Dcreate(file, ("Bcc/B_" + std::to_string(n+1) + "/").c_str(), H5T_NATIVE_DOUBLE,
          dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, Bcc[n].data());
      H5Dclose(dataset);
    }

    for (int n=0; n<Z4c::N_ADM; ++n) {
      dataset = H5Dcreate(file, ("adm/" + adm_names[n] + "/").c_str(), H5T_NATIVE_DOUBLE,
          dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, adm[n].data());
      H5Dclose(dataset);
    }
    for (int n=0; n<Z4c::N_Z4c; ++n) {
      dataset = H5Dcreate(file, ("z4c/" + z4c_names[n] + "/").c_str(), H5T_NATIVE_DOUBLE,
          dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, z4c[n].data());
      H5Dclose(dataset);
    }

    for (int n=0; n<NOTHER; ++n) {
      dataset = H5Dcreate(file, ("other/" + other_names[n] + "/").c_str(), H5T_NATIVE_DOUBLE,
          dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, other[n].data());
      H5Dclose(dataset);
    }

    // M
    dim_1[0] = 1;
    dataspace = H5Screate_simple(1, dim_1, NULL);
    Real Mvec[1] = {mass_contained};
    dataset = H5Dcreate(file, "mass", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, Mvec);
    H5Dclose(dataset);

    // Mdot
    dim_1[0] = 1;
    dataspace = H5Screate_simple(1, dim_1, NULL);
    Real Mdot_vec[1] = {Mdot_total};
    dataset = H5Dcreate(file, "Mdot_total", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, Mdot_vec);
    H5Dclose(dataset);

    H5Sclose(dataspace);
    H5Gclose(prim_id);
    H5Gclose(cons_id);
    H5Gclose(Bcc_id);
    H5Gclose(adm_id);
    H5Gclose(z4c_id);
    H5Gclose(other_id);
    H5Fclose(file);


    // Summary file
    for(int n=0; n<n_unbound; ++n){
      fprintf(pofile_unbound[n], "%d %g ", iter, time);
      fprintf(pofile_unbound[n], "%.15e ", integrals_unbound(n,I_int_mass));
      for(int m=I_int_entr; m < n_int; m++){
        fprintf(pofile_unbound[n], "%.15e ", integrals_unbound(n,m)/integrals_unbound(n,I_int_mass));//normalise by average mass
      }
      fprintf(pofile_unbound[n],"\n");
      fflush(pofile_unbound[n]);
    }
    for(int n=0; n<n_unbound; ++n){
      fprintf(pofile_az_unbound[n], "### Time = %g \n", time);
      for(int i = 0; i < ntheta; ++i){
        fprintf(pofile_az_unbound[n], "%.15e ",theta(i));
        fprintf(pofile_az_unbound[n], "%.15e ", az_integrals_unbound(n,I_int_mass,i));
        for(int m=I_int_entr; m < n_int; m++){
          fprintf(pofile_az_unbound[n], "%.15e ", az_integrals_unbound(n,m,i)/az_integrals_unbound(n,I_int_mass,i)); //normalise by mass
        }
        fprintf(pofile_az_unbound[n],"\n");
      }
      fprintf(pofile_az_unbound[n],"\n");
      fflush(pofile_az_unbound[n]);
    }
    for(int n=0; n<n_unbound; ++n){
      for (int m=0; m<n_hist; ++m){
        fprintf(pofile_hist_unbound[n][m], "### Time = %g \n", time);
        for(int l = 0; l < n_bins[m]; ++l){
            fprintf(pofile_hist_unbound[n][m], "%.15e %.15e \n",hist_grid[m](l)+delta_hist[m]/2.0,hist[m](n,l)); //bin value is centre of bin
	}
	fprintf(pofile_hist_unbound[n][m], "\n");
        fflush(pofile_hist_unbound[n][m]);
      }
    }

  }
}


//----------------------------------------------------------------------------------------
// \!fn int AHF::tpindex(const int i, const int j)
// \brief spherical grid single index (i,j) -> index
int Ejecta::tpindex(const int i, const int j)
{
  return i*nphi + j; 
}

//----------------------------------------------------------------------------------------
// \!fn Real AHF::th_grid(const int i)
// \brief theta coordinate from index
Real Ejecta::th_grid(const int i)
{
  Real dtheta = dth_grid();
  return dtheta*(0.5 + i);
}

//----------------------------------------------------------------------------------------
// \!fn Real AHF::ph_grid(const int i)
// \brief phi coordinate from index
Real Ejecta::ph_grid(const int j)
{
  Real dphi = dph_grid();
  return dphi*(0.5 + j);
}

//----------------------------------------------------------------------------------------
// \!fn Real AHF::dth_grid()
// \brief compute spacing dtheta 
Real Ejecta::dth_grid()
{
  return PI/ntheta;
}

//----------------------------------------------------------------------------------------
// \!fn Real AHF::dph_grid()
// \brief compute spacing dphi

Real Ejecta::dph_grid()
{
  return 2.0*PI/nphi;
}

//----------------------------------------------------------------------------------------
// \!fn Real AHF::SpatialDet(Real gxx, ... , Real gzz)
// \brief returns determinant of 3-metric
// Taken from class Z4c 

Real Ejecta::SpatialDet(Real const gxx, Real const gxy, Real const gxz,
                     Real const gyy, Real const gyz, Real const gzz)
{
  return - SQR(gxz)*gyy + 2*gxy*gxz*gyz - gxx*SQR(gyz) - SQR(gxy)*gzz + gxx*gyy*gzz;
}

//----------------------------------------------------------------------------------------
// \!fn void AHF::SpatialInv(Real const detginv,
//           Real const gxx, Real const gxy, Real const gxz,
//           Real const gyy, Real const gyz, Real const gzz,
//           Real * uxx, Real * uxy, Real * uxz,
//           Real * uyy, Real * uyz, Real * uzz)
// \brief returns inverse of 3-metric
// Taken from class Z4c

void Ejecta::SpatialInv(Real const detginv,
                     Real const gxx, Real const gxy, Real const gxz,
                     Real const gyy, Real const gyz, Real const gzz,
                     Real * uxx, Real * uxy, Real * uxz,
                     Real * uyy, Real * uyz, Real * uzz)
{
  *uxx = (-SQR(gyz) + gyy*gzz)*detginv;
  *uxy = (gxz*gyz  - gxy*gzz)*detginv;
  *uyy = (-SQR(gxz) + gxx*gzz)*detginv;
  *uxz = (-gxz*gyy + gxy*gyz)*detginv;
  *uyz = (gxy*gxz  - gxx*gyz)*detginv;
  *uzz = (-SQR(gxy) + gxx*gyy)*detginv;
  return;
}

//----------------------------------------------------------------------------------------
// \!fn Real AHF::Trace(Real detginv, Real gxx, ... , Real gzz, Real Axx, ..., Real Azz)
// \brief returns Trace of extrinsic curvature
// Taken from class Z4c

Real Ejecta::Trace(Real const detginv,
                Real const gxx, Real const gxy, Real const gxz,
                Real const gyy, Real const gyz, Real const gzz,
                Real const Axx, Real const Axy, Real const Axz,
                Real const Ayy, Real const Ayz, Real const Azz)
{
  return (detginv*(
       - 2.*Ayz*gxx*gyz + Axx*gyy*gzz +  gxx*(Azz*gyy + Ayy*gzz)
       + 2.*(gxz*(Ayz*gxy - Axz*gyy + Axy*gyz) + gxy*(Axz*gyz - Axy*gzz))
       - Azz*SQR(gxy) - Ayy*SQR(gxz) - Axx*SQR(gyz)
       ));
}

Real Ejecta::MassLossRate(Real const fx, Real const fy, Real const fz,
                          Real const sinth, Real const costh, Real const sinph, Real const cosph)
{
  Real r_x = cosph * sinth;
  Real r_y = sinph * sinth;
  Real r_z = costh;
  return (fx*r_x + fy*r_y + fz*r_z) * SQR(radius) * sinth * dth_grid() * dph_grid();
}

Real Ejecta::MassLossRate2(Real const D, Real const ux, Real const uy, Real const uz, Real const W,
                           Real const alpha, Real const betax, Real const betay, Real const betaz,
                           Real const sinth, Real const costh, Real const sinph, Real const cosph)
{
  Real r_x = cosph * sinth;
  Real r_y = sinph * sinth;
  Real r_z = costh;
  Real v_x = alpha*ux/W - betax;
  Real v_y = alpha*uy/W - betay;
  Real v_z = alpha*uz/W - betaz;
  return D * (v_x*r_x + v_y*r_y + v_z*r_z) * SQR(radius) * sinth * dth_grid() * dph_grid();
}

