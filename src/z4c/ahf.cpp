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
#include <unistd.h>
#include <cmath> // NAN

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#define DEBUG_OUTPUT 0

#include "ahf.hpp"
#include "../globals.hpp"
#include "../parameter_input.hpp"
#include "../mesh/mesh.hpp"
#include "../utils/linear_algebra.hpp"
#include "../utils/tensor.hpp"
#include "../coordinates/coordinates.hpp"
#include "puncture_tracker.hpp"

using namespace utils::tensor;

//----------------------------------------------------------------------------------------
//! \fn AHF::AHF(Mesh * pmesh, ParameterInput * pin, int n)
//  \brief class for apparent horizon finder
AHF::AHF(Mesh * pmesh, ParameterInput * pin, int n):
  pmesh(pmesh) 
{
  nhorizon = pin->GetOrAddInteger("ahf", "num_horizons",1);
  
  nh = n;
  std::string parname;
  std::string n_str = std::to_string(nh);
  
  ntheta = pin->GetOrAddInteger("ahf", "ntheta",60);

  nphi = pin->GetOrAddInteger("ahf", "nphi",30);
  if ((nphi+1)%2==0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in AHF setup" << std::endl
        << "nphi must be even " << nphi << std::endl;
    ATHENA_ERROR(msg);
  }

  lmax = pin->GetOrAddInteger("ahf", "lmax",10);
  lmax1 = lmax+1;
  
  flow_iterations = pin->GetOrAddInteger("ahf", "flow_iterations",100);
  flow_alpha_beta_const = pin->GetOrAddReal("ahf", "flow_alpha_beta_const",1.0);
  hmean_tol = pin->GetOrAddReal("ahf", "hmean_tol",100.);
  mass_tol = pin->GetOrAddReal("ahf", "mass_tol",1e-2);
  verbose = pin->GetOrAddBoolean("ahf", "verbose", 0);
  root = pin->GetOrAddInteger("ahf", "mpi_root", 0);
  merger_distance = pin->GetOrAddReal("ahf", "merger_distance", 0.1);
  bitant = pin->GetOrAddBoolean("mesh", "bitant", false);
  
  // Initial guess
  parname = "initial_radius_";
  parname += n_str;
  initial_radius = pin->GetOrAddReal("ahf", parname, 1.0);

  expand_guess = pin->GetOrAddReal("ahf", "expand_guess",1.1);
  npunct = pin->GetOrAddInteger("z4c", "npunct", 0);

  // Center
  parname = "center_x_";
  parname += n_str;
  center[0] = pin->GetOrAddReal("ahf", parname, 0.0);
  parname = "center_y_";
  parname += n_str;
  center[1] = pin->GetOrAddReal("ahf", parname, 0.0);
  parname = "center_z_";
  parname += n_str;
  center[2] = pin->GetOrAddReal("ahf", parname, 0.0);

  parname = "use_puncture_";
  parname += n_str;
  use_puncture = pin->GetOrAddInteger("ahf", parname, -1);

  parname = "compute_every_iter_";
  parname += n_str;
  compute_every_iter = pin->GetOrAddInteger("ahf", parname, 1);

  if (use_puncture>=0) {
    // Center is determined on the fly during the initial guess
    // to follow the chosen puncture 
    if (use_puncture >= npunct) {
        std::stringstream msg;
        msg << "### FATAL ERROR in AHF constructor" << std::endl;
        msg << " : punc = " << use_puncture << " > npunct = " << npunct;
        throw std::runtime_error(msg.str().c_str());
    }
  }
  parname = "use_puncture_massweighted_center_";
  parname += n_str;
  use_puncture_massweighted_center = pin->GetOrAddBoolean("ahf", parname, 0);

  parname = "start_time_";
  parname += n_str;
  start_time = pin->GetOrAddReal("ahf", parname, std::numeric_limits<double>::max());

  parname = "stop_time_";
  parname += n_str;
  stop_time = pin->GetOrAddReal("ahf", parname, -1.0);

  parname = "wait_until_punc_are_close_";
  parname += n_str;
  wait_until_punc_are_close = pin->GetOrAddBoolean("ahf", parname, 0);
  
  // Initialize last & found
  last_a0 = -1;
  ah_found = false;
  
  //TODO guess from file
  // * if found, set ah_found to true & store the guess
  
  // Points for sph harm l>=1
  lmpoints = lmax1*lmax1;

  // Coefficients
  a0.NewAthenaArray(lmax1);
  ac.NewAthenaArray(lmpoints);
  as.NewAthenaArray(lmpoints);
  
  // Spherical harmonics
  // the spherical grid is the same for all surfaces
  Y0.NewAthenaArray(ntheta,nphi,lmax1);
  Yc.NewAthenaArray(ntheta,nphi,lmpoints);
  Ys.NewAthenaArray(ntheta,nphi,lmpoints);

  dY0dth.NewAthenaArray(ntheta,nphi,lmax1);
  dYcdth.NewAthenaArray(ntheta,nphi,lmpoints);
  dYsdth.NewAthenaArray(ntheta,nphi,lmpoints);
  dYcdph.NewAthenaArray(ntheta,nphi,lmpoints);
  dYsdph.NewAthenaArray(ntheta,nphi,lmpoints);

  dY0dth2.NewAthenaArray(ntheta,nphi,lmax1);
  dYcdth2.NewAthenaArray(ntheta,nphi,lmpoints); 
  dYcdthdph.NewAthenaArray(ntheta,nphi,lmpoints);
  dYsdth2.NewAthenaArray(ntheta,nphi,lmpoints);
  dYsdthdph.NewAthenaArray(ntheta,nphi,lmpoints);
  dYcdph2.NewAthenaArray(ntheta,nphi,lmpoints);
  dYsdph2.NewAthenaArray(ntheta,nphi,lmpoints);
  
  ComputeSphericalHarmonics();
  
  // Fields on the sphere 
  rr.NewAthenaArray(ntheta,nphi);
  g.NewAthenaTensor(ntheta,nphi);
  dg.NewAthenaTensor(ntheta,nphi);
  K.NewAthenaTensor(ntheta,nphi);
  
  // Array computed in surface integrals
  rho.NewAthenaArray(ntheta,nphi);

  // Flag points existing on this mesh
  havepoint.NewAthenaArray(ntheta,nphi);

  // Initialize horizon properties to NAN
  for (int v=0; v<hnvar; ++v) {
    ah_prop[v] = NAN;
  }
  
  // Prepare output
  parname = "horizon_file_summary_";
  parname += n_str;
  ofname_summary = pin->GetString("job", "problem_id") + ".";
  ofname_summary += pin->GetOrAddString("ahf", parname, "horizon_summary_"+n_str);
  ofname_summary += ".txt";

  parname = "horizon_file_shape_";
  parname += n_str;
  ofname_shape = pin->GetString("job", "problem_id") + ".";
  ofname_shape += pin->GetOrAddString("ahf", parname, "horizon_shape_"+n_str);
  ofname_shape += ".txt";

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
      fprintf(pofile_summary, "# 1:iter 2:time 3:mass 4:Sx 5:Sy 6:Sz 7:S 8:area 9:hrms 10:hmean 11:meanradius\n");
      fflush(pofile_summary);
    }
    //TODO Output Ylm, grid and center?
  }
}

AHF::~AHF() {
  // Coefficients
  a0.DeleteAthenaArray();
  ac.DeleteAthenaArray();
  as.DeleteAthenaArray();
  
  // Spherical harmonics
  Y0.DeleteAthenaArray();
  Yc.DeleteAthenaArray();
  Ys.DeleteAthenaArray();
  
  dY0dth.DeleteAthenaArray();
  dYcdth.DeleteAthenaArray();
  dYcdph.DeleteAthenaArray();
  dYsdth.DeleteAthenaArray();
  dYsdph.DeleteAthenaArray();

  dY0dth2.DeleteAthenaArray();
  dYcdth2.DeleteAthenaArray(); 
  dYcdthdph.DeleteAthenaArray();
  dYcdph2.DeleteAthenaArray();
  dYsdth2.DeleteAthenaArray();
  dYsdthdph.DeleteAthenaArray();
  dYsdph2.DeleteAthenaArray();
  

  // Fields on the sphere
  rr.DeleteAthenaArray();
  g.DeleteAthenaTensor();
  dg.DeleteAthenaTensor();
  K.DeleteAthenaTensor();

  // Array computed in surface integrals
  rho.DeleteAthenaArray();
  
  // Flag points existing on this mesh
  havepoint.DeleteAthenaArray();

  // Close files
  if (ioproc) {
    fclose(pofile_summary);
  }
}

//----------------------------------------------------------------------------------------
// \!fn void AHF::Write(int iter, Real time)
// \brief output summary and shape file, for each horizon 
void AHF::Write(int iter, Real time)
{
  if (ioproc) {
    std::string i_str = std::to_string(iter);     
    if((time < start_time) || (time > stop_time)) return;
    if (wait_until_punc_are_close && !(PuncAreClose())) return;
    if (iter % compute_every_iter != 0) return;
    
    // Summary file
    fprintf(pofile_summary, "%d %g ", iter, time);  
    fprintf(pofile_summary, "%.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e",
        ah_prop[hmass],
        ah_prop[hSx],
        ah_prop[hSy],
        ah_prop[hSz],
        ah_prop[hS],
        ah_prop[harea],
        ah_prop[hhrms],
        ah_prop[hhmean],
        ah_prop[hmeanradius]);
    fprintf(pofile_summary, "\n");
    fflush(pofile_summary);
    if (ah_found) {
      // Shape file
      pofile_shape = fopen(ofname_shape.c_str(), "a");
      if (NULL == pofile_shape) {
        std::stringstream msg;
        msg << "### FATAL ERROR in AHF constructor" << std::endl;
        msg << "Could not open file '" << pofile_shape << "' for writing!";
        throw std::runtime_error(msg.str().c_str());
      }
      fprintf(pofile_shape, "# iter = %d, Time = %g\n",iter,time);
      for(int l=0; l<=lmax; l++){
        fprintf(pofile_shape,"%e ", a0(l)); 
        for(int m=0; m<=l; m++){
          int l1 = lmindex(l,m);
          fprintf(pofile_shape,"%e ",ac(l1));
          fprintf(pofile_shape,"%e ",as(l1));
        }
      }
      fprintf(pofile_shape,"\n");
      fclose(pofile_shape);
    }
  }
}

//----------------------------------------------------------------------------------------
// \!fn void AHF::MetricDerivatives(MeshBlock * pmb)
// \brief compute drvts of ADM metric at MB level
// Use 2nd order FD for simplicity and avoid using ghosts
// This assumes there is a special storage for the ADM drvts at the MB level:
//   pmb->pz4c->aux_g_ddd
void AHF::MetricDerivatives(MeshBlock * pmy_block) 
{
  Z4c *pz4c = pmy_block->pz4c;  // also needed for LOOP macros etc.

  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> adm_g_dd; // 3-metric  (NDIM=3 in z4c.hpp)
  adm_g_dd.InitWithShallowSlice(pmy_block->pz4c->storage.adm, Z4c::I_ADM_gxx);
  pz4c->aux_g_ddd.ZeroClear();

#if DEBUG_OUTPUT
  FILE *fp, *fpd;
  std::string fname ="metricderiv_gxx_";
  std::string fnamed="metricderiv_dgxxdx_";
  std::stringstream ss;
  ss << pmy_block;
  fname += ss.str()+".txt";
  fnamed+= ss.str()+".txt";
  fp = fopen(fname.c_str(),"w");
  fpd= fopen(fnamed.c_str(),"w");
#endif
  
  GLOOP2(k,j){
    Real oofdz = 1.0 / pz4c->mbi.dx3(k);
    Real oofdy = 1.0 / pz4c->mbi.dx2(j);
    for(int a = 0; a < NDIM; ++a){
      for(int b = 0; b < NDIM; ++b){
        GLOOP1(i){
          Real oofdx = 1.0 / pz4c->mbi.dx1(i);
          
          if (i==IX_IL-GSIZEI){
            pz4c->aux_g_ddd(0,a,b,k,j,i) =     oofdx * ( adm_g_dd(a,b,k,j,i+1) - adm_g_dd(a,b,k,j, i ) );
          } else if (i==IX_IU+GSIZEI){
            pz4c->aux_g_ddd(0,a,b,k,j,i) =     oofdx * ( adm_g_dd(a,b,k,j, i ) - adm_g_dd(a,b,k,j,i-1) );
          } else {
            pz4c->aux_g_ddd(0,a,b,k,j,i) = 0.5*oofdx * ( adm_g_dd(a,b,k,j,i+1) - adm_g_dd(a,b,k,j,i-1) );
          }
          
          if (j==IX_JL-GSIZEJ){
            pz4c->aux_g_ddd(1,a,b,k,j,i) =     oofdy * ( adm_g_dd(a,b,k,j+1,i) - adm_g_dd(a,b,k, j ,i) );
          } else if (j==IX_JU+GSIZEJ){
            pz4c->aux_g_ddd(1,a,b,k,j,i) =     oofdy * ( adm_g_dd(a,b,k, j ,i) - adm_g_dd(a,b,k,j-1,i) );
          } else {
            pz4c->aux_g_ddd(1,a,b,k,j,i) = 0.5*oofdy * ( adm_g_dd(a,b,k,j+1,i) - adm_g_dd(a,b,k,j-1,i) );
          }
          
          if (k==IX_KL-GSIZEK){
            pz4c->aux_g_ddd(2,a,b,k,j,i) =     oofdz * ( adm_g_dd(a,b,k+1,j,i) - adm_g_dd(a,b, k ,j,i) );
          } else if (k==IX_KU+GSIZEK){
            pz4c->aux_g_ddd(2,a,b,k,j,i) =     oofdz * ( adm_g_dd(a,b, k ,j,i) - adm_g_dd(a,b,k-1,j,i) );
          } else {
            pz4c->aux_g_ddd(2,a,b,k,j,i) = 0.5*oofdz * ( adm_g_dd(a,b,k+1,j,i) - adm_g_dd(a,b,k-1,j,i) );
          }

#if DEBUG_OUTPUT
          if (a==0 && b==0){
            fprintf(fp, "%23.15e %23.15e %23.15e %23.15e\n",pz4c->mbi.x1(i),pz4c->mbi.x2(j), 
                pz4c->mbi.x3(k), adm_g_dd(0,0,k,j,i));
            fprintf(fpd, "%23.15e %23.15e %23.15e %23.15e\n",pz4c->mbi.x1(i),pz4c->mbi.x2(j), 
                pz4c->mbi.x3(k), pz4c->aux_g_ddd(0,0,0,k,j,i));
          }
#endif

        } 
      }
    }
  }
  
  adm_g_dd.DeleteAthenaTensor();
#if DEBUG_OUTPUT
  fclose(fp);
  fclose(fpd);
#endif
  
}

//----------------------------------------------------------------------------------------
// \!fn void AHF::MetricInterp(MeshBlock * pmb)
// \brief interpolate metric on the surface n
// Flag here the surface points contained in the MB
void AHF::MetricInterp(MeshBlock * pmb)
{
  Z4c *pz4c = pmb->pz4c;

  LagrangeInterpND<metric_interp_order, 3> * pinterp3 = nullptr;
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> adm_g_dd;      // 3-metric  (NDIM=3 in z4c.hpp)
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> adm_K_dd;      // extr.curv.
  adm_g_dd.InitWithShallowSlice(pz4c->storage.adm, Z4c::I_ADM_gxx);
  adm_K_dd.InitWithShallowSlice(pz4c->storage.adm, Z4c::I_ADM_Kxx);

  // For interp
  Real origin[NDIM];
  Real delta[NDIM];
  int size[NDIM];
  Real coord[NDIM];
  
  // Center of the surface
  const Real xc = center[0];
  const Real yc = center[1];
  const Real zc = center[2];

#if DEBUG_OUTPUT
  FILE *fp, *fp_d;
  std::string fname   = "metricinterp_gxx_iter"+std::to_string(fastflow_iter)+"_";
  std::string fname_d = "metricinterp_dgxxdx_iter"+std::to_string(fastflow_iter)+"_";
  std::stringstream ss;
  ss << pmb;
  fname   += ss.str();
  fname_d += ss.str();
#ifdef MPI_PARALLEL
  int rank_tmp;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_tmp);
  fname   += "_rank"+std::to_string(rank_tmp);
  fname_d += "_rank"+std::to_string(rank_tmp);
#endif
  fname   += ".txt";
  fname_d += ".txt";
  fp   = fopen(fname.c_str(), "w");
  fp_d = fopen(fname_d.c_str(), "w");
#endif
  
  for(int i=0; i<ntheta; i++){
    
    Real theta = th_grid(i);
    Real sinth = std::sin(theta);
    Real costh = std::cos(theta);
    
    for(int j=0; j<nphi; j++){
      
      Real phi   = ph_grid(j);
      Real sinph = std::sin(phi);
      Real cosph = std::cos(phi);
      
      // Global coordinates of the surface
      Real x = xc + rr(i,j) * sinth * cosph;
      Real y = yc + rr(i,j) * sinth * sinph;
      Real z = zc + rr(i,j) * costh;

      // Impose bitant symmetry below
      bool bitant_sym = ( bitant && z < 0 ) ? true : false;
      // Associate z -> -z if bitant
      if (bitant) z = std::abs(z);


      if (!pmb->PointContained(x,y,z)) continue;

      // this surface point is in this MB
      havepoint(i,j) += 1;

      // Interpolate
      origin[0] = pz4c->mbi.x1(0);
      size[0]   = pz4c->mbi.nn1;
      delta[0]  = pz4c->mbi.dx1(0);
      coord[0]  = x;

      origin[1] = pz4c->mbi.x2(0);
      size[1]   = pz4c->mbi.nn2;
      delta[1]  = pz4c->mbi.dx2(0);
      coord[1]  = y;

      origin[2] = pz4c->mbi.x3(0);
      size[2]   = pz4c->mbi.nn3;
      delta[2]  = pz4c->mbi.dx3(0);
      coord[2]  = z;

      pinterp3 =  new LagrangeInterpND<metric_interp_order, 3>(origin, delta, size, coord);

      // With bitant wrt z=0, pick a (-) sign every time a z component is 
      // encountered.
      for(int a = 0; a < NDIM; ++a)
      for(int b = a; b < NDIM; ++b) {
        int bitant_z_fac = 1;
        if (bitant_sym) {
          if (a == 2) bitant_z_fac *= -1;
          if (b == 2) bitant_z_fac *= -1;
        }
        g(a,b,i,j) = pinterp3->eval(&(adm_g_dd(a,b,0,0,0)))*bitant_z_fac;
        K(a,b,i,j) = pinterp3->eval(&(adm_K_dd(a,b,0,0,0)))*bitant_z_fac;
        for(int c = 0; c < NDIM; ++c) {
          if (bitant_sym)  {
            if (c == 2) bitant_z_fac *= -1;
          }
          dg(c,a,b,i,j) = pinterp3->eval(&(pz4c->aux_g_ddd(c,a,b,0,0,0)))*bitant_z_fac;
        }
      }

      
      delete pinterp3;

#if DEBUG_OUTPUT
      fprintf(fp,   "%23.15e %23.15e %23.15e\n", theta, phi,  g(0,0,i,j));
      fprintf(fp_d, "%23.15e %23.15e %23.15e\n", theta, phi, dg(0,0,0,i,j));
#endif

    } // phi loop
  } // theta loop
  
  adm_g_dd.DeleteAthenaTensor();
  adm_K_dd.DeleteAthenaTensor();
  
#if DEBUG_OUTPUT
  fclose(fp);
  fclose(fp_d);
#endif  

}
//----------------------------------------------------------------------------------------
//! \fn void AHF::SurfaceIntegrals(const int n)
//  \brief compute expansion, surface element and spin integrand on surface n
// Needs metric and extr. curv. interpolated on the surface
// Performs local sums and MPI reduce
void AHF::SurfaceIntegrals()
{
  using namespace LinearAlgebra;

  const Real dtheta = dth_grid(); 
  const Real dphi   = dph_grid();
  const Real dthdph = dtheta * dphi; 
  const int nphihalf = (int)(nphi/2);
  const Real min_rp = 1e-10; 

  // Derivatives of (r,theta,phi) w.r.t (x,y,z)
  TensorPointwise<Real, Symmetries::NONE, NDIM, 1> drdi;
  TensorPointwise<Real, Symmetries::NONE, NDIM, 1> dthetadi;
  TensorPointwise<Real, Symmetries::NONE, NDIM, 1> dphidi; 

  TensorPointwise<Real, Symmetries::SYM2, NDIM, 2> drdidj;
  TensorPointwise<Real, Symmetries::SYM2, NDIM, 2> dthetadidj;
  TensorPointwise<Real, Symmetries::SYM2, NDIM, 2> dphididj;

  // Derivatives of F
  TensorPointwise<Real, Symmetries::NONE, NDIM, 1> dFdi;
  TensorPointwise<Real, Symmetries::NONE, NDIM, 1> dFdi_u; // upper index
  TensorPointwise<Real, Symmetries::SYM2, NDIM, 2> dFdidj;

  // Inverse metric
  TensorPointwise<Real, Symmetries::SYM2, NDIM, 2> ginv;

  // Normal
  TensorPointwise<Real, Symmetries::NONE, NDIM, 1> R;

  // dx^adth , dx^a/dph
  TensorPointwise<Real, Symmetries::NONE, NDIM, 1> dXdth;
  TensorPointwise<Real, Symmetries::NONE, NDIM, 1> dXdph;
  
  // Flat-space coordinate rotational KV
  TensorPointwise<Real, Symmetries::NONE, NDIM, 1> phix;
  TensorPointwise<Real, Symmetries::NONE, NDIM, 1> phiy;
  TensorPointwise<Real, Symmetries::NONE, NDIM, 1> phiz;

  TensorPointwise<Real, Symmetries::SYM2, NDIM, 2> nnF;
  
  // Initialize integrals
  for(int v=0; v<invar; v++){
    integrals[v] = 0.0;
  }

  drdi.NewTensorPointwise();
  dthetadi.NewTensorPointwise();
  dphidi.NewTensorPointwise();

  drdidj.NewTensorPointwise();
  dthetadidj.NewTensorPointwise();
  dphididj.NewTensorPointwise();
  
  dFdi.NewTensorPointwise();
  dFdi_u.NewTensorPointwise();
  dFdidj.NewTensorPointwise();
  
  ginv.NewTensorPointwise();
  
  R.NewTensorPointwise();

  dXdth.NewTensorPointwise();
  dXdph.NewTensorPointwise();
  
  phix.NewTensorPointwise();
  phiy.NewTensorPointwise();
  phiz.NewTensorPointwise();
  
  nnF.NewTensorPointwise();

  rho.ZeroClear();
  
#if DEBUG_OUTPUT
  FILE *fp_dFdxdx, *fp_nnFxx;
  std::string fname_dFdxdx="dFdxdx_iter"+std::to_string(fastflow_iter);
  std::string fname_nnFxx ="nnFxx_iter"+std::to_string(fastflow_iter);
#ifdef MPI_PARALLEL
  int rank_tmp;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_tmp);
  fname_dFdxdx += "_rank"+std::to_string(rank_tmp);
  fname_nnFxx  += "_rank"+std::to_string(rank_tmp);
#endif 
  fname_dFdxdx += ".txt";
  fname_nnFxx  += ".txt";
  fp_dFdxdx = fopen(fname_dFdxdx.c_str(),"w");
  fp_nnFxx  = fopen(fname_nnFxx.c_str(),"w");
#endif
  
  // Loop over surface points
  for(int i=0; i<ntheta; i++){
    
    Real const theta = th_grid(i);
    Real const sinth = std::sin(theta);
    Real const costh = std::cos(theta);
    
    for(int j=0; j<nphi; j++){
      
      if (!havepoint(i,j)) continue;

      Real const phi = ph_grid(j);
      Real const sinph = std::sin(phi);
      Real const cosph = std::cos(phi);

      // Calculate the expansion
      // -----------------------

      // Determinant of 3-metric
      Real detg = Det3Metric(
        g(0,0,i,j), g(0,1,i,j), g(0,2,i,j),
        g(1,1,i,j), g(1,2,i,j), g(2,2,i,j)
      );

      // Inverse metric
      Inv3Metric(
        1.0/detg,
		    g(0,0,i,j), g(0,1,i,j),  g(0,2,i,j),
		    g(1,1,i,j), g(1,2,i,j),  g(2,2,i,j),
		    &ginv(0,0), &ginv(0,1),  &ginv(0,2),
		    &ginv(1,1), &ginv(1,2) , &ginv(2,2)
      );

      // Trace of K
      Real TrK = TraceRank2(1.0/detg,
		       g(0,0,i,j), g(0,1,i,j), g(0,2,i,j),
		       g(1,1,i,j), g(1,2,i,j), g(2,2,i,j),
		       K(0,0,i,j), K(0,1,i,j), K(0,2,i,j),
		       K(1,1,i,j), K(1,2,i,j), K(2,2,i,j));

      // Local coordinates of the surface (re-used below)
      Real const xp = rr(i,j) * sinth * cosph;
      Real const yp = rr(i,j) * sinth * sinph;
      Real const zp = rr(i,j) * costh;
      
      Real const rp   = std::sqrt(xp*xp + yp*yp + zp*zp);
      Real const rhop = std::sqrt(xp*xp + yp*yp);
        
      if (rp < min_rp) {
        std::stringstream msg;
        msg << "### FATAL ERROR in AHF" << std::endl
        << "surface radius cannot be zero (rp=" << rp
        << " < min_rp" << min_rp << ")"  << std::endl;
        ATHENA_ERROR(msg);
      }

      Real const _divrp = 1.0/rp;
      Real const _divrp3 = SQR(_divrp)*_divrp;
      Real const _divrhop = 1.0/rhop;
      
      // First derivatives of (r,theta,phi) with respect to (x,y,z)
      drdi(0) = xp*_divrp;
      drdi(1) = yp*_divrp;
      drdi(2) = zp*_divrp;
      
      dthetadi(0) = zp*xp*(SQR(_divrp)*_divrhop);
      dthetadi(1) = zp*yp*(SQR(_divrp)*_divrhop);
      dthetadi(2) = -rhop*SQR(_divrp);
      
      dphidi(0) = -yp*SQR(_divrhop);
      dphidi(1) = xp*SQR(_divrhop);
      dphidi(2) = 0.0;
      
      // Second derivatives of (r,theta,phi) with respect to (x,y,z)
      drdidj(0,0) = _divrp - xp*xp*_divrp3;
      drdidj(0,1) = - xp*yp*_divrp3;
      drdidj(0,2) = - xp*zp*_divrp3;
      drdidj(1,1) = _divrp - yp*yp*_divrp3;
      drdidj(1,2) = - yp*zp*_divrp3;
      drdidj(2,2) = _divrp - zp*zp*_divrp3;
      
      dthetadidj(0,0) = zp*(-2.0*xp*xp*xp*xp-xp*xp*yp*yp+yp*yp*yp*yp+zp*zp*yp*yp)*(SQR(_divrp)*SQR(_divrp)*SQR(_divrhop)*_divrhop);
      dthetadidj(0,1) = - xp*yp*zp*(3.0*xp*xp+3.0*yp*yp+zp*zp)*(SQR(_divrp)*SQR(_divrp)*SQR(_divrhop)*_divrhop);
      dthetadidj(0,2) = xp*(xp*xp+yp*yp-zp*zp)*(SQR(_divrp)*(SQR(_divrp)*_divrhop));
      dthetadidj(1,1) = zp*(-2.0*yp*yp*yp*yp-yp*yp*xp*xp+xp*xp*xp*xp+zp*zp*xp*xp)*(SQR(_divrp)*SQR(_divrp)*SQR(_divrhop)*_divrhop);
      dthetadidj(1,2) = yp*(xp*xp+yp*yp-zp*zp)*(SQR(_divrp)*(SQR(_divrp)*_divrhop));
      dthetadidj(2,2) = 2.0*zp*rhop/(rp*rp*rp*rp);
						
      dphididj(0,0) = 2.0*yp*xp*(SQR(_divrhop)*SQR(_divrhop));  
      dphididj(0,1) = (yp*yp-xp*xp)*(SQR(_divrhop)*SQR(_divrhop));  
      dphididj(0,2) = 0.0;  
      dphididj(1,1) = - 2.0*yp*xp*(SQR(_divrhop)*SQR(_divrhop));  
      dphididj(1,2) = 0.0;  
      dphididj(2,2) = 0.0;  
      
      // Compute first derivatives of F
      for (int a = 0; a < NDIM; ++a) {
        dFdi(a) = drdi(a);
      }
      for (int a = 0; a < NDIM; ++a) {
        for(int l=0; l<=lmax; l++){  
          dFdi(a) -= a0(l) * dthetadi(a) * dY0dth(i,j,l);   
        }
      }
      for (int a = 0; a < NDIM; ++a) {
        for(int l=1; l<=lmax; l++){  
          for(int m=1; m<=l; m++){  
            int l1 = lmindex(l,m);
            dFdi(a) -= 
              ac(l1)*(dthetadi(a)*dYcdth(i,j,l1) + dphidi(a)*dYcdph(i,j,l1))+
              as(l1)*(dthetadi(a)*dYsdth(i,j,l1) + dphidi(a)*dYsdph(i,j,l1));   
          }  
        }
      }
      
      // Compute second derivatives of F
      for (int a = 0; a < NDIM; ++a) {
        for (int b = 0; b < NDIM; ++b) {
          dFdidj(a,b) = drdidj(a,b);  
	}
      }
      
      for (int a = 0; a < NDIM; ++a) {
        for (int b = 0; b < NDIM; ++b) {
          for(int l=0;l<=lmax;l++){  
            dFdidj(a,b) -= a0(l)*(dthetadidj(a,b)*dY0dth(i,j,l) + dthetadi(a)*dthetadi(b)*dY0dth2(i,j,l));   
          }   
          for(int l=1; l<=lmax; l++) {  
            for(int m=1;m<=l;m++){  
              int l1 = lmindex(l,m);
              dFdidj(a,b) -= ac(l1)*(dthetadidj(a,b)*dYcdth(i,j,l1)
                + dthetadi(a)*(dthetadi(b)*dYcdth2(i,j,l1) + dphidi(b)*dYcdthdph(i,j,l1))
                + dphididj(a,b)*dYcdph(i,j,l1)   
                + dphidi(a)*(dthetadi(b)*dYcdthdph(i,j,l1) + dphidi(b)*dYcdph2(i,j,l1)))
                + as(l1)*(dthetadidj(a,b)*dYsdth(i,j,l1)
                + dthetadi(a)*(dthetadi(b)*dYsdth2(i,j,l1) + dphidi(b)*dYsdthdph(i,j,l1))
                + dphididj(a,b)*dYsdph(i,j,l1)   
                + dphidi(a)*(dthetadi(b)*dYsdthdph(i,j,l1) + dphidi(b)*dYsdph2(i,j,l1)));  
            }  
          }
        }
      }
      
      // Compute dFdi with the index up
      for (int a = 0; a < NDIM; ++a) {
        dFdi_u(a) = 0;
        for (int b = 0; b < NDIM; ++b) {
          dFdi_u(a) += ginv(a,b)*dFdi(b);
        }
      }
      
      // Compute norm of dFdi
      Real norm = 0;
      for (int a = 0; a < NDIM; ++a) {
      	norm += dFdi_u(a)*dFdi(a);
      }
      
      Real u = (norm>0)? std::sqrt(norm) : 0.0;  
      
      // Compute nabla_a nabla_b F
      for (int a = 0; a < NDIM; ++a) {
        for (int b = 0; b < NDIM; ++b) {
          nnF(a,b) = dFdidj(a,b);
        }
      }
      nnF(0,0) -= 0.5*(dFdi_u(0)*dg(0,0,0,i,j) + dFdi_u(1)*(2.0*dg(0,1,0,i,j)-dg(1,0,0,i,j)) + dFdi_u(2)*(2.0*dg(0,2,0,i,j)-dg(2,0,0,i,j)));  
      nnF(0,1) -= 0.5*(dFdi_u(0)*dg(1,0,0,i,j) + dFdi_u(1)*dg(0,1,1,i,j) + dFdi_u(2)*(dg(0,2,1,i,j)+dg(1,2,0,i,j)-dg(2,1,0,i,j)));  
      nnF(0,2) -= 0.5*(dFdi_u(0)*dg(2,0,0,i,j) + dFdi_u(1)*(dg(0,2,1,i,j)+dg(2,1,0,i,j)-dg(1,2,0,i,j)) + dFdi_u(2)*dg(0,2,2,i,j)); 
      nnF(1,1) -= 0.5*(dFdi_u(0)*(2.0*dg(1,1,0,i,j)-dg(0,1,1,i,j)) + dFdi_u(1)*dg(1,1,1,i,j) + dFdi_u(2)*(2.0*dg(1,2,1,i,j)-dg(2,1,1,i,j)));  
      nnF(1,2) -= 0.5*(dFdi_u(0)*(2.0*dg(1,2,0,i,j)+dg(2,1,0,i,j)-dg(0,2,1,i,j)) + dFdi_u(1)*dg(2,1,1,i,j) + dFdi_u(2)*dg(1,2,2,i,j));
      nnF(2,2) -= 0.5*(dFdi_u(0)*(2.0*dg(2,2,0,i,j)-dg(0,2,2,i,j)) + dFdi_u(1)*(2.0*dg(2,2,1,i,j)-dg(1,2,2,i,j)) + dFdi_u(2)*dg(2,2,2,i,j));  
      
#if DEBUG_OUTPUT
      fprintf(fp_dFdxdx, "%23.15e %23.15e %23.15e\n", theta, phi, dFdidj(0,0));
      fprintf(fp_nnFxx, "%23.15e %23.15e %23.15e\n", theta, phi, nnF(0,0));
#endif      

      // Compute d2F = g^{ab} nabla_a nabla_b F
      Real d2F = 0.;
      for (int a = 0; a < NDIM; ++a) {
        for (int b = 0; b < NDIM; ++b) {
	  d2F += ginv(a,b)*nnF(a,b);
	}
      }
      
      // Compute dFd^a dFd^b Kab
      Real dFdadFdbKab = 0.;
      for (int a = 0; a < NDIM; ++a) {
	for (int b = 0; b < NDIM; ++b) {
	  dFdadFdbKab += dFdi_u(a) * dFdi_u(b) * K(a,b,i,j);
        }
      }
      
      // Compute dFd^a dFd^b Fdadb
      Real dFdadFdbFdadb = 0.;
      for (int a = 0; a < NDIM; ++a) {
	for (int b = 0; b < NDIM; ++b) {
	  dFdadFdbFdadb += dFdi_u(a) * dFdi_u(b) * nnF(a,b);
        }
      }
      
      // Expansion & rho = H * u * sigma (sigma=1)
      Real divu = (norm>0)? 1.0/u : NAN;
      Real H = (norm>0)? (d2F*divu + dFdadFdbKab*(divu*divu) - dFdadFdbFdadb*(divu*divu*divu) - TrK) : 0.0;
      rho(i,j) = H * u; //Real rho = H * u;

      // Normal vector
      for (int a = 0; a < NDIM; ++a) {
        R(a) = dFdi_u(a) * divu;
      }
      
      // Surface Element
      // ---------------
      
      // Derivatives of (x,y,z) vs (thetas, phi)

      // dr/dtheta, dr/dphi
      Real rtp1,rtm1,rpp1,rpm1; 
      
      if (i==0){
        rtp1 = rr(1,j);
        if (j<nphihalf)  rtm1 = rr(0,j+nphihalf);
        if (j>=nphihalf) rtm1 = rr(0,j-nphihalf);
      } else if (i==ntheta-1) {
        if (j<nphihalf)  rtp1 = rr(ntheta-1,j+nphihalf);
        if (j>=nphihalf) rtp1 = rr(ntheta-1,j-nphihalf);
        rtm1 = rr(ntheta-2,j);
      } else {               
        rtp1 = rr(i+1,j);
        rtm1 = rr(i-1,j);
      }
      
      if(j==0) {
        rpp1 = rr(i,1);
        rpm1 = rr(i,nphi-1);
      } else if(j==nphi-1) {
        rpp1 = rr(i,0);
        rpm1 = rr(i,nphi-2);
      } else {               
        rpp1 = rr(i,j+1);
        rpm1 = rr(i,j-1);
      }
            
      Real drdt = (rtp1-rtm1)/(2.0*dtheta);
      Real drdp = (rpp1-rpm1)/(2.0*dphi);

      // Derivatives of (x,y,z) with respect to theta
      dXdth(0) = (drdt*sinth + rr(i,j)*costh)*cosph;
      dXdth(1) = (drdt*sinth + rr(i,j)*costh)*sinph;
      dXdth(2) = drdt*costh - rr(i,j)*sinth;
      
      // Derivatives of (x,y,z) with respect to phi
      dXdph(0) = (drdp*cosph - rr(i,j)*sinph)*sinth;
      dXdph(1) = (drdp*sinph + rr(i,j)*cosph)*sinth;
      dXdph(2) = drdp*costh;
      
      // Induced metric on the horizon
      Real h11 = 0.;
      for (int a = 0; a < NDIM; ++a) {
        for (int b = 0; b < NDIM; ++b) {
	  h11 += dXdth(a) * dXdth(b) * g(a,b,i,j);
      	}
      }
      
      Real h12 = 0.;
      for (int a = 0; a < NDIM; ++a) {
        for (int b = 0; b < NDIM; ++b) {
          h12 += dXdth(a) * dXdph(b) * g(a,b,i,j);
        }
      }
      
      Real h22 = 0.;
      for (int a = 0; a < NDIM; ++a) {
        for (int b = 0; b < NDIM; ++b) {
	  h22 += dXdph(a) * dXdph(b) * g(a,b,i,j);
	}
      }
      
      // Determinant of the induced metric
      Real deth = h11*h22 - h12*h12;
      if (deth<0.) deth = 0.0;
      
      
      // Spin integrand
      // --------------
      
      // Local coordinates on surface (already defined above)
      // xp = rr(i,j) * sinth * cosph;
      // yp = rr(i,j) * sinth * sinph;
      // zp = rr(i,j) * costh;
           
      // Flat-space coordinate rotational KV
      phix(0) =  0;  
      phix(1) = -zp; // -(z-zc); 
      phix(2) =  yp; // (y-yc); 
      phiy(0) =  zp; // (z-zc);  
      phiy(1) =  0;  
      phiy(2) = -xp; // -(x-xc);  
      phiz(0) = -yp; // -(y-yc);  
      phiz(1) =  xp; // (x-xc);  
      phiz(2) =  0;  
      
      // Integrand of spin
      Real intSx = 0;
      for(int a = 0; a < NDIM; ++a) {
	      for(int b = 0; b < NDIM; ++b) {
	        intSx += phix(a) * R(b) * K(a,b,i,j);
	      }
      }
      
      Real intSy = 0;
      for(int a = 0; a < NDIM; ++a) {
	      for(int b = 0; b < NDIM; ++b) {
	        intSy += phiy(a) * R(b) * K(a,b,i,j);
	      }
      }
      
      Real intSz = 0;
      for(int a = 0; a < NDIM; ++a) {
	      for(int b = 0; b < NDIM; ++b) {
	        intSz += phiz(a) * R(b) * K(a,b,i,j);
	      }
      }
      
      // Local sums
      // ----------
      
      Real dw = dthdph * std::sqrt(deth);
      
      integrals[iarea]   += dw;
      integrals[icoarea] += dthdph * sinth * SQR(rr(i,j));
      integrals[ihrms]   += dw * SQR(H);
      integrals[ihmean]  += dw * H;
      integrals[iSx]     += dw * intSx;
      integrals[iSy]     += dw * intSy;
      integrals[iSz]     += dw * intSz;
      
    } // phi loop
  } // theta loop

#if DEBUG_OUTPUT
  fclose(fp_dFdxdx);
  fclose(fp_nnFxx);
#endif
  
#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, integrals, invar, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  
  drdi.DeleteTensorPointwise();
  dthetadi.DeleteTensorPointwise();
  dphidi.DeleteTensorPointwise();
  
  drdidj.DeleteTensorPointwise();
  dthetadidj.DeleteTensorPointwise();
  dphididj.DeleteTensorPointwise();
  
  dFdi.DeleteTensorPointwise();
  dFdi_u.DeleteTensorPointwise();
  dFdidj.DeleteTensorPointwise();
  
  ginv.DeleteTensorPointwise();

  R.DeleteTensorPointwise();

  dXdth.DeleteTensorPointwise();
  dXdph.DeleteTensorPointwise();
  
  phix.DeleteTensorPointwise();
  phiy.DeleteTensorPointwise();
  phiz.DeleteTensorPointwise();
  
  nnF.DeleteTensorPointwise();
}

//----------------------------------------------------------------------------------------
// \!fn bool AHF::CalculateMetricDerivatives(int iter, Real time)
// \brief CalculateMetricDerivatives
bool AHF::CalculateMetricDerivatives(int iter, Real time)
{
  if((time < start_time) || (time > stop_time)) return false;
  if (wait_until_punc_are_close && !(PuncAreClose())) return false;
  if (iter % compute_every_iter != 0) return false;

  // Compute and store ADM metric drvts at this iteration
  MeshBlock * pmb = pmesh->pblock;
  while (pmb != nullptr) {
    Z4c *pz4c = pmb->pz4c;
    pz4c->aux_g_ddd.NewAthenaTensor(pz4c->mbi.nn3, pz4c->mbi.nn2, pz4c->mbi.nn1);
    MetricDerivatives(pmb);
    pmb = pmb->next;
  }
  return true;
}

//----------------------------------------------------------------------------------------
// \!fn void AHF::Find(int iter, Real time)
// \brief Search for the horizons
void AHF::Find(int iter, Real time)
{
  if((time < start_time) || (time > stop_time)) return;
  if (wait_until_punc_are_close && !(PuncAreClose())) return;
  if (iter % compute_every_iter != 0) return;
  InitialGuess();
  FastFlowLoop();
}

//----------------------------------------------------------------------------------------
// \!fn bool AHF::DeleteMetricDerivatives(int iter, Real time)
// \brief DeleteMetricDerivatives
bool AHF::DeleteMetricDerivatives(int iter, Real time)
{
  if((time < start_time) || (time > stop_time)) return false;
  if (wait_until_punc_are_close && !(PuncAreClose())) return false;
  if (iter % compute_every_iter != 0) return false;

  // Delete tensors
  MeshBlock * pmb = pmesh->pblock;
  while (pmb != nullptr) {
    pmb->pz4c->aux_g_ddd.DeleteAthenaTensor();
    pmb = pmb->next;
  }
  return true;
}

//----------------------------------------------------------------------------------------
// \!fn void AHF::FastFlowLoop()
// \brief Fast Flow loop for horizon n
void AHF::FastFlowLoop()
{
  ah_found = false;

  Real meanradius = a0(0)/std::sqrt(4.0*PI);  
  Real mass = 0;
  Real mass_prev = 0;
  Real area = 0;
  Real hrms = 0;
  Real hmean = 0;
  Real Sx = 0;
  Real Sy = 0;
  Real Sz = 0;
  Real S = 0;
  bool failed = false;

  if (verbose && ioproc) {
    std::cout << "\nSearching for horizon " << nh << std::endl;
    std::cout << "center = ("
	      << center[0] << ","
	      << center[1] << ","
	      << center[2] << ")" << std::endl;
    std::cout << "r_mean = " << meanradius << std::endl;
    std::cout << " iter      area            mass         meanradius        hmean            Sx              Sy              Sz             S" << std::endl;
  }

  for(int k=0; k<flow_iterations; k++){
    fastflow_iter = k;
    // Compute radius r = a_lm Y_lm
    RadiiFromSphericalHarmonics();
    
    // In MetricInterp() we'll flag the surface points on this mesh
    // default to 0 (no points)
    havepoint.ZeroClear();

    // Metric interpolated on the surface
    g.ZeroClear();
    dg.ZeroClear();
    K.ZeroClear();
    
    // Interpolate metric on surface
    // Flag surface points contained in the MBs
    MeshBlock * pmb = pmesh->pblock;
    while (pmb != nullptr) {
      MetricInterp(pmb);
      pmb = pmb->next; 
    }

    // if havepoint(i,j) > 1 point (i,j) belongs to multibe MBs ...
    //TODO should not happen, check?
    SurfaceIntegrals();

    area  = integrals[iarea];
    hrms  = integrals[ihrms]/area;
    hmean = integrals[ihmean];
    Sx = integrals[iSx]/(8*PI);
    Sy = integrals[iSy]/(8*PI);
    Sz = integrals[iSz]/(8*PI);
    S  = std::sqrt(SQR(Sx)+SQR(Sy)+SQR(Sz));

    meanradius = a0(0)/std::sqrt(4.0*PI);

    // Irreducible mass
    mass_prev = mass;
    mass = std::sqrt(area/(16.0*PI));     

    if (verbose && ioproc) {
      printf("%3d %15.7e %15.7e %15.7e %15.7e %15.7e %15.7e %15.7e %15.7e\n", k, area, mass, meanradius, hmean, Sx, Sy, Sz, S);
    }

    if (std::fabs(hmean) > hmean_tol) {
      if (verbose && ioproc) {
        std::cout << "Failed hmean > " << hmean_tol << std::endl;
      }
      failed = true;
      break;
     }

    if (meanradius < 0.) {
      if (verbose && ioproc) {
       std::cout << "Failed meanradius < 0" << std::endl;
      }
      failed = true;
      break;
    }

    // End flow when mass difference is small
    if (std::fabs(mass_prev-mass) < mass_tol) {
      ah_found = true;
      break;
    }

    // Find new spectral components
    UpdateFlowSpectralComponents();
  }
   
  if (ah_found) {

    last_a0 = a0(0);
    
    ah_prop[harea] = area;
    ah_prop[hcoarea] = integrals[icoarea];
    ah_prop[hhrms] = hrms;
    ah_prop[hhmean] = hmean;
    ah_prop[hmeanradius] = meanradius;
    ah_prop[hSx] = Sx;
    ah_prop[hSy] = Sy;
    ah_prop[hSz] = Sz;    
    ah_prop[hS]  = S;
    ah_prop[hmass] = std::sqrt( SQR(mass) + 0.25*SQR(S/mass) ); // Christodoulu mass
    
  }
  
  if (verbose && ioproc) {
    
    if (ah_found) {
      std::cout << "Found horizon " << nh << std::endl;
      std::cout << " mass_irr = " << mass << std::endl;
      std::cout << " meanradius = " << meanradius << std::endl;
      std::cout << " hrms = " << hrms << std::endl;
      std::cout << " hmean = " << hmean << std::endl;
      std::cout << " Sx = " << Sx << std::endl;
      std::cout << " Sy = " << Sy << std::endl;
      std::cout << " Sz = " << Sz << std::endl;
      std::cout << " S  = " << S << std::endl;
    } else if (!failed && !ah_found) {
      std::cout << "Failed, reached max iterations " << flow_iterations << std::endl;
    }
  }
}

//----------------------------------------------------------------------------------------
// \!fn void AHF::UpdateFlowSpectralComponents(const int n)
// \brief find new spectral components with fast flow
void AHF::UpdateFlowSpectralComponents()
{
  const Real alpha = flow_alpha_beta_const;
  const Real beta = 0.5 * flow_alpha_beta_const;
  const Real A = alpha/(lmax*lmax1) + beta;
  const Real B = beta/alpha;

  Real * spec0 = new Real[lmax1];
  Real * specc = new Real[lmpoints];
  Real * specs = new Real[lmpoints];
 
  const Real dtheta = dth_grid(); 
  const Real dphi   = dph_grid();
  const Real dthdph = dtheta*dphi;

  // Local sums
  for(int l=0; l<=lmax; l++){   
    
    spec0[l] = 0;
    
    for(int m=0; m<=l; m++){
      
      int l1 = lmindex(l,m);
      
      specc[l1] = 0;
      specs[l1] = 0;
      
      for(int i=0; i<ntheta; i++){
	
	      Real theta = th_grid(i);
	      Real dw = dthdph * std::sin(theta);
	
        for(int j=0; j<nphi; j++){ 
	  
          if (!havepoint(i,j)) continue;
          
          if (m==0) {
            spec0[l] += dw * Y0(i,j,l) * rho(i,j);
          }
          specc[l1] += dw * Yc(i,j,l1) * rho(i,j);
          specs[l1] += dw * Ys(i,j,l1) * rho(i,j);
          
	      }//phi loop
      }//theta loop
      
    }//m loop
  }//l loop
  
// MPI reduce  
#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, spec0, lmax1,    MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, specc, lmpoints, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, specs, lmpoints, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  
  // Update the coefs
  for(int l=0; l<=lmax; l++){   
    
    Real ABfact = A/(1.0+B*l*(l+1));
    
    a0(l) -= ABfact * spec0[l]; 
    
    for(int m=0; m<=l; m++){
      int l1 = lmindex(l,m);
      ac(l1) -= ABfact * specc[l1];
      as(l1) -= ABfact * specs[l1];
    }
  }
  
  delete[] spec0;
  delete[] specc;
  delete[] specs;
  
}

//----------------------------------------------------------------------------------------
// \!fn void AHF::RadiiFromSphericalHarmonics()
// \brief compute the radius of the surface
void AHF::RadiiFromSphericalHarmonics()
{
  rr.ZeroClear();
  for(int i=0; i<ntheta; i++){
    for(int j=0; j<nphi; j++){      
      for(int l=0; l<=lmax; l++){
        rr(i,j) += a0(l)*Y0(i,j,l);
      }
      for(int l=1; l<=lmax; l++){
        for(int m=1;m<=l;m++){
	        int l1 = lmindex(l,m);
	        rr(i,j) += Yc(i,j,l1) * ac(l1) + Ys(i,j,l1) * as(l1);
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
// \!fn void AHF::InitialGuess()
// \brief initial guess for spectral coefs of horizon n
void AHF::InitialGuess()
{
  // Reset Coefficients to Zero  
  a0.ZeroClear();
  ac.ZeroClear();
  as.ZeroClear();
	
  if (use_puncture>=0) {
    // Update the center to the puncture position
    center[0] = pmesh->pz4c_tracker[use_puncture]->GetPos(0);
    center[1] = pmesh->pz4c_tracker[use_puncture]->GetPos(1);
    center[2] = pmesh->pz4c_tracker[use_puncture]->GetPos(2);
    // Update a0
    // For single BH in isotropic coordinates: horizon radius=m/2
    // but make sure it can surround all punctures comfortably, i.e.
    // make radius a bit larger than half the distance between any of the punctures
    Real mass = pmesh->pz4c_tracker[use_puncture]->GetMass();
    Real largedist = PuncMaxDistance(use_puncture);
    if (ah_found && last_a0>0) {
      a0(0) = last_a0 * expand_guess;
    } else {
      a0(0) = std::max(0.5 * mass, std::min(mass, 0.5 * largedist)); 
      a0(0) *= std::sqrt(4.0*PI);
    }
    return;
  }

  if (use_puncture_massweighted_center) {
    // Update the center based on the mass-weighted distance
    Real pos[3];
    PuncWeightedMassCentralPoint(&pos[0],&pos[1],&pos[2]);
    center[0] = pos[0];
    center[1] = pos[1];
    center[2] = pos[2];
  }

  // Take a0 either from previous or from input value
  if (ah_found && last_a0>0) {
    a0(0) = last_a0 * expand_guess;
  } else {
    a0(0) = std::sqrt(4.0*PI) * initial_radius;
  }

}

//----------------------------------------------------------------------------------------
// \!fn void AHF::ComputeSphericalHarmonics()
// \brief compute spherical harmonics for grid of size ntheta*nphi.
// Results are used for all horizons.
void AHF::ComputeSphericalHarmonics()
{
  const Real sqrt2 = std::sqrt(2.0);

  Y0.ZeroClear();
  Yc.ZeroClear();
  Ys.ZeroClear();

  dY0dth.ZeroClear();
  dYcdth.ZeroClear();
  dYsdth.ZeroClear();
  dYcdph.ZeroClear();
  dYsdph.ZeroClear();

  dY0dth2.ZeroClear();
  dYcdth2.ZeroClear(); 
  dYcdthdph.ZeroClear();
  dYsdth2.ZeroClear();
  dYsdthdph.ZeroClear();
  dYcdph2.ZeroClear();
  dYsdph2.ZeroClear();
	    
  // Legendre polynomials
  P.NewAthenaArray(lmax1,lmax1);
  dPdth.NewAthenaArray(lmax1,lmax1);
  dPdth2.NewAthenaArray(lmax1,lmax1);
  
  for(int i=0; i<ntheta; ++i){    
    
    Real theta = th_grid(i);
    
    ComputeLegendre(theta);
    
    for(int j=0; j<nphi; ++j){	

      Real phi = ph_grid(j);
      
      // l=0 spherical harmonics and drvts
      for(int l=0; l<=lmax; l++){
        Y0(i,j,l) = P(l,0);
	      dY0dth(i,j,l) = dPdth(l,0);
	      dY0dth2(i,j,l) = dPdth2(l,0);
      }
      
      // l>=1 spherical harmonics and drvts
      for(int l=1; l<=lmax; l++){
        for(int m=1; m<=l; m++){
	  
          int l1 = lmindex(l,m);
	  
	        Real cosmph = std::cos(m*phi);
	        Real sinmph = std::sin(m*phi);
	   
	        // spherical harmonics
          Yc(i,j,l1) = sqrt2 * P(l,m) * cosmph;
          Ys(i,j,l1) = sqrt2 * P(l,m) * sinmph;
	  
          // first drvts
          dYcdth(i,j,l1) =  sqrt2 * dPdth(l,m) * cosmph;
          dYsdth(i,j,l1) =  sqrt2 * dPdth(l,m) * sinmph;
          dYcdph(i,j,l1) = -sqrt2 * P(l,m) * m * sinmph;
          dYsdph(i,j,l1) =  sqrt2 * P(l,m) * m * cosmph;
	  
	        // second drvts
	        dYcdth2(i,j,l1)   =  sqrt2 * dPdth2(l,m) * cosmph;
          dYcdthdph(i,j,l1) = -sqrt2 * dPdth(l,m)  * m * sinmph;
          dYsdth2(i,j,l1)   =  sqrt2 * dPdth2(l,m) * sinmph;
          dYsdthdph(i,j,l1) =  sqrt2 * dPdth(l,m)  * m * cosmph;
          dYcdph2(i,j,l1)   = -sqrt2 * P(l,m) * m * m * cosmph;
          dYsdph2(i,j,l1)   = -sqrt2 * P(l,m) * m * m * sinmph;
        }
      }
    } // phi loop
  } // theta loop

  P.DeleteAthenaArray();
  dPdth.DeleteAthenaArray();
  dPdth2.DeleteAthenaArray();

}

//----------------------------------------------------------------------------------------
// \!fn void AHF::ComputeLegendre(const Real theta)
// \brief compute Legendre polys for l>=m and derivatives
void AHF::ComputeLegendre(const Real theta)
{
  const Real costh = std::cos(theta);
  const Real sinth = std::sin(theta);
  //const Real sqrt3 = std::sqrt(3.);
  
  int l,m; // Need persistent indexes
  
  // Precompute list of factorial
  Real * fac = new Real[2*lmax1+1];
  factorial_list(fac, 2*lmax1);  

  // Initialize P, dPdth and dPdth2
  P.ZeroClear();
  dPdth.ZeroClear();
  dPdth2.ZeroClear();

  // Compute the Legendre functions
  // diagonal terms
  for(l=0; l<=lmax; ++l){
    P(l,l) = std::sqrt((2*l+1)*fac[2*l]/(4.0*PI))/(std::pow(2,l)*fac[l])*std::pow((-sinth),l);
  }
  
  // the loop has special treatment for all (l,l-1) where (l-2,l-1) is not needed.
  P(1,0) = SQRT3*costh*P(0,0);
  for(l=2; l<=lmax; l++){
    for(m=0; m<l-1; m++){
      P(l,m) = std::sqrt((Real)(2*l+1)/(l*l-m*m));
      P(l,m) *= (std::sqrt((Real)2*l-1)*costh*P(l-1,m)
		 - std::sqrt((Real)((l-1)*(l-1)-m*m)/(2*l-3))*P(l-2,m));
    }
    // do (l,l-1) separately otherwise P(l-2,m) not defined
    P(l,l-1) = std::sqrt((Real)(2*l+1)/(l*l-m*m))*(std::sqrt((Real)2*l-1)*costh*P(l-1,m));
  }

  // Compute first derivatives of the Legendre functions
  for(l=0;l<=lmax;l++){
    dPdth(l,l) = std::sqrt((2*l+1)*fac[2*l]/(4.0*PI))
      /(std::pow(2,l)*fac[l])*l*std::pow((-sinth),l-1)*(-costh);
  }
  
  dPdth(1,0) = SQRT3*(-sinth*P(0,0)+costh*dPdth(0,0));

  for(l=2;l<=lmax;l++){
    for(m=0;m<l-1;m++){
      dPdth(l,m) = std::sqrt((Real)(2*l+1)/(l*l-m*m));
      dPdth(l,m) *= (std::sqrt((Real)2*l-1)*(-sinth*P(l-1,m) + costh*dPdth(l-1,m))
			-std::sqrt((Real)((l-1)*(l-1)-m*m)/(2*l-3))*dPdth(l-2,m));
    } 
    dPdth(l,l-1) = std::sqrt((Real)(2*l+1)/(l*l-m*m));
    dPdth(l,l-1) *= (std::sqrt((Real)2*l-1)*(-sinth*P(l-1,m) + costh*dPdth(l-1,m)) );
  }

  // Compute second derivatives of the Legendre functions
  for(l=0;l<=lmax;l++){
    dPdth2(l,l) = std::sqrt((Real)(2*l+1)*fac[2*l]/(4.0*PI))/(std::pow(2,l)*fac[l])*l
      *((l-1)*std::pow(-sinth,l-2)*costh*costh 
	    + std::pow(-sinth,l-1)*sinth);
  }
  
  dPdth2(1,0) = SQRT3*(- costh*P(0,0)-2.0*sinth*dPdth(0,0) + costh*dPdth2(0,0));
  
  for(l=2;l<=lmax;l++){
    for(m=0;m<l-1;m++){
      dPdth2(l,m) =
	      std::sqrt((Real)(2*l+1)/(l*l-m*m))*(std::sqrt((Real)2*l-1)*(-costh*P(l-1,m) - 2.0*sinth*dPdth(l-1,m) + costh*dPdth2(l-1,m))
			  - std::sqrt((Real)((l-1)*(l-1)-m*m)/(2*l-3))*dPdth2(l-2,m));
    }
    // m will be l-1
    dPdth2(l,m) = std::sqrt((Real)(2*l+1)/(l*l-m*m));
    dPdth2(l,m) *= (std::sqrt((Real)2*l-1)*(-costh*P(l-1,m) - 2.0*sinth*dPdth(l-1,m) + costh*dPdth2(l-1,m)) );
  }
  
  delete[] fac;
}

//----------------------------------------------------------------------------------------
// \!fn int AHF::lmindex(const int l, const int m)
// \brief multipolar single index (l,m) -> index
int AHF::lmindex(const int l, const int m)
{
  return l*lmax1 + m; 
}

//----------------------------------------------------------------------------------------
// \!fn int AHF::tpindex(const int i, const int j)
// \brief spherical grid single index (i,j) -> index
int AHF::tpindex(const int i, const int j)
{
  return i*nphi + j; 
}

//----------------------------------------------------------------------------------------
// \!fn Real AHF::th_grid(const int i)
// \brief theta coordinate from index
Real AHF::th_grid(const int i)
{
  Real dtheta = dth_grid();
  return dtheta*(0.5 + i);
}

//----------------------------------------------------------------------------------------
// \!fn Real AHF::ph_grid(const int i)
// \brief phi coordinate from index
Real AHF::ph_grid(const int j)
{
  Real dphi = dph_grid();
  return dphi*(0.5 + j);
}

//----------------------------------------------------------------------------------------
// \!fn Real AHF::dth_grid()
// \brief compute spacing dtheta 
Real AHF::dth_grid()
{
  return PI/ntheta;
}

//----------------------------------------------------------------------------------------
// \!fn Real AHF::dph_grid()
// \brief compute spacing dphi

Real AHF::dph_grid()
{
  return 2.0*PI/nphi;
}

//----------------------------------------------------------------------------------------
// \!fn void AHF::factorial_list(Real * fac, int maxn)
// \brief ist of factorials up to maxn

void AHF::factorial_list(Real * fac, const int maxn)
{
  fac[0] = 1.0;
  for (int i=1; i<=maxn; ++i)
    fac[i] = fac[i-1]*i;
}

//----------------------------------------------------------------------------------------
// \!fn Real AHF::PuncMaxDistance()
// \brief Max Euclidean distance between punctures

Real AHF::PuncMaxDistance() {
  Real maxdist = 0.0;
  for (int pix = 0; pix < npunct; ++pix) {
    Real xp = pmesh->pz4c_tracker[pix]->GetPos(0);
    Real yp = pmesh->pz4c_tracker[pix]->GetPos(1);
    Real zp = pmesh->pz4c_tracker[pix]->GetPos(2);
    for (int p = pix+1; p < npunct; ++p) {
      Real x = pmesh->pz4c_tracker[p]->GetPos(0);
      Real y = pmesh->pz4c_tracker[p]->GetPos(1);
      Real z = pmesh->pz4c_tracker[p]->GetPos(2);
      maxdist = std::max(maxdist, std::sqrt(SQR(x-xp)+SQR(y-yp)+SQR(z-zp)));
    }
  }
  return maxdist;
}

//----------------------------------------------------------------------------------------
// \!fn Real AHF::PuncMaxDistance(const int pix)
// \brief Max Euclidean distance from puncture pix to other punctures

Real AHF::PuncMaxDistance(const int pix) {
  Real xp = pmesh->pz4c_tracker[pix]->GetPos(0);
  Real yp = pmesh->pz4c_tracker[pix]->GetPos(1);
  Real zp = pmesh->pz4c_tracker[pix]->GetPos(2);
  Real maxdist = 0.0;
  for (int p = 0; p < npunct; ++p) {
    if (p==pix) continue;
    Real x = pmesh->pz4c_tracker[p]->GetPos(0);
    Real y = pmesh->pz4c_tracker[p]->GetPos(1);
    Real z = pmesh->pz4c_tracker[p]->GetPos(2);
    maxdist = std::max(maxdist, std::sqrt(SQR(x-xp)+SQR(y-yp)+SQR(z-zp)));
  }
  return maxdist;
}

//----------------------------------------------------------------------------------------
// \!fn Real AHF::PuncSumMasses() 
// \brief Return sum of puncture's intial masses

Real AHF::PuncSumMasses() {
  Real mass = 0.0;
  for (int p = 0; p < npunct; ++p) {
    mass += pmesh->pz4c_tracker[p]->GetMass();
  }
  return mass;
}

//----------------------------------------------------------------------------------------
// \!fn void AHF::PuncWeightedMassCentralPoint(Real *xc, Real *yc, Real *zc) 
// \brief Return mss-weighted center of puncture positions

void AHF::PuncWeightedMassCentralPoint(Real *xc, Real *yc, Real *zc) {  
  Real sumx = 0.0; // sum of m_i*x_i
  Real sumy = 0.0;
  Real sumz = 0.0;
  Real divsum = 0.0; // sum of m_i to later divide by
  for (int p = 0; p < npunct; p++) {
    Real x = pmesh->pz4c_tracker[p]->GetPos(0);
    Real y = pmesh->pz4c_tracker[p]->GetPos(1);
    Real z = pmesh->pz4c_tracker[p]->GetPos(2);
    Real m = pmesh->pz4c_tracker[p]->GetMass();
    sumx += m*x;
    sumy += m*y;
    sumz += m*z;
    divsum += m;
  }
  divsum = 1.0/divsum;
  *xc = sumx * divsum;
  *yc = sumy * divsum;
  *zc = sumz * divsum;
}

//----------------------------------------------------------------------------------------
// \!fn int AHF::PuncAreClose()
// \brief Check when the maximal distance between all punctures is below threshold

bool AHF::PuncAreClose() {
  Real const mass = PuncSumMasses();
  Real const maxdist = PuncMaxDistance();
  return (maxdist < merger_distance * mass);  
}
