//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//! \file gr_SGRID_bns.cpp
//  \brief Initial conditions for binary neutron stars.
//         Interpolation of SGRID initial data.
//         Requires SGRID library:
//         https://github.com/sgridsource

#include <cassert>
#include <iostream>
#include <cstring>
#include <filesystem>

// libsgrid
// Functions protoypes are "SGRID_*"
// #include "DNSdataReader.h"

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../athena_tensor.hpp"
#include "../parameter_input.hpp"
#include "../coordinates/coordinates.hpp"
#include "../z4c/z4c.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../bvals/bvals.hpp"
#include "../mesh/mesh.hpp"
#include "../mesh/mesh_refinement.hpp"
#include "../trackers/extrema_tracker.hpp"
#include "../utils/linear_algebra.hpp"
#include "../utils/utils.hpp"
#include "../globals.hpp"

extern "C" {
  int libsgrid_main(int argc, char **argv);
  extern int SGRID_memory_persists;
  int SGRID_grid_exists(void);

  void SGRID_errorexits(char *file, int line, char *s, char *t);
  #define SGRID_errorexits(s,t) SGRID_errorexits(__FILE__, __LINE__, (s), (t))

  int SGRID_system2(char *s1, char *s2);
  int SGRID_lock_curr_til_EOF(FILE *out);
  int SGRID_construct_argv(char *str, char ***argv);

  int SGRID_fgotonext(FILE *in, const char *label);
  int SGRID_fgetparameter(FILE *in, const char *par, char *str);
  int SGRID_extract_after_EQ(char *str);
  int SGRID_extrstr_before_after_EQ(const char *str, char *before, char *after);
  int SGRID_fscanline(FILE *in,char *str);
  int SGRID_extrstr_before_after(const char *str, char *before, char *after, char z);
  int SGRID_find_before_after(const char *str, char *before, char *after, const char *z);
  int SGRID_pfind_before_after(const char *str,int p,char *before,char *after,const char *z);
  int SGRID_sscan_word_at_p(const char *str, int p, char *word);
  int SGRID_fscan_str_using_getc(FILE *in, char *str);
  int SGRID_fscanf1(FILE *in, char *fmt, char *str);
  void SGRID_free_everything();

  void SGRID_EoS_T0_rho0_P_rhoE_from_hm1(double hm1,
                                         double *rho0, double *P, double *rhoE);
  double SGRID_epsl_of_rho0_rhoE(double rho0, double rhoE);

  int SGRID_DNSdata_Interpolate_ADMvars_to_xyz(double xyz[3], double *vars,
                                               int init);
}

#define STRLEN (16384)

// Indexes for Initial data variables (IDVars) 
enum{idvar_alpha,
  idvar_Bx,idvar_By, idvar_Bz,
  idvar_gxx, idvar_gxy, idvar_gxz, idvar_gyy, idvar_gyz, idvar_gzz,
  idvar_Kxx, idvar_Kxy, idvar_Kxz, idvar_Kyy, idvar_Kyz, idvar_Kzz,
  idvar_q,
  idvar_VRx, idvar_VRy, idvar_VRz, 
  idvar_NDATAMAX, //TODO in NMESH this is 23, but they are 20, and only 20 used...
};

namespace {
  int RefinementCondition(MeshBlock *pmb);

  Real linear_interp(Real *f, Real *x, int n, Real xv);
  int interp_locate(Real *x, int Nx, Real xval);

#if MAGNETIC_FIELDS_ENABLED
  Real DivBface(MeshBlock *pmb, int iout);
#endif

  Real max_rho(      MeshBlock *pmb, int iout);
  Real min_alpha(    MeshBlock *pmb, int iout);
  Real max_abs_con_H(MeshBlock *pmb, int iout);

  // Global variables
  //TODO ... EOS etc.

  // Utilities wrapping various SGRID DNS calls (DNS_*)
  void DNS_init_sgrid(ParameterInput *pin);
  int DNS_position_fileptr_after_str(FILE *in, const char *str);
  int DNS_parameters(ParameterInput *pin);
  int DNS_call_sgrid(const char *command);
}

namespace fs = std::filesystem;

// void SGRIDHistory(HistoryData *pdata, Mesh *pm);

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class. Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file. Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin)
{
  if(adaptive==true)
    EnrollUserRefinementCondition(RefinementCondition);

  AllocateUserHistoryOutput(4);
  EnrollUserHistoryOutput(0, max_rho,   "max-rho",
    UserHistoryOperation::max);
  EnrollUserHistoryOutput(1, min_alpha, "min-alpha",
    UserHistoryOperation::min);
  EnrollUserHistoryOutput(2, max_abs_con_H, "max-abs-con.H",
    UserHistoryOperation::max);

#if MAGNETIC_FIELDS_ENABLED
  // AllocateUserHistoryOutput(1);
  EnrollUserHistoryOutput(3, DivBface, "divBface");
#endif

  //TODO ... Here we might need some preparation for EOS, etc. 
  // #ifdef LORENE_EOS
  //   LORENE_EoS_fname   = pin->GetString("hydro", "lorene");
  // #if EOS_POLICY_CODE == 2
  //   LORENE_EoS_fname_Y = pin->GetString("hydro", "lorene_Y");
  // #endif
  //   LORENE_EoS_Table = new LoreneTable;
  //   ReadLoreneTable(LORENE_EoS_fname, LORENE_EoS_Table);
  // #if EOS_POLICY_CODE == 2
  //   ReadLoreneFractions(LORENE_EoS_fname_Y, LORENE_EoS_Table);
  // #endif
  //   ConvertLoreneTable(LORENE_EoS_Table);
  //   LORENE_EoS_Table->rho_atm = pin->GetReal("hydro", "dfloor"); 
  // #endif
 
  if (!resume_flag) {
    // Check on some input parameters
    std::string datadir = pin->GetOrAddString("problem", "datadir", "");
    if (datadir.empty()) {
          std::stringstream msg;
	  msg << "### FATAL ERROR parameter datadir: " << datadir << " "
	      << " not found. This is needed.";
	  ATHENA_ERROR(msg);
    }
    std::string outdir  = pin->GetOrAddString("problem", "outdir", "SGRID");
    // Check if the directory exists and create it if it doesn't
    if (!fs::exists(outdir)) {
        if (!fs::create_directory(outdir)) {
            std::cerr << "Failed to create directory!" << std::endl;
        }
    }

     // Alloc memory and read data
    // Alloc memory and read data
    DNS_init_sgrid(pin);
    // Read SGRID parameters from BNSdata_properties.txt file
    DNS_parameters(pin);
  }
  
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Interpolate SGRID DNS data onto the grid.
//         Assumes SGRID parameters are read correctly from SGRID's BNSdata_properties.txt
//========================================================================================
void MeshBlock::ProblemGenerator(ParameterInput *pin)
{
  //if (resume_flag) 
    //return;

  bool verbose = pin->GetOrAddBoolean("problem", "verbose", 0);
//
  // Initialize the data reader
  DNS_init_sgrid(pin);
  DNS_parameters(pin);

  //
  // Get settings for SGRID lib
  //
  
  const Real sgrid_x_CM = pin->GetReal("problem", "x_CM");
  const Real Omega = pin->GetReal("problem", "Omega");
  const Real ecc = pin->GetReal("problem", "ecc");
  const Real xmax1 = pin->GetReal("problem","xmax1");
  const Real xmax2 = pin->GetReal("problem","xmax2");

  const Real rdot  = pin->GetReal("problem","rdot");
  const Real rdotor = rdot/(xmax1-xmax2);

  const int rotation180 = pin->GetOrAddInteger("problem","180rotation",0);
  const Real s180 = (1 - 2*rotation180);

  //TODO these setting are in the NMESH example,
  //     but seem redundant, untested, useless here. Remove.
  //int idtalpha = -666; //FIXME: wrong!!!
  //int idtbetax = -666;
  //int set_dtlapse = 0;
  //int set_dtshift = 0;
  //int set_lapse = 1;
  //int set_shift = 1;
  //int set_hydro = 1;

  // Prepare SGRID interpolator
  if (verbose) std::printf("Initializing SGRID_DNSdata_Interpolate_ADMvars_to_xyz\n");
  SGRID_DNSdata_Interpolate_ADMvars_to_xyz(NULL, NULL, 1);

  // Initial data variables at one point 
  // 20 values for the fields at (x_i,y_j,z_k) ordered as:
  //  alpha DNSdata_Bx DNSdata_By DNSdata_Bz
  //  gxx gxy gxz gyy gyz gzz
  //  Kxx Kxy Kxz Kyy Kyz Kzz
  //  q VRx VRy VRz
  Real IDvars[idvar_NDATAMAX]; 
  
  //
  // Settings for Athena++
  //

  Real const tol_det_zero = pin->GetOrAddReal("problem","tolerance_det_zero",1e-10);
  
  // container with idx / grids pertaining z4c
  MB_info* mbi = &(pz4c->mbi);

  const int N = NDIM;

  typedef AthenaTensor<Real, TensorSymm::NONE, N, 0> AT_N_sca;
  typedef AthenaTensor<Real, TensorSymm::NONE, N, 1> AT_N_vec;
  typedef AthenaTensor<Real, TensorSymm::SYM2, N, 2> AT_N_sym;

  // Set some aliases for the variables.
  AT_N_sca alpha( pz4c->storage.adm, Z4c::I_ADM_alpha);
  AT_N_vec beta_u(pz4c->storage.adm, Z4c::I_ADM_betax);
  AT_N_sym g_dd(  pz4c->storage.adm, Z4c::I_ADM_gxx);
  AT_N_sym K_dd(  pz4c->storage.adm, Z4c::I_ADM_Kxx);

  // Stuff for matter grid
  const int il = 0;
  const int iu = ncells1-1;
  const int jl = 0;
  const int ju = ncells2-1;
  const int kl = 0;
  const int ku = ncells3-1;

  // Atmosphere and EOS parameters
  //TODO ...
#if USETM
    Real rho_atm = pin->GetReal("hydro", "dfloor");
    Real T_atm = pin->GetReal("hydro", "tfloor");
    Real mb = peos->GetEOS().GetBaryonMass();
    Real Y_atm[MAX_SPECIES] = {0.0};
#if EOS_POLICY_CODE == 2
    Y_atm[0] = pin->GetReal("hydro", "y0_atmosphere");
#endif
#else
    Real k_adi = pin->GetReal("hydro", "k_adi");
    Real gamma_adi = pin->GetReal("hydro","gamma");
#endif
  
  //
  // Interpolate on spacetime grid 
  //
  
  for (int k=0; k<mbi->nn3; ++k)
  for (int j=0; j<mbi->nn2; ++j)
  for (int i=0; i<mbi->nn1; ++i)
    {
      Real zb = mbi->x3(k);
      Real yb = mbi->x2(j) * s180; // multiply by -1 if 180 degree rotation
      Real xb = mbi->x1(i) * s180; 
      Real xs = xb + sgrid_x_CM;   // shift x-coord 
      Real xyz[3] = {xs, yb, zb};

      // Interpolate
      // This call is supposed to be threadsafe, it contains an OMP Critical
      SGRID_DNSdata_Interpolate_ADMvars_to_xyz(xyz, IDvars, 0);

      // transform some tensor components, if we have a 180 degree rotation 
      IDvars[idvar_Bx]  *= s180;
      IDvars[idvar_By]  *= s180;
      IDvars[idvar_gxz] *= s180;
      IDvars[idvar_gyz] *= s180;
      IDvars[idvar_Kxz] *= s180;
      IDvars[idvar_Kyz] *= s180;
      IDvars[idvar_VRx] *= s180;
      IDvars[idvar_VRy] *= s180;

      alpha(k, j, i)     = IDvars[idvar_alpha];
      beta_u(0, k, j, i) = IDvars[idvar_Bx];
      beta_u(1, k, j, i) = IDvars[idvar_By];
      beta_u(2, k, j, i) = IDvars[idvar_Bz];

      g_dd(0, 0, k, j, i) = IDvars[idvar_gxx];
      g_dd(0, 1, k, j, i) = IDvars[idvar_gxy];
      g_dd(0, 2, k, j, i) = IDvars[idvar_gxz];
      g_dd(1, 1, k, j, i) = IDvars[idvar_gyy];
      g_dd(1, 2, k, j, i) = IDvars[idvar_gyz];
      g_dd(2, 2, k, j, i) = IDvars[idvar_gzz];
      
      K_dd(0, 0, k, j, i) = IDvars[idvar_Kxx];
      K_dd(0, 1, k, j, i) = IDvars[idvar_Kxy];
      K_dd(0, 2, k, j, i) = IDvars[idvar_Kxz];
      K_dd(1, 1, k, j, i) = IDvars[idvar_Kyy];
      K_dd(1, 2, k, j, i) = IDvars[idvar_Kyz];
      K_dd(2, 2, k, j, i) = IDvars[idvar_Kzz];

      const Real det = LinearAlgebra::Det3Metric(g_dd, k, j, i);
      assert(std::fabs(det) > tol_det_zero);

    }
  
  //
  // Interpolate on matter grid 
  //

  for (int k = kl; k <= ku; ++k)
  for (int j = jl; j <= ju; ++j)
  for (int i = il; i <= iu; ++i)
    {
      Real zb = pcoord->x3v(k);
      Real yb = pcoord->x2v(j) * s180; // multiply by -1 if 180 degree rotation
      Real xb = pcoord->x1v(i) * s180; 
      Real xs = xb + sgrid_x_CM;       // shift x-coord 
      Real xyz[3] = {xs, yb, zb};

      // Interpolate
      // This call is supposed to be threadsafe, it contains an OMP Critical
      SGRID_DNSdata_Interpolate_ADMvars_to_xyz(xyz, IDvars, 0);

      // Transform some tensor components, if we have a 180 degree rotation 
      IDvars[idvar_Bx]  *= s180;
      IDvars[idvar_By]  *= s180;
      IDvars[idvar_gxz] *= s180;
      IDvars[idvar_gyz] *= s180;
      IDvars[idvar_Kxz] *= s180;
      IDvars[idvar_Kyz] *= s180;
      IDvars[idvar_VRx] *= s180;
      IDvars[idvar_VRy] *= s180;

      // Primitives
      Real rho = 0.0;
      Real pre = 0.0;
      Real eps = 0.0;
      Real v_u_x = 0.0, v_u_y = 0.0, v_u_z = 0.0;
      
      // if we are in matter region, convert q, VR to rho, press, eps, v^i :
      if (IDvars[idvar_q]>0.0) {
	
	SGRID_EoS_T0_rho0_P_rhoE_from_hm1(IDvars[idvar_q], &rho, &pre, &eps);
	
	// 3-velocity  v^i
	Real xmax = (xb>0)?xmax1:xmax2;
	
	// construct KV xi from Omega, ecc, rdot, xmax1-xmax2 
	Real xix = -Omega*yb + xb*rdotor; // CM is at (0,0,0) in bam 
	Real xiy =  Omega*(xb - ecc*xmax) + yb*rdotor;
	Real xiz =  zb*rdotor;

	// vI^i = VR^i + xi^i 
	Real vIx = IDvars[idvar_VRx] + xix;
	Real vIy = IDvars[idvar_VRy] + xiy;
	Real vIz = IDvars[idvar_VRz] + xiz;

	// Note: vI^i = u^i/u^0 in DNSdata,
	//       while matter_v^i = u^i/(alpha u^0) + beta^i / alpha
	//   ==> matter_v^i = (vI^i + beta^i)/alpha                    
	v_u_x = (vIx + IDvars[idvar_Bx])/IDvars[idvar_alpha];
	v_u_y = (vIy + IDvars[idvar_By])/IDvars[idvar_alpha];
	v_u_z = (vIz + IDvars[idvar_Bz])/IDvars[idvar_alpha];
      }

      // If scalars are on we should initialise to zero
      Real Y[MAX_SPECIES];
#if NSCALARS > 0
      for (int r=0;r<NSCALARS;r++) {
        pscalars->r(r,k,j,i) = 0.;
        pscalars->s(r,k,j,i) = 0.;
        Y[r] = 0.0;
      }
#endif

      // Deal with atmosphere
      //TODO this needs to be re-written with proper EOS call etc.
      //     together with above blocks of code on EOS call
#if USETM
      rho = (rho > rho_atm ? rho : 0.0);
      Real nb = rho/mb;
#if NSCALARS > 0
      for (int l=0; l<NSCALARS; ++l) {
        // Y[l] = (rho > rho_atm ? linear_interp(LORENE_EoS_Table->Y[l], LORENE_EoS_Table->data[tab_logrho], LORENE_EoS_Table->size, log(w_rho)) : Y_atm[l]);
      }
#endif
      pre = (rho > rho_atm ? peos->GetEOS().GetPressure(nb, T_atm, Y) : 0.0);
      phydro->temperature(k,j,i) = T_atm;
#else
      pre = k_adi*pow(w_rho,gamma_adi);
#endif

      // Lorentz factor
      const Real vsq = (
        2.0*(v_u_x * v_u_y * IDvars[idvar_gxy]  +
             v_u_x * v_u_z * IDvars[idvar_gxz]  +
             v_u_y * v_u_z * IDvars[idvar_gyz]  +
        v_u_x * v_u_x * IDvars[idvar_gxx]  +
        v_u_y * v_u_y * IDvars[idvar_gyy]  +
        v_u_z * v_u_z * IDvars[idvar_gzz]) 
      );

      const Real W = 1.0 / std::sqrt(1.0 - vsq);

      // Fill primitive storage
      phydro->w(IDN, k, j, i) = rho;
      phydro->w(IVX, k, j, i) = W * v_u_x;
      phydro->w(IVY, k, j, i) = W * v_u_y;
      phydro->w(IVZ, k, j, i) = W * v_u_z;
      phydro->w(IPR, k, j, i) = pre;

#if NSCALARS > 0
      for (int r=0;r<NSCALARS;r++) {
        pscalars->r(r,k,j,i) = Y[r];
      }
#endif
      
    } // k,j,i loop

  // Copy primitive stack
  phydro->w1 = phydro->w;
  
  //
  // Add magnetic field as needed
  //
  
  if (MAGNETIC_FIELDS_ENABLED)
    {
      // Assume stars are located on x axis
      
      const Real pgasmax = pin->GetReal("problem","pmax");
      const Real pcut = pin->GetReal("problem","pcut") * pgasmax;
      const Real b_amp = pin->GetReal("problem","b_amp");
      const int magindex = pin->GetInteger("problem","magindex");
      const Real sep = pin->GetReal("problem","DNSdata_b");
      
      const int nx1 = (ie-is)+1 + 2*(NGHOST); //TODO Shouldn't this be ncell[123]?
      const int nx2 = (je-js)+1 + 2*(NGHOST);
      const int nx3 = (ke-ks)+1 + 2*(NGHOST);
      
      pfield->b.x1f.ZeroClear();
      pfield->b.x2f.ZeroClear();
      pfield->b.x3f.ZeroClear();
      pfield->bcc.ZeroClear();
      
      AthenaArray<Real> bxcc,bycc,bzcc;
      bxcc.NewAthenaArray(nx3,nx2,nx1);
      bycc.NewAthenaArray(nx3,nx2,nx1);
      bzcc.NewAthenaArray(nx3,nx2,nx1);
      
      AthenaArray<Real> Atot;
      Atot.NewAthenaArray(3,nx3,nx2,nx1);
      
      for (int k = kl; k <= ku; ++k)
      for (int j = jl; j <= ju; ++j)
      for (int i = il; i <= iu; ++i)
	{
	  if(pcoord->x1v(i) > 0){
	    Atot(0,k,j,i) = -pcoord->x2v(j) * b_amp * std::max(phydro->w(IPR,k,j,i) - pcut, 0.0);
	    Atot(1,k,j,i) = (pcoord->x1v(i) - 0.5*sep) * b_amp * std::max(phydro->w(IPR,k,j,i) - pcut, 0.0);
	    Atot(2,k,j,i) = 0.0;
	  } else {
	    Atot(0,k,j,i) = -pcoord->x2v(j) * b_amp * std::max(phydro->w(IPR,k,j,i) - pcut, 0.0);
	    Atot(1,k,j,i) = (pcoord->x1v(i) + 0.5*sep) * b_amp * std::max(phydro->w(IPR,k,j,i) - pcut, 0.0);
	    Atot(2,k,j,i) = 0.0;
	  }
	}
      
      for(int k = ks-1; k<=ke+1; k++)
      for(int j = js-1; j<=je+1; j++)
      for(int i = is-1; i<=ie+1; i++)
	{

      bxcc(k,j,i) = - ((Atot(1,k+1,j,i) - Atot(1,k-1,j,i))/(2.0*pcoord->dx3v(k)));
      bycc(k,j,i) =  ((Atot(0,k+1,j,i) - Atot(0,k-1,j,i))/(2.0*pcoord->dx3v(k)));
      bzcc(k,j,i) = ( (Atot(1,k,j,i+1) - Atot(1,k,j,i-1))/(2.0*pcoord->dx1v(i))
                    - (Atot(0,k,j+1,i) - Atot(0,k,j-1,i))/(2.0*pcoord->dx2v(j)));
	}

      for(int k = ks; k<=ke; k++)
      for(int j = js; j<=je; j++)
      for(int i = is; i<=ie+1; i++)
	{

      pfield->b.x1f(k,j,i) = 0.5*(bxcc(k,j,i-1) + bxcc(k,j,i));
	}

      for(int k = ks; k<=ke; k++)
      for(int j = js; j<=je+1; j++)
      for(int i = is; i<=ie; i++)
	{
	  pfield->b.x2f(k,j,i) = 0.5*(bycc(k,j-1,i) + bycc(k,j,i));
	}

      for(int k = ks; k<=ke+1; k++)
      for(int j = js; j<=je; j++)
      for(int i = is; i<=ie; i++)
	{
	  pfield->b.x3f(k,j,i) = 0.5*(bzcc(k-1,j,i) + bzcc(k,j,i));
	}

      pfield->CalculateCellCenteredField(pfield->b, pfield->bcc, pcoord, il,iu,jl,ju,kl,ku);
    } // MAGNETIC_FIELDS_ENABLED
  
  
  //
  // Construct Z4c vars from ADM vars
  //
    
  pz4c->ADMToZ4c(pz4c->storage.adm, pz4c->storage.u);
  // pz4c->ADMToZ4c(pz4c->storage.adm, pz4c->storage.u1);
  
  //
  // Allow override of SGRID gauge
  //

  bool fix_gauge_precollapsed = pin->GetOrAddBoolean("problem", "fix_gauge_precollapsed", false);
  
  if (fix_gauge_precollapsed)
  {
    // to construct psi4
    pz4c->Z4cToADM(pz4c->storage.u, pz4c->storage.adm);
    pz4c->GaugePreCollapsedLapse(pz4c->storage.adm, pz4c->storage.u);
  }

  //
  // Consistent pressure atmosphere
  //
  
  bool id_floor_primitives = pin->GetOrAddBoolean(
    "problem", "id_floor_primitives", false);

  if (id_floor_primitives)
  {

    for (int k = 0; k <= ncells3-1; ++k)
    for (int j = 0; j <= ncells2-1; ++j)
    for (int i = 0; i <= ncells1-1; ++i)
    {
#if USETM
      peos->ApplyPrimitiveFloors(phydro->w, pscalars->r, k, j, i);
#else
      peos->ApplyPrimitiveFloors(phydro->w, k, j, i);
#endif
    }

  }

  //
  // Initialise conserved variables
  //
  
#if USETM
  peos->PrimitiveToConserved(phydro->w, pscalars->r, pfield->bcc, phydro->u, pscalars->s, pcoord,
                             0, ncells1-1,
                             0, ncells2-1,
                             0, ncells3-1);
#else
  peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord,
                             0, ncells1-1,
                             0, ncells2-1,
                             0, ncells3-1);
#endif
  //TODO Check if the momentum and velocity are finite.

  // Set up the matter tensor in the Z4c variables.
  // TODO: BD - this needs to be fixed properly
  // No magnetic field, pass dummy or fix with overload
  //  AthenaArray<Real> null_bb_cc;
#if USETM
  pz4c->GetMatter(pz4c->storage.mat, pz4c->storage.adm, phydro->w, pscalars->r, pfield->bcc);
#else
  pz4c->GetMatter(pz4c->storage.mat, pz4c->storage.adm, phydro->w, pfield->bcc);
#endif

  pz4c->ADMConstraints(pz4c->storage.con,
                       pz4c->storage.adm,
                       pz4c->storage.mat,
                       pz4c->storage.u);

  return;
}

//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop(ParameterInput *pin, int res_flag)
//  \brief Free SGRID memory as soon as possible
//========================================================================================
void Mesh::DeleteTemporaryUserMeshData()
{
  if (!resume_flag && SGRID_grid_exists()) {
    SGRID_free_everything();
  }
  return;
}



namespace {

//----------------------------------------------------------------------------------------
//! \fn void DNS_init_sgrid(ParameterInput *pin)
//  \brief Initialize libsgrid: alloc mem, build the command and read checkpoint
//  This code is adapted from W.Tichy NMESH example
void DNS_init_sgrid(ParameterInput *pin)
{
  const int level_l = 0; 
  const int myrank = Globals::my_rank;

  const bool verbose = pin->GetOrAddBoolean("problem", "verbose", 0);
  std::string const sgrid_datadir_const = pin->GetString("problem", "datadir");
  const bool keep_sgrid_output = pin->GetOrAddBoolean("problem", "keep_sgrid_output", 0);
  const bool Interpolate_verbose = pin->GetOrAddBoolean("problem", "Interpolate_verbose", 0);
  const bool Interpolate_make_finer_grid2 = pin->GetOrAddBoolean("problem", "Interpolate_make_finer_grid2", 0);
  const Real Interpolate_max_xyz_diff = pin->GetOrAddReal("problem", "Interpolate_max_xyz_diff",0.);
  std::string outdir = pin->GetOrAddString("problem", "outdir","SGRID");
  
  char * sgrid_datadir = (char *)malloc(sgrid_datadir_const.length() + 1);
  std::strcpy(sgrid_datadir, sgrid_datadir_const.c_str());

  char command[STRLEN+65676];
  char sgrid_exe[] = "sgrid"; // name is not important 
  char sgridoutdir[STRLEN], sgridoutdir_previous[STRLEN];
  char sgridcheckpoint_indir[STRLEN];
  char sgridparfile[STRLEN];
  char *stringptr;
  int ret;

  // initialize file names 
  //std::sprintf(gridfile, "%s/grid_level_%d_proc_%d.dat", outdir, level_l, MPIrank);
  std::sprintf(sgridoutdir, "%s/sgrid_level_%d_proc_%d", outdir.c_str(), level_l, myrank);
  std::sprintf(sgridoutdir_previous, "%s/sgrid_level_%d_proc_%d_previous",
          outdir.c_str(), level_l, myrank);
  std::snprintf(sgridcheckpoint_indir, STRLEN-1, "%s", sgrid_datadir);
  stringptr = std::strrchr(sgrid_datadir, '/'); // find last / 
  if(stringptr==NULL) { // no / found in DNSdataReader_sgrid_datadir 
    std::snprintf(sgridparfile, STRLEN-1, "%s.par", sgrid_datadir);
  } else {
    std::snprintf(sgridparfile, STRLEN-1, "%s%s", stringptr+1, ".par");
  }
  
  // IMPORTANT: Put sgrid in a mode where it does not free its memory before
  // returning from libsgrid_main. So later we need to explicitly call
  //   SGRID_free_everything();
  // Done in UserWorkAfterLoop
  SGRID_memory_persists = 1;

  // init sgrid if needed, so that we can call funcs in it 
  if(!SGRID_grid_exists())
  {
    if (verbose) std::printf("Init sgrid\n");
    
    // call sgrid without running interpolator 
    std::sprintf(command, "%s %s/%s "
            "--modify-par:BNSdata_Interpolate_pointsfile=%s "
            "--modify-par:BNSdata_Interpolate_output=%s "
            "--modify-par:outdir=%s "
            "--modify-par:checkpoint_indir=%s",
            sgrid_exe, sgrid_datadir, sgridparfile,
            "****NONE****", "<NONE>", sgridoutdir, sgridcheckpoint_indir);

    // low verbosity 
    std::strcat(command,
           " --modify-par:Coordinates_set_bfaces=no"
           " --modify-par:verbose=no"
           " --modify-par:Coordinates_verbose=no");

    // add other pars 
    if(Interpolate_verbose)
      std::strcat(command, " --modify-par:BNSdata_Interpolate_verbose=yes");
    if(Interpolate_max_xyz_diff>0.0)
    {
      char str[STRLEN];
      std::sprintf(str, " --modify-par:BNSdata_Interpolate_max_xyz_diff=%g",
              Interpolate_max_xyz_diff);
      std::strcat(command, str);
    }
    if(!Interpolate_make_finer_grid2)
      std::strcat(command, " --modify-par:BNSdata_Interpolate_make_finer_grid2_forXYZguess=no");
    if(!keep_sgrid_output)
      std::strcat(command, " > /dev/null");

    int ret = DNS_call_sgrid(command);
    if (verbose) std::printf("DNS_call_sgrid returned: %d\n", ret);
  }
  
}
  
//----------------------------------------------------------------------------------------
//! \fn int DNS_call_sgrid(const char *command)
//  \brief Utility for SGRID DNS files: call libsgrid
//  This code is minimally changed from W.Tichy NMESH example
int DNS_call_sgrid(const char *command)
{
  char *com = strdup(command); /* duplicate since construct_argv modifies its args */
  char **argv;
  int argc, status=-911;
  int size = Globals::nranks;
  int rkop;

  // cleanup in case we have called this already before 
  if(SGRID_grid_exists()) SGRID_free_everything();

  std::printf("calling libsgrid_main with these arguments:\n%s\n", command);

  //argc = construct_argv(com, &argv);
  argc = SGRID_construct_argv(com, &argv);
 
  status = libsgrid_main(argc, argv);

  if(status!=0) {
    std::printf("WARNING: Return value = %d\n", status); 
  }
  
  free(argv); // free since construct_argv allocates argv 
  free(com);

  return status;
}
  
//----------------------------------------------------------------------------------------
//! \fn int DNS_parameters(ParameterInput *pin)
//  \brief Utility for SGRID DNS files: read SGRID BNSdata_properties.txt and get pars 
//  This code is minimally changed from W.Tichy NMESH example
int DNS_parameters(ParameterInput *pin)
{
  bool verbose = pin->GetOrAddBoolean("problem", "verbose", 0);

  FILE *fp1;
  char str[STRLEN];
  char strn[STRLEN], strrho0[STRLEN], strkappa[STRLEN];
  char EoS_type[STRLEN], EoS_file[STRLEN];
  char datadir[STRLEN];

  // put empty string in some strings
  strn[0] = strrho0[0] = strkappa[0] = EoS_type[0] = EoS_file[0] = 0;

  // Get datadir and remove any trailing "/"
  std::snprintf(datadir, STRLEN-1, "%s",
	   pin->GetString("problem", "datadir").c_str());
  int j = strlen(datadir);
  if(datadir[j-1]=='/')
  {
    datadir[j-1]=0;
    pin->SetString("problem", "datadir", datadir);
  }
  std::strcat(datadir, "/BNSdata_properties.txt");

  //
  // Open file
  //
  
  fp1 = fopen(datadir, "r");
  if(fp1==NULL) {
    std::stringstream msg;
    msg << "### FATAL ERROR datadir: " << datadir << " "
        << " could not be accessed.";
    ATHENA_ERROR(msg);
  }
  
  // move fp1 to place where time = 0 is 
  j = DNS_position_fileptr_after_str(fp1, "NS data properties (time = 0):\n");
  if(j==EOF) {
    std::stringstream msg;
    msg << "### FATAL ERROR could not find (time = 0) in: " << datadir;
    ATHENA_ERROR(msg);
  }

  //
  // Get SGRID pars
  //

  // EOS
  Real ret = SGRID_fgetparameter(fp1, "EoS_type", EoS_type);
  if(ret==EOF)
  {
    // if we can't find EoS_type default to PwP 
    std::sprintf(EoS_type, "%s", "PwP");
    rewind(fp1);
    j = DNS_position_fileptr_after_str(fp1, "NS data properties (time = 0):\n");
    if (verbose) std::printf("Cannot find EoS, use default ...\n");
  }
  if (verbose) std::printf("EoS_type = %s\n", EoS_type);

  // Check if we need to read piecewise poly (PwP) pars 
  if( (strcmp(EoS_type,"PwP")==0) || (strcmp(EoS_type,"pwp")==0) )
  {
    SGRID_fgotonext(fp1, "n_list");
    SGRID_fscanline(fp1, strn);
    for(j=0; strn[j]==' ' || strn[j]=='\t'; j++) ;
    if(j) std::memmove(strn, strn+j, strlen(strn)+1);

    SGRID_fgotonext(fp1, "rho0_list");
    SGRID_fscanline(fp1, strrho0);
    for(j=0; strrho0[j]==' ' || strrho0[j]=='\t'; j++) ;
    if(j) std::memmove(strrho0, strrho0+j, strlen(strrho0)+1);

    SGRID_fgetparameter(fp1, "kappa", strkappa);

    if (verbose) {
      std::printf("initial data uses PwP EoS with:\n");
      std::printf("n_list    = %s\n", strn);
      std::printf("rho0_list = %s\n", strrho0);
      std::printf("kappa     = %s\n", strkappa);
      std::printf("Note: n_list contains the polytropic indices n,\n"
	     "      compute each Gamma using:  Gamma = 1 + 1/n\n");
    }
  }
  
  // Check if EoS is in sgrid table
  if(strcmp(EoS_type,"tab1d_AtT0")==0)
  {
    SGRID_fgetparameter(fp1, "EoS_file", EoS_file);
    if (verbose) {
      std::printf("initial data uses T=0 EoS table:\n");
      std::printf("EoS_file = %s\n", EoS_file);
    }
  }

  //TODO set/adapt Athena++ EOS parameters from SGRID data 
  /*
    Real rho0max, epslmax, Pmax;

    set some Nmesh EoS pars according to what was read 
    if(strrho0[0])  Sets(Par("EoS_PwP_rho0"), strrho0);
    if(strn[0])     Sets(Par("EoS_PwP_n"), strn);
    if(strkappa[0]) Sets(Par("EoS_PwP_kappa"), strkappa);
    if(EoS_file[0]) Sets(Par("EoS_tab1d_load_file"), EoS_file);

    std::printf("NOTE: Some nmesh pars have been set:\n");
    if(EoS_type[0]) printf("EoS_type = %s\n", Gets(Par("EoS_type")));
    if(strrho0[0])  printf("EoS_PwP_rho0 = %s\n", Gets(Par("EoS_PwP_rho0")));
    if(strn[0])     printf("EoS_PwP_n = %s\n", Gets(Par("EoS_PwP_n")));
    if(strkappa[0]) printf("EoS_PwP_kappa = %s\n", Gets(Par("EoS_PwP_kappa")));
    if(EoS_file[0]) printf("EoS_tab1d_load_file = %s\n", Gets(Par("EoS_tab1d_load_file")));
    printf("Make sure PwP_init_from_parameters gives a compatible EoS!!!\n");
  
    EoS_reinit_from_pars(mesh);

    qmax = fmax(qmax1, qmax2);
    if( (strcmp(EoS_type,"PwP")==0) || (strcmp(EoS_type,"pwp")==0) )
    {
    Real rhoEmax, drho0dhm1;
    PwP_polytrope_of_hm1(qmax, &rho0max, &Pmax, &rhoEmax, &drho0dhm1);
    epslmax = (rhoEmax-rho0max)/rho0max; //rhoE=(1+epsl)rho0
    }
    else if(strcmp(EoS_type,"tab1d_AtT0")==0)
    {
    Real dPdrho0, dPdepsl;
    tab1d_Of_hm1_AtT0(qmax, &rho0max, &epslmax, &Pmax, &dPdrho0, &dPdepsl);
    }
    else // read rho0max, Pmax from file fp1 
    {
    Real rho0max1=-1, Pmax1=-1, rho0max2=-1, Pmax2=-1;
    rewind(fp1);
    j=DNS_position_fileptr_after_str(fp1, "NS data properties (time = 0):\n");
    
    ret = SGRID_fgetparameter(fp1, "rho0max1", str);
    if(ret!=EOF) rho0max1 = atof(str);
    ret = SGRID_fgetparameter(fp1, "Pmax1", str);
    if(ret!=EOF) Pmax1 = atof(str);
    
    ret = SGRID_fgetparameter(fp1, "rho0max2", str);
    if(ret!=EOF) rho0max2 = atof(str);
    ret = SGRID_fgetparameter(fp1, "Pmax2", str);
    if(ret!=EOF) Pmax2 = atof(str);
    
    rho0max = fmax(rho0max1, rho0max2);
    Pmax    = fmax(Pmax1, Pmax2);
    if(rho0max<0. || Pmax<0.)
    errorexit("unable to find rho0max1/2 and Pmax1/2 in "
    "BNSdata_properties.txt");
    
    // Set epslmax using: q = h-1 = epsl + P/rho0  =>  epsl = q - P/rho0 
    epslmax = qmax - Pmax/rho0max;
    }
  */


  // Other parameters
  SGRID_fgetparameter(fp1, "x_CM", str);
  Real sgrid_x_CM = atof(str);
  SGRID_fgetparameter(fp1, "Omega", str);
  Real Omega = atof(str);
  SGRID_fgetparameter(fp1, "ecc", str);
  Real ecc = atof(str);
  SGRID_fgetparameter(fp1, "rdot", str);
  Real rdot = atof(str);
  SGRID_fgetparameter(fp1, "m01", str);
  Real m01 = atof(str);
  SGRID_fgetparameter(fp1, "m02", str);
  Real m02 = atof(str);

  // Shift xmax1/2 such that CM is at 0, also read qmax1/2 
  SGRID_fgetparameter(fp1, "xin1", str);
  Real xin1 = atof(str)-sgrid_x_CM;  
  SGRID_fgetparameter(fp1, "xmax1", str);
  Real xmax1 = atof(str)-sgrid_x_CM;
  SGRID_fgetparameter(fp1, "xout1", str);
  Real xout1 = atof(str)-sgrid_x_CM;
  SGRID_fgetparameter(fp1, "qmax1", str);
  Real qmax1 = atof(str);
  SGRID_fgetparameter(fp1, "xin2", str);
  Real xin2 = atof(str)-sgrid_x_CM;
  SGRID_fgetparameter(fp1, "xmax2", str);
  Real xmax2 = atof(str)-sgrid_x_CM;
  SGRID_fgetparameter(fp1, "xout2", str);
  Real xout2 = atof(str)-sgrid_x_CM;
  SGRID_fgetparameter(fp1, "qmax2", str);
  Real qmax2 = atof(str);
  SGRID_fgetparameter(fp1, "DNSdata_b", str);
  Real sep = atof(str);


  //
  // Set Athena++ parameters for later
  //
  
  pin->SetReal("problem", "x_CM" , sgrid_x_CM);
  pin->SetReal("problem", "Omega", Omega);
  pin->SetReal("problem", "ecc"  , ecc);
  pin->SetReal("problem", "rdot" , rdot);
  pin->SetReal("problem", "m01"  , m01);
  pin->SetReal("problem", "m02"  , m02);
  
  pin->SetReal("problem", "xin1" , xin1);
  pin->SetReal("problem", "xmax1", xmax1);
  pin->SetReal("problem", "qmax1", qmax1);
  pin->SetReal("problem", "xin2" , xin2);
  pin->SetReal("problem", "xmax2", xmax2);
  pin->SetReal("problem", "xout2", xout2);
  pin->SetReal("problem", "qmax2", qmax2);
  pin->SetReal("problem", "DNSdata_b", sep);

  pin->SetReal("problem", "center1_mass", m01);
  pin->SetReal("problem", "center2_mass", m02);
  pin->SetReal("problem", "center0_x", 0.);
  pin->SetReal("problem", "center0_y", 0.);
  pin->SetReal("problem", "center0_z", 0.);
  pin->SetReal("problem", "center1_x", xmax1);
  pin->SetReal("problem", "center1_y", 0.);
  pin->SetReal("problem", "center1_z", 0.);
  pin->SetReal("problem", "center2_x", xmax2);
  pin->SetReal("problem", "center2_y", 0.);
  pin->SetReal("problem", "center2_z", 0.);

  //
  // Close file 
  //
  
  fclose(fp1);

  if (verbose) {
    printf("Done with reading SGRID parameters:\n");
    printf("Omega = %g\n", Omega);
    printf("ecc = %g\n", ecc);
    printf("rdot = %g\n", rdot);
    printf("m01 = %g\n", m01);
    printf("m02 = %g\n", m02);
    printf("sgrid_x_CM = %g\n", sgrid_x_CM);
    printf("xmax1 - sgrid_x_CM = %g\n", xmax1);
    printf("xmax2 - sgrid_x_CM = %g\n", xmax2);
    printf("Make sure to center the mesh on the latter two!!!\n");
  }

  return 0;
}

//----------------------------------------------------------------------------------------
//! \fn int DNS_position_fileptr_after_str(FILE *in, const char *str)
//  \brief Utility for SGRID DNS files: position filepointer after the string str
//  This code is minimally changed from W.Tichy NMESH example
int DNS_position_fileptr_after_str(FILE *in, const char *str)
{
  char line[STRLEN];
  while(fgets(line, STRLEN-1, in)!=NULL)
  {
    if(strstr(line, str)!=NULL) return 1; //break;
  }
  return EOF;
}
  
//----------------------------------------------------------------------------------------
//! \fn
//  \brief refinement condition: extrema based
// 1: refines, -1: de-refines, 0: does nothing
int RefinementCondition(MeshBlock *pmb)
{
  Mesh * pmesh = pmb->pmy_mesh;
  ExtremaTracker * ptracker_extrema = pmesh->ptracker_extrema;

  int root_level = ptracker_extrema->root_level;
  int mb_physical_level = pmb->loc.level - root_level;

  // Iterate over refinement levels offered by trackers.
  //
  // By default if a point is not in any sphere, completely de-refine.
  int req_level = 0;

  for (int n=1; n<=ptracker_extrema->N_tracker; ++n)
  {
    bool is_contained = false;
    int cur_req_level = ptracker_extrema->ref_level(n-1);

    {
      if (ptracker_extrema->ref_type(n-1) == 0)
      {
        is_contained = pmb->PointContained(
          ptracker_extrema->c_x1(n-1),
          ptracker_extrema->c_x2(n-1),
          ptracker_extrema->c_x3(n-1)
        );
      }
      else if (ptracker_extrema->ref_type(n-1) == 1)
      {
        is_contained = pmb->SphereIntersects(
          ptracker_extrema->c_x1(n-1),
          ptracker_extrema->c_x2(n-1),
          ptracker_extrema->c_x3(n-1),
          ptracker_extrema->ref_zone_radius(n-1)
        );
      }
    }

    if (is_contained)
    {
      req_level = std::max(cur_req_level, req_level);
    }

  }

  if (req_level > mb_physical_level)
  {
    return 1;  // currently too coarse, refine
  }
  else if (req_level == mb_physical_level)
  {
    return 0;  // level satisfied, do nothing
  }

  // otherwise de-refine
  return -1;

}

#if MAGNETIC_FIELDS_ENABLED
Real DivBface(MeshBlock *pmb, int iout) {
  Real divB = 0.0;
  Real vol,dx,dy,dz;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        dx = pmb->pcoord->dx1v(i);
        dy = pmb->pcoord->dx2v(j);
        dz = pmb->pcoord->dx3v(k);
        vol = dx*dy*dz;
        divB += ((pmb->pfield->b.x1f(k,j,i+1) - pmb->pfield->b.x1f(k,j,i))/ dx +
                 (pmb->pfield->b.x2f(k,j+1,i) - pmb->pfield->b.x2f(k,j,i))/ dy +
                 (pmb->pfield->b.x3f(k+1,j,i) - pmb->pfield->b.x3f(k,j,i))/ dz) * vol;
      }
    }
  }
  return divB;
}
#endif

Real max_rho(MeshBlock *pmb, int iout)
{
  Real max_rho = -std::numeric_limits<Real>::infinity();
  int is = pmb->is, ie = pmb->ie;
  int js = pmb->js, je = pmb->je;
  int ks = pmb->ks, ke = pmb->ke;

  AthenaArray<Real> &w = pmb->phydro->w;
  for (int k=ks; k<=ke; k++)
  for (int j=js; j<=je; j++)
  for (int i=is; i<=ie; i++)
  {
    max_rho = std::max(std::abs(w(IDN,k,j,i)), max_rho);
  }

  return max_rho;
}

Real min_alpha(MeshBlock *pmb, int iout)
{

  const int N = NDIM;

  typedef AthenaTensor<Real, TensorSymm::NONE, N, 0> AT_N_sca;
  typedef AthenaTensor<Real, TensorSymm::NONE, N, 1> AT_N_vec;
  typedef AthenaTensor<Real, TensorSymm::SYM2, N, 2> AT_N_sym;

  // --------------------------------------------------------------------------
  // Set some aliases for the variables.
  AT_N_sca alpha(pmb->pz4c->storage.u, Z4c::I_Z4c_alpha);

  // container with idx / grids pertaining z4c
  MB_info* mbi = &(pmb->pz4c->mbi);

  Real m_alpha = std::numeric_limits<Real>::infinity();

  for (int k=mbi->kl; k<=mbi->ku; k++)
  for (int j=mbi->jl; j<=mbi->ju; j++)
  for (int i=mbi->il; i<=mbi->iu; i++)
  {
    m_alpha = std::min(alpha(k,j,i), m_alpha);
  }

  return m_alpha;
}

Real max_abs_con_H(MeshBlock *pmb, int iout)
{

  const int N = NDIM;

  typedef AthenaTensor<Real, TensorSymm::NONE, N, 0> AT_N_sca;
  typedef AthenaTensor<Real, TensorSymm::NONE, N, 1> AT_N_vec;
  typedef AthenaTensor<Real, TensorSymm::SYM2, N, 2> AT_N_sym;

  // --------------------------------------------------------------------------
  // Set some aliases for the variables.
  AT_N_sca con_H(pmb->pz4c->storage.con, Z4c::I_CON_H);

  // container with idx / grids pertaining z4c
  MB_info* mbi = &(pmb->pz4c->mbi);

  Real m_abs_con_H = -std::numeric_limits<Real>::infinity();

  for (int k=mbi->kl; k<=mbi->ku; k++)
  for (int j=mbi->jl; j<=mbi->ju; j++)
  for (int i=mbi->il; i<=mbi->iu; i++)
  {
    m_abs_con_H = std::max(std::abs(con_H(k,j,i)), m_abs_con_H);
  }

  return m_abs_con_H;
}

//TODO ... following is LORENE_EOS block in gr_Lorene_bns.cpp, might need something similar
#if (0) 
  //--------------------------------------------------------------------------------------
  //! \fn Real linear_interp(Real *f, Real *x, int n, Real xv)
  // \brief linearly interpolate f(x), compute f(xv)
  Real linear_interp(Real *f, Real *x, int n, Real xv)
  {
    int i = interp_locate(x,n,xv);
    if (i < 0)  i=1;
    if (i == n) i=n-1;
    int j;
    if(xv < x[i]) j = i-1;
    else j = i+1;
    Real xj = x[j]; Real xi = x[i];
    Real fj = f[j]; Real fi = f[i];
    Real m = (fj-fi)/(xj-xi);
    Real df = m*(xv-xj)+fj;
    return df;
  }

  //-----------------------------------------------------------------------------------------
  //! \fn int interp_locate(Real *x, int Nx, Real xval)
  // \brief Bisection to find closest point in interpolating table
  // 
  int interp_locate(Real *x, int Nx, Real xval) {
    int ju,jm,jl;
    int ascnd;
    jl=-1;
    ju=Nx;
    if (xval <= x[0]) {
      return 0;
    } else if (xval >= x[Nx-1]) {
      return Nx-1;
    }
    ascnd = (x[Nx-1] >= x[0]);
    while (ju-jl > 1) {
      jm = (ju+jl) >> 1;
      if (xval >= x[jm] == ascnd)
  jl=jm;
      else
  ju=jm;
    }
    return jl;
  }
#endif
}
