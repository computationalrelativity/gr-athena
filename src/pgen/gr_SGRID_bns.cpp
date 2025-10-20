//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//! \file gr_Lorene_BinNSs.cpp
//  \brief Initial conditions for binary neutron stars.
//         Interpolation of Lorene initial data.
//         Requires the library:
//         https://lorene.obspm.fr/

#include <algorithm>
#include <cstring> // strcmp()
#include <cassert>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <streambuf>
#include <cmath>
#include <filesystem>

// Athena++ headers
#include "../globals.hpp"
#include "../athena_aliases.hpp"
#include "../parameter_input.hpp"
#include "../coordinates/coordinates.hpp"
#include "../z4c/ahf.hpp"
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

#if M1_ENABLED
#include "../m1/m1.hpp"
#include "../m1/m1_set_equilibrium.hpp"
#endif  // M1_ENABLED


//----------------------------------------------------------------------------------------
using namespace gra::aliases;
#if USETM
using namespace Primitive;
#endif
//----------------------------------------------------------------------------------------

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

#if USETM
  // Global variables
  ColdEOS<COLDEOS_POLICY> * ceos = NULL;
#else
  Real k_adi;
  Real gamma_adi;
#endif

  Real sep;
  Real pgasmax_1;
  Real pgasmax_2;

  // constants ----------------------------------------------------------------
  Real const B_unit = 8.351416e19; // almost the same as athenaB * 1.0e4;
  // --------------------------------------------------------------------------
  
  // Utilities wrapping various SGRID DNS calls (DNS_*)
  void DNS_init_sgrid(ParameterInput *pin);
  int DNS_position_fileptr_after_str(FILE *in, const char *str);
  int DNS_parameters(ParameterInput *pin);
  int DNS_call_sgrid(const char *command);
}

namespace {

void SeedMagneticFields(MeshBlock *pmb, ParameterInput *pin)
{
  GRDynamical * pcoord { static_cast<GRDynamical*>(pmb->pcoord) };
  Field * pfield { pmb->pfield };
  Hydro * phydro { pmb->phydro };

  // Prepare CC index bounds
  const int il = 0;
  const int iu = (pmb->ncells1>1)? pmb->ncells1-1: 0;

  const int jl = 0;
  const int ju = (pmb->ncells2>1)? pmb->ncells2-1: 0;

  const int kl = 0;
  const int ku = (pmb->ncells3>1)? pmb->ncells3-1: 0;


  // B field ------------------------------------------------------------------
  // Assume stars are located on x axis

  Real pcut_1 = pin->GetReal("problem","pcut_1") * pgasmax_1;
  Real pcut_2 = pin->GetReal("problem","pcut_2") * pgasmax_2;

  // Real b_amp = pin->GetReal("problem","b_amp");
  // Scaling taken from project_bnsmhd
  Real ns_1 = pin->GetReal("problem","ns_1");
  Real ns_2 = pin->GetReal("problem","ns_2");

  // Read b_amp and rescale it from gaus to code units
  Real A_amp_1 = pin->GetReal("problem","b_amp_1") *
    0.5/std::pow(pgasmax_1-pcut_1, ns_1)/B_unit;
  Real A_amp_2 = pin->GetReal("problem","b_amp_2") *
    0.5/std::pow(pgasmax_2-pcut_2, ns_2)/B_unit;

  pfield->b.x1f.ZeroClear();
  pfield->b.x2f.ZeroClear();
  pfield->b.x3f.ZeroClear();
  pfield->bcc.ZeroClear();

  AthenaArray<Real> Acc(NFIELD,pmb->ncells3,pmb->ncells2,pmb->ncells1);

  // Initialize cell centred potential
  for (int k=0; k<pmb->ncells3; k++)
  for (int j=0; j<pmb->ncells2; j++)
  for (int i=0; i<pmb->ncells1; i++)
  {
    const Real x1 = pcoord->x1v(i);
    const Real x2 = pcoord->x2v(j);

    const Real w_p   = phydro->w(IPR,k,j,i);
    const Real w_rho = phydro->w(IDN,k,j,i);

    if(x1 > 0)
    {
      Real A_amp =
          A_amp_2 * std::max(std::pow(w_p - pcut_2, ns_2), 0.0);
      Acc(0,k,j,i) = -x2 * A_amp;
      Acc(1,k,j,i) = (x1 - sep) * A_amp;
      Acc(2,k,j,i) = 0.0;
    }
    else
    {
      Real A_amp =
          A_amp_1 * std::max(std::pow(w_p - pcut_1, ns_1), 0.0);
      Acc(0,k,j,i) = -x2 * A_amp;
      Acc(1,k,j,i) = (x1 + sep) * A_amp;
      Acc(2,k,j,i) = 0.0;
    }
  }

  // Construct cell centred B field from cell centred potential
  for(int k=pmb->ks-1; k<=pmb->ke+1; k++)
  for(int j=pmb->js-1; j<=pmb->je+1; j++)
  for(int i=pmb->is-1; i<=pmb->ie+1; i++)
  {
    const Real dx1 = pcoord->dx1v(i);
    const Real dx2 = pcoord->dx2v(j);
    const Real dx3 = pcoord->dx3v(k);

    pfield->bcc(0,k,j,i) = -((Acc(1,k+1,j,i) - Acc(1,k-1,j,i))/(2.0*dx3));
    pfield->bcc(1,k,j,i) =  ((Acc(0,k+1,j,i) - Acc(0,k-1,j,i))/(2.0*dx3));
    pfield->bcc(2,k,j,i) =  ((Acc(1,k,j,i+1) - Acc(1,k,j,i-1))/(2.0*dx1) -
                             (Acc(0,k,j+1,i) - Acc(0,k,j-1,i))/(2.0*dx2));

  }

  // Initialise face centred field by averaging cc field
  for(int k=pmb->ks; k<=pmb->ke;   k++)
  for(int j=pmb->js; j<=pmb->je;   j++)
  for(int i=pmb->is; i<=pmb->ie+1; i++)
  {
  	pfield->b.x1f(k,j,i) = 0.5*(pfield->bcc(0,k,j,i-1) +
                                pfield->bcc(0,k,j,i));
  }

  for(int k=pmb->ks; k<=pmb->ke;   k++)
  for(int j=pmb->js; j<=pmb->je+1; j++)
  for(int i=pmb->is; i<=pmb->ie;   i++)
  {
  	pfield->b.x2f(k,j,i) = 0.5*(pfield->bcc(1,k,j-1,i) +
                                pfield->bcc(1,k,j,i));
  }

  for(int k=pmb->ks; k<=pmb->ke+1; k++)
  for(int j=pmb->js; j<=pmb->je;   j++)
  for(int i=pmb->is; i<=pmb->ie;   i++)
  {
	  pfield->b.x3f(k,j,i) = 0.5*(pfield->bcc(2,k-1,j,i) +
                                pfield->bcc(2,k,j,i));
  }

}

} // namespace

namespace fs = std::filesystem;

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin)
{
  if(adaptive==true)
    EnrollUserRefinementCondition(RefinementCondition);

  EnrollUserStandardHydro(pin);
  EnrollUserStandardField(pin);
  EnrollUserStandardZ4c(pin);
  EnrollUserStandardM1(pin);

  /*
  // New outputs can now be specified with the form:
  EnrollUserHistoryOutput(
    [&](MeshBlock *pmb, int iout){ return 1.0; },
    "some_name",
    UserHistoryOperation::min
  );
  */

  if (resume_flag)
    return;

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

  sep = pin->GetReal("problem", "DNSdata_b");
  const Real sgrid_x_CM = pin->GetReal("problem", "x_CM");

#if USETM
  // initialize the cold EOS
  ceos = new ColdEOS<COLDEOS_POLICY>();
  InitColdEOS(ceos, pin);
#else
  k_adi = pin->GetReal("hydro", "k_adi");
  gamma_adi = pin->GetReal("hydro", "gamma");
#endif

  // read it in again to get the central densities
  Real xyz1[3] = {sep + sgrid_x_CM, 0.0, 0.0};
  Real xyz2[3] = {-sep + sgrid_x_CM, 0.0, 0.0};
  Real IDvars[idvar_NDATAMAX];

  Real rho_1 = 0.0;
  Real pre_1 = 0.0;
  Real eps_1 = 0.0;
  SGRID_DNSdata_Interpolate_ADMvars_to_xyz(xyz1, IDvars, 0);
  SGRID_EoS_T0_rho0_P_rhoE_from_hm1(IDvars[idvar_q], &rho_1, &pre_1, &eps_1);

  Real rho_2 = 0.0;
  Real pre_2 = 0.0;
  Real eps_2 = 0.0;
  SGRID_DNSdata_Interpolate_ADMvars_to_xyz(xyz2, IDvars, 0);
  SGRID_EoS_T0_rho0_P_rhoE_from_hm1(IDvars[idvar_q], &rho_2, &pre_2, &eps_2);


  // for tabulated EOS need to convert baryon mass
//#if defined(USE_COMPOSE_EOS) || defined(USE_HYBRID_EOS)
//  Real rho_1 = bns->nbar[0] / m_u_si * 1e-45 * ceos->GetBaryonMass();
//  Real rho_2 = bns->nbar[1] / m_u_si * 1e-45 * ceos->GetBaryonMass();
//#endif

#if USETM
  pgasmax_1 = ceos->GetPressure(rho_1);
  pgasmax_2 = ceos->GetPressure(rho_2);
#else
  pgasmax_1 = k_adi * pow(rho_1, gamma_adi);
  pgasmax_2 = k_adi * pow(rho_2, gamma_adi);
#endif

  // sanity check if the internal energy matches the eos
//#if defined(USE_COMPOSE_EOS)  || defined(USE_TABULATED_EOS)
//  eps_1 = m_u_mev/ceos->mb * (eps_1 + 1) - 1; // convert eos baryon mass
//#endif

#if USETM
  Real eps_ceos = ceos->GetSpecificInternalEnergy(rho_1);
#else
  Real eps_ceos = k_adi * pow(w_rho, gamma_adi -1 )/(gamma_adi - 1);
#endif
  Real eps_err = std::abs(eps_ceos/eps_1 - 1);

#ifdef MPI_PARALLEL
  int rank;
  int root = pin->GetOrAddInteger("problem", "mpi_root", 0);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  bool ioproc = (root == rank);
#else
  bool ioproc = true;
#endif

  if (ioproc && (eps_err > 1.0e-5))
  {
    printf("Warning: Internal energy in SGRID data and eos do not match "
           "in the center of star 1!\n");
    printf("rho=%.16e, eps_lorene=%.16e, eps_eos=%.16e, rel. err.=%.16e\n",
           rho_1, eps_1, eps_ceos, eps_err);
  }

  return;
}

// BD: TODO- shift to standard enroll?
void MeshBlock::InitUserMeshBlockData(ParameterInput *pin)
{
  const bool use_fb = precon->xorder_use_fb;
  AllocateUserOutputVariables(use_fb + M1_ENABLED * 4);
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin)
{
  MeshBlock * pmb = this;

  const bool use_fb = precon->xorder_use_fb;

  if (use_fb)
  CC_GLOOP3(k, j, i)
  {
    user_out_var(0,k,j,i) = phydro->fallback_mask(k,j,i);
  }
}

void MeshBlock::UserWorkAfterOutput(ParameterInput *pin) {
  // Reset the status
  AA c2p_status;
  c2p_status.InitWithShallowSlice(phydro->derived_ms, IX_C2P, 1);
  c2p_status.Fill(0);
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Sets the initial conditions.
//========================================================================================
void MeshBlock::ProblemGenerator(ParameterInput *pin)
{
  using namespace LinearAlgebra;

  // Interpolate Lorene data onto the grid.

  // settings -----------------------------------------------------------------
  bool verbose = pin->GetOrAddBoolean("problem", "verbose", false);

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

  // --------------------------------------------------------------------------
  // Set some aliases for the variables.
  AT_N_sca alpha( pz4c->storage.adm, Z4c::I_ADM_alpha);
  AT_N_vec beta_u(pz4c->storage.adm, Z4c::I_ADM_betax);
  AT_N_sym g_dd(  pz4c->storage.adm, Z4c::I_ADM_gxx);
  AT_N_sym K_dd(  pz4c->storage.adm, Z4c::I_ADM_Kxx);


  // matter grid idx limits ---------------------------------------------------
  const int il = 0;
  const int iu = ncells1-1;

  const int jl = 0;
  const int ju = ncells2-1;

  const int kl = 0;
  const int ku = ncells3-1;


  // --------------------------------------------------------------------------
  #pragma omp critical
  {

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
      const Real det = Det3Metric(g_dd,k,j,i);
      assert(std::fabs(det) > tol_det_zero);

    }
    // ------------------------------------------------------------------------
    // Interpolate on matter grid 
    //
    
    AthenaArray<Real> & w = phydro->w;
#if NSCALARS > 0
    AthenaArray<Real> & r = pscalars->r;
    AthenaArray<Real> & s = pscalars->s;
    r.Fill(0.0);
    s.Fill(0.0);
#endif

    for (int k=kl; k<=ku; ++k)
    for (int j=jl; j<=ju; ++j)
    for (int i=il; i<=iu; ++i)
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
      w(IDN, k, j, i) = rho;
      w(IVX, k, j, i) = W * v_u_x;
      w(IVY, k, j, i) = W * v_u_y;
      w(IVZ, k, j, i) = W * v_u_z;
      w(IPR, k, j, i) = 0.0;

  }

  } // OMP Critical

  // --------------------------------------------------------------------------

  // Treat EOS derived quantities ---------------------------------------------
  {
    // Split into two blocks:
    // PrimitiveSolver (useful for physics) & Reprimand (useful for debug)

    AthenaArray<Real> & w  = phydro->w;
#if NSCALARS > 0
    AthenaArray<Real> & r = pscalars->r;
    r.Fill(0.0);
#endif

#if !USETM
    // Reprimand --------------------------------------------------------------
    // Reprimand fill
    for (int k=kl; k<=ku; ++k)
    for (int j=jl; j<=ju; ++j)
    for (int i=il; i<=iu; ++i)
    {
      w(IPR,k,j,i) = k_adi*std::pow(w(IDN,k,j,i),gamma_adi);
    }

#else
    // PrimitiveSolver --------------------------------------------------------
    Real w_rho_atm = pin->GetReal("hydro", "dfloor");
    Real rho_cut = std::max(pin->GetOrAddReal("problem", "rho_cut", w_rho_atm),
                            w_rho_atm);

#if NSCALARS > 0
    Real Y_atm[NSCALARS] = {0.0};
    for (int iy=0; iy<NSCALARS; ++iy)
    {
      Y_atm[iy] = pin->GetReal("hydro", "y" + std::to_string(iy) + "_atmosphere");
    }
#endif

    // USETM fill
    for (int k=kl; k<=ku; ++k)
    for (int j=jl; j<=ju; ++j)
    for (int i=il; i<=iu; ++i)
    {
      // Check if density admissible first -
      // This controls velocity reset & Y interpolation (if applicable)
      if (w(IDN,k,j,i) > rho_cut)
      {
        w(IPR,k,j,i) = ceos->GetPressure(w(IDN,k,j,i));

#if NSCALARS > 0
        for (int iy=0; iy<NSCALARS; ++iy)
          r(iy,k,j,i) = ceos->GetY(w(IDN,k,j,i), iy);
#endif
      }
      else
      {
        // Reset primitives
        w(IPR,k,j,i) = 0;

#if NSCALARS > 0
        for (int iy=0; iy<NSCALARS; ++iy)
          r(iy,k,j,i) = Y_atm[iy];
#endif

        // Assume that we always have (IVX, IVY, IVZ)
        for (int ix=0; ix<3; ++ix)
          w(IVX+ix,k,j,i) = 0;
      }
    }

    // ------------------------------------------------------------------------
#endif // !USETM
  }
  // --------------------------------------------------------------------------

  // --------------------------------------------------------------------------
#if MAGNETIC_FIELDS_ENABLED
  // Regularize prims
  for (int k=0; k<ncells3; k++)
  for (int j=0; j<ncells2; j++)
  for (int i=0; i<ncells1; i++)
  {
    for (int n=0; n<NHYDRO; ++n)
    if (!std::isfinite(phydro->w(n,k,j,i)))
    {
#if USETM
      peos->ApplyPrimitiveFloors(phydro->w, pscalars->r, k, j, i);
#else
      peos->ApplyPrimitiveFloors(phydro->w, k, j, i);
#endif
      continue;
    }
  }

  SeedMagneticFields(this, pin);
#endif
  //  -------------------------------------------------------------------------

  // Construct Z4c vars from ADM vars ------------------------------------------
  pz4c->ADMToZ4c(pz4c->storage.adm, pz4c->storage.u);
  // pz4c->ADMToZ4c(pz4c->storage.adm, pz4c->storage.u1);

  // Allow override of Lorene gauge -------------------------------------------
  bool fix_gauge_precollapsed = pin->GetOrAddBoolean(
    "problem", "fix_gauge_precollapsed", false);

  if (fix_gauge_precollapsed)
  {
    // to construct psi4
    pz4c->Z4cToADM(pz4c->storage.u, pz4c->storage.adm);
    pz4c->GaugePreCollapsedLapse(pz4c->storage.adm, pz4c->storage.u);
  }
  // --------------------------------------------------------------------------

  // Have geom & primitive hydro
  /*
#if M1_ENABLED
  // Mesh::Initialize calls FinalizeM1 which contains the following 3 lines;
  // We need it here if we want to equilibriate @ ID
  //
  // Note that a call of ConservedToPrimitive should be made prior to this
  // That is required to populate auxiliary vars. (not currently done)
  pm1->UpdateGeometry(pm1->geom, pm1->scratch);
  pm1->UpdateHydro(pm1->hydro, pm1->geom, pm1->scratch);
  pm1->CalcFiducialVelocity();

  if (pm1->opt_solver.equilibrium_initial)
  {
    M1::M1::vars_Lab U_C { {pm1->N_GRPS,pm1->N_SPCS},
                          {pm1->N_GRPS,pm1->N_SPCS},
                          {pm1->N_GRPS,pm1->N_SPCS} };

    pm1->SetVarAliasesLab(pm1->storage.u, U_C);

    M1::M1::vars_Source U_S { {pm1->N_GRPS,pm1->N_SPCS},
                              {pm1->N_GRPS,pm1->N_SPCS},
                              {pm1->N_GRPS,pm1->N_SPCS} };


    M1_ILOOP3(k, j, i)
    {
      M1::Equilibrium::SetEquilibrium(*pm1, U_C, U_S, k, j, i);
    }
  }
#endif  // M1_ENABLED
  */
  // consistent pressure atmosphere -------------------------------------------
  bool id_floor_primitives = pin->GetOrAddBoolean(
    "problem", "id_floor_primitives", false);

  if (id_floor_primitives)
  {

    for (int k=0; k<=ncells3-1; ++k)
    for (int j=0; j<=ncells2-1; ++j)
    for (int i=0; i<=ncells1-1; ++i)
    {
#if USETM
      peos->ApplyPrimitiveFloors(phydro->w, pscalars->r, k, j, i);
#else
      peos->ApplyPrimitiveFloors(phydro->w, k, j, i);
#endif
    }

  }
  // --------------------------------------------------------------------------


  // Initialise conserved variables
  peos->PrimitiveToConserved(phydro->w,
                             pscalars->r,
                             pfield->bcc,
                             phydro->u,
                             pscalars->s,
                             pcoord,
                             0, ncells1-1,
                             0, ncells2-1,
                             0, ncells3-1);

  // --------------------------------------------------------------------------
  // The following is now done else-where and is redundant here
  /*
  // Set up ADM matter variables
  pz4c->GetMatter(pz4c->storage.mat,
                  pz4c->storage.adm,
                  phydro->w,
                  pscalars->r,
                  pfield->bcc);

  pz4c->ADMConstraints(pz4c->storage.con,
                       pz4c->storage.adm,
                       pz4c->storage.mat,
                       pz4c->storage.u);
  */
  // --------------------------------------------------------------------------
  return;
}


void Mesh::DeleteTemporaryUserMeshData()
{
 if (!resume_flag && SGRID_grid_exists()) {
    SGRID_free_everything();
  }
#if USETM
  // Free cold EOS data
  delete ceos;
#endif
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
  /*
  // BD: TODO in principle this should be possible
  Z4c_AMR *const pz4c_amr = pmb->pz4c->pz4c_amr;

  // ensure we actually have a tracker
  if (pmb->pmy_mesh->ptracker_extrema->N_tracker > 0)
  {
    return 0;
  }

  return pz4c_amr->ShouldIRefine(pmb);
  */

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
      else if (ptracker_extrema->ref_type(n-1) == 2)
      {
        // If any excision; activate this refinement
        bool use = false;

        // Get the minimal radius over all apparent horizons
        Real horizon_radius = std::numeric_limits<Real>::infinity();

        for (auto pah_f : pmesh->pah_finder)
        {
          if (not pah_f->ah_found)
            continue;

          if (pah_f->rr_min < horizon_radius)
          {
            horizon_radius = pah_f->rr_min;
          }
          else
          {
            continue;
          }

          // populate the tracker with AHF based information
          // ptracker_extrema->c_x1(n-1) = pah_f->center[0];
          // ptracker_extrema->c_x2(n-1) = pah_f->center[1];
          // ptracker_extrema->c_x3(n-1) = pah_f->center[2];
          ptracker_extrema->ref_zone_radius(n-1) = (
            pah_f->rr_min
          );

          use = true;
        }

        if (use)
        {
          is_contained = pmb->SphereIntersects(
            ptracker_extrema->c_x1(n-1),
            ptracker_extrema->c_x2(n-1),
            ptracker_extrema->c_x3(n-1),
            ptracker_extrema->ref_zone_radius(n-1)
          );
        }
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

}
