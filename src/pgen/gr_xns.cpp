//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file gr_xns.cpp
//  \brief Initial conditions for rotating neutron star from XNS code
//         https://github.com/niccolo-bucciantini/XNS4.0
//         HDF5 data can be generated using
//         https://bitbucket.org/merlin-neutronstars/xns_tools

#include <cassert> // assert
#include <iostream>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
#include "../utils/interp_table.hpp" // InterpTable2D
#include "../utils/inputs/hdf5_reader.hpp" // HDF5TableLoader
#include "../z4c/z4c.hpp"
#include "../hydro/hydro.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#if M1_ENABLED
#include "../m1/m1.hpp"
#include "../m1/m1_set_equilibrium.hpp"
#endif  // M1_ENABLED

// Only proceed if HDF5 enabled
#ifdef HDF5OUTPUT

// External library headers
#include <hdf5.h>

// Determine floating-point precision (in memory, not file)
#if SINGLE_PRECISION_ENABLED
#define H5T_REAL H5T_NATIVE_FLOAT
#else
#define H5T_REAL H5T_NATIVE_DOUBLE
#endif

using namespace std;

namespace {

  // Names of XNS variables
  static constexpr char const * const XNS_fields[] = {
    "xns.R", "xns.TH",
    "xns.radius", "xns.theta",
    "xns.alpha", "xns.beta", "xns.psi", 
    "xns.rho", "xns.pres", "xns.vphi"
    "xns.bpol", "xns.btor",
    "xns.bpolr", "xns.bpolt",
    "xns.btot",
    "xns.chi",
    "xns.epol",
    "xns.jpol", "xns.jtor",
  };
  // Indexes of XNS variables
  enum{IXNS_R, IXNS_TH,
       IXNS_radius, IXNS_theta,
       IXNS_alpha, IXNS_beta, IXNS_psi,
       IXNS_rho, IXNS_pres, IXNS_vphi,
       IXNS_bpol, IXNS_btor,
       IXNS_bpolr, IXNS_bpolt,
       IXNS_btot,
       IXNS_chi,
       IXNS_epol,
       IXNS_jpol,
       IXNS_jtor,
       NXNSVars,
  };

  class XNSData {
  public:
    InterpTable2D * table;
    int NTH, NR; // XNS grid points
    Real mb; // XNS Baryon mass
    
    // Unit conversion from XNS code to CGS
    Real b_units;   // B-field
    Real r_units;   // length
    Real rho_units; // mass density

  private:
    const int nvar = NXNSVars;
    const char *fields = XNS_fields;

  public:
    XNSData() {
    }
    
    ~XNSData() {
      if(table)
        delete table;
    }
    
    void ReadData(string h5filename) {
      
      // Read attributes from HDF5 file 
      H5File file(h5filename, H5F_ACC_RDONLY);
      Group root = file.openGroup("/");
      
      Attribute attr = root.openAttribute("NR");
      attr.read(attr.getDataType(), &NR);

      Attribute attr = root.openAttribute("NTH");
      attr.read(attr.getDataType(), &NTH);
      
      Attribute attr = root.openAttribute("b_units");
      attr.read(attr.getDataType(), &b_units);
      
      Attribute attr = root.openAttribute("r_units");
      attr.read(attr.getDataType(), &r_units);
      
      Attribute attr = root.openAttribute("rho_units");
      attr.read(attr.getDataType(), &rho_units);

      //TODO: 
      //Attribute attr = root.openAttribute("mb");
      //attr.read(attr.getDataType(), &mb);
      
      file.close();

      // Set the table
      table = new InterpTable2D;
      table->InterpTable2D(nvar, NTH,NTR);
      
      // Read HDF5 dataset into 2D table
      HDF5TableLoader(h5filename, table, nvar,
                      fields, "R", "TH");
    }
        
    Real Interp(int var, Real xp, Real yp, Real zp) {
      //TODO check the grid in interp_table.cpp
      const Real rp = std::sqrt(xp*xp + yp*yp + zp*zp);
      const Real thetap = (rp>0.0)? std::acos(zp/rp) : 0.0; //TODO check acos range!
      // Bilinear interp
      return table.interpolate(var,rp,thetap);
      //TODO: Add 4th order interp from RNSC
    }
    
    void CartesianMetric(Real alpha, Real beta, Real psi,
                         Real xp, Real yp, Real zp,
                         Real &betax, Real &betay, Real &betaz,
                         Real &gxx, Real &gxy, Real &gxz,
                         Real &gyy, Real &gyz, Real &gzz) {
      //Real rp = std::sqrt(xp*xp + yp*yp + zp*zp);
      Real rcylp = std::sqrt(xp*xp + yp*yp);
      //Real costhp = zp/rp;
      //Real sinthp = rcylp/rp;
      
      Real psi4 = std::pow(psi,4);

      // beta_i
      beta_x = - psi4 * beta * yp;
      beta_y =   psi4 * beta * xp;
      beta_z = 0.0;
      
      // beta^i = (omega(ZAMO) y, - omega(ZAMO) x, 0)
      betax = - beta * yp; 
      betay =   beta * xp;
      betaz = 0.0;

      // gamma_ij
      gxx = psi4; 
      gxy = 0.0;
      gxz = 0.0;
      gyy = psi4;
      gyz = 0.0;
      gzz = psi4;
    }

    Real CartesianVector(Real vphi, Real psi,
                         Real xp, Real yp, Real zp,
                         Real &vx, Real &vy, Real &vz) {
      Real rcylp = std::sqrt(xp*xp + yp*yp);
      vx = - vphi * yp; 
      vy =   vphi * xp;
      vz = 0.0;
      return SQR(psi) * rcylp * std::fabs(omega);
    }
    
  } XNS;

#if USETM
  Primitive::ColdEOS<Primitive::COLDEOS_POLICY> * ceos = NULL;
#endif

  //void SeedMagneticFields(MeshBlock *pmb, ParameterInput *pin);
  // int RefinementCondition(MeshBlock *pmb);
}

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin)
{
  EnrollUserStandardHydro(pin);
  EnrollUserStandardField(pin);
  EnrollUserStandardZ4c(pin);
  EnrollUserStandardM1(pin);

  if (!resume_flag) {
    // Read XNS data
    string h5_fname = pin->GetOrAddString("problem", "filename", "xns.hdf5");
    XNS.ReadData(h5_fname);
#if USETM
    ceos = new Primitive::ColdEOS<Primitive::COLDEOS_POLICY>;
    InitColdEOS(ceos, pin);
#endif
  }

  // if(adaptive==true)
  //   EnrollUserRefinementCondition(RefinementCondition);

  return;
}

void Mesh::UserWorkAfterLoop(ParameterInput *pin)
{
  return;
}

//----------------------------------------------------------------------------------------
//! \fn
// \brief Setup User work

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin)
{
  // Allocate output arrays for fluxes
#if M1_ENABLED
  AllocateUserOutputVariables(4);
#endif // M1_ENABLED
  return;
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin)
{
#if M1_ENABLED
  AA & fl = pm1->ev_strat.masks.flux_limiter;

  int il = pm1->mbi.nn1;
  int jl = pm1->mbi.nn2;
  int kl = pm1->mbi.nn3;

  if (pm1->opt.flux_limiter_use_mask)
  M1_ILOOP3(k, j, i)
  {
    user_out_var(0,k,j,i) = fl(0,k,j,i);
    user_out_var(1,k,j,i) = fl(1,k,j,i);
    user_out_var(2,k,j,i) = fl(2,k,j,i);
  }

  if (pm1->opt.flux_lo_fallback)
  M1_ILOOP3(k, j, i)
  {
    user_out_var(3,k,j,i) = pm1->ev_strat.masks.pp(k,j,i);
  }
#endif // M1_ENABLED
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Sets the initial conditions.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin)
{

#ifdef Z4C_ASSERT_FINITE
  // as a sanity check (these should be over-written)
  pz4c->adm.psi4.Fill(NAN);
  pz4c->adm.g_dd.Fill(NAN);
  pz4c->adm.K_dd.Fill(NAN);

  pz4c->z4c.chi.Fill(NAN);
  pz4c->z4c.Khat.Fill(NAN);
  pz4c->z4c.Theta.Fill(NAN);
  pz4c->z4c.alpha.Fill(NAN);
  pz4c->z4c.Gam_u.Fill(NAN);
  pz4c->z4c.beta_u.Fill(NAN);
  pz4c->z4c.g_dd.Fill(NAN);
  pz4c->z4c.A_dd.Fill(NAN);

  /*
  pz4c->con.C.Fill(NAN);
  pz4c->con.H.Fill(NAN);
  pz4c->con.M.Fill(NAN);
  pz4c->con.Z.Fill(NAN);
  pz4c->con.M_d.Fill(NAN);
  */
#endif

  bool verbose = pin->GetOrAddBoolean("problem", "verbose", 0);

  // container with idx / grids pertaining z4c
  MB_info* mbi = &(pz4c->mbi);

  //---------------------------------------------------------------------------
  // Interpolate ADM metric

  if(verbose)
    std::cout << "Interpolating ADM metric on current MeshBlock." << std::endl;

  // Populate coordinates
  Real *x = new Real[mbi->nn1];
  Real *y = new Real[mbi->nn2];
  Real *z = new Real[mbi->nn3];

  for(int i = 0; i < mbi->nn1; ++i) {
    x[i] = mbi->x1(i);
  }
  for(int i = 0; i < mbi->nn2; ++i) {
    y[i] = mbi->x2(i);
  }
  for(int i = 0; i < mbi->nn3; ++i) {
    z[i] = mbi->x3(i);
  }

  for (int k=0; k<mbi->nn3; ++k)
  for (int j=0; j<mbi->nn2; ++j)
  for (int i=0; i<mbi->nn1; ++i)
  {
    Real xp = x[i];
    Real yp = y[j];
    Real zp = z[k];
    
    // Interpolate
    Real alpha = XNS.Interp(IXNS_alpha, xp,yp,zp);
    Real beta = XNS.Interp(IXNS_beta, xp,yp,zp); // NB beta^phi = - omega(ZAMO)
    Real psi = XNS.Interp(IXNS_psi, xp,yp,zp);

    // Get Cartesian components
    Real betax = 0.0; // beta^i
    Real betay = 0.0;
    Real betaz = 0.0;

    Real gxx = 0.0; // gamma_ij
    Real gxy = 0.0;
    Real gxz = 0.0;
    Real gyy = 0.0;
    Real gyz = 0.0;
    Real gzz = 0.0;

    XNS.CartesianMetric(alpha, beta, psi,
                        xp,yp,zp,
                        betax,betay,betaz,
                        gxx,gxy,gxz,gyy,gyz,gzz);
    
    pz4c->storage.adm(Z4c::I_ADM_gxx,k,j,i) = gxx; // gamma_ij
    pz4c->storage.adm(Z4c::I_ADM_gxy,k,j,i) = gxy;
    pz4c->storage.adm(Z4c::I_ADM_gxz,k,j,i) = gxz;
    pz4c->storage.adm(Z4c::I_ADM_gyy,k,j,i) = gyy;
    pz4c->storage.adm(Z4c::I_ADM_gyz,k,j,i) = gyz;
    pz4c->storage.adm(Z4c::I_ADM_gzz,k,j,i) = gzz;

    pz4c->storage.adm(Z4c::I_ADM_Kxx,k,j,i) = 0.0;
    pz4c->storage.adm(Z4c::I_ADM_Kxy,k,j,i) = 0.0;
    pz4c->storage.adm(Z4c::I_ADM_Kxz,k,j,i) = 0.0;
    pz4c->storage.adm(Z4c::I_ADM_Kyy,k,j,i) = 0.0;
    pz4c->storage.adm(Z4c::I_ADM_Kyz,k,j,i) = 0.0;
    pz4c->storage.adm(Z4c::I_ADM_Kzz,k,j,i) = 0.0;

    pz4c->storage.adm(Z4c::I_ADM_alpha,k,j,i) = alpha;
    pz4c->storage.adm(Z4c::I_ADM_betax,k,j,i) = betax; // beta^i
    pz4c->storage.adm(Z4c::I_ADM_betay,k,j,i) = betay;
    pz4c->storage.adm(Z4c::I_ADM_betaz,k,j,i) = betaz;

    pz4c->storage.adm(Z4c::I_ADM_psi4,k,j,i) = std::pow(psi,4);
  }

  delete x; delete y; delete z;

  //---------------------------------------------------------------------------
  // ADM-to-Z4c
  
  pz4c->ADMToZ4c(pz4c->storage.adm, pz4c->storage.u);
  pz4c->ADMToZ4c(pz4c->storage.adm, pz4c->storage.u1);

  //---------------------------------------------------------------------------
  // Interpolate primitives

  if(verbose)
    std::cout << "Interpolating primitives on current MeshBlock." << std::endl;

  x = new Real[ncells1];
  y = new Real[ncells2];
  z = new Real[ncells3];

  // Populate coordinates
  for(int i = 0; i < ncells1; ++i) {
    x[i] = pcoord->x1v(i);
  }
  for(int i = 0; i < ncells2; ++i) {
    y[i] = pcoord->x2v(i);
  }
  for(int i = 0; i < ncells3; ++i) {
    z[i] = pcoord->x3v(i);
  }

  Real pres_diff = 0.0;

#if USETM
  Real rho_min = pin->GetReal("hydro", "dfloor");
#endif

  for (int k=0; k<ncells3; ++k)
  for (int j=0; j<ncells2; ++j)
  for (int i=0; i<ncells1; ++i)
  {
    Real xp = x[i];
    Real yp = y[j];
    Real zp = z[k];

    // Interpolate fluid
    Real rho = XNS.Interp(IXNS_rho, xp,yp,zp);
    Real pres = XNS.Interp(IXNS_pres, xp,yp,zp); 
    Real vphi = XNS.Interp(IXNS_psi, xp,yp,zp); // v^\phi 

    // Interpolate metric & get Cartesian components
    Real alpha = XNS.Interp(IXNS_alpha, xp,yp,zp);
    Real beta = XNS.Interp(IXNS_beta, xp,yp,zp); 
    Real psi = XNS.Interp(IXNS_psi, xp,yp,zp);

    // Metric
    Real betax = 0.0;
    Real betay = 0.0;
    Real betaz = 0.0;

    Real gxx = 0.0; 
    Real gxy = 0.0;
    Real gxz = 0.0;
    Real gyy = 0.0;
    Real gyz = 0.0;
    Real gzz = 0.0;

    XNS.CartesianMetric(alpha, beta, psi,
                        xp,yp,zp,
                        betax,betay,betaz,
                        gxx,gxy,gxz,gyy,gyz,gzz);
    
    // Cartesian components v^i
    Real vx = 0.0; 
    Real vy = 0.0; 
    Real vz = 0.0;
    Real v2 = XNS.CartesianVector(vphi, psi,
                                  xp,yp,rp,
                                  vx, vy, vz);

    // Check //TODO remove
    Real v_x = gxx*vx + gxy*vy + gxz*vz;
    Real v_y = gxy*vx + gyy*vy + gyz*vz;
    Real v_z = gxz*vx + gyz*vy + gzz*vz; 
    Real _v2 = v_x*vx + v_y*vy + v_z*vz;
    if (fabs(v2) < 1e-20) v2 = 0.;
    assert(std::abs(v2=_v2)<1e-15); // Some formulas are wrong

    // Lorentz factor
    Real W = 1.0/std::sqrt(1.0-v2);

    // u^i velocity
    Real ux = W * vx;
    Real uy = W * vy;
    Real uz = W * vz;
    
#if USETM
#if defined(USE_COMPOSE_EOS) || defined(USE_HYBRID_EOS)
    rho *= ceos->mb/mb_xns; // adjust for baryon mass
#endif
    if (rho > rho_min) {
      Real pres_eos = ceos->GetPressure(rho[flat_ix]);
      Real pres_diff = max(abs(pres / pres_eos - 1), pres_diff);
      pres = pres_eos;
    }

#if NSCALARS > 0
    for (int l=0; l<NSCALARS; ++l)
      pscalars->r(l,k,j,i) = ceos->GetY(rho, l);
#endif
#endif

    phydro->w(IDN,k,j,i) = rho;
    phydro->w(IPR,k,j,i) =  pres;
    phydro->w(IVX, k, j, i) = ux;
    phydro->w(IVY, k, j, i) = uy;
    phydro->w(IVZ, k, j, i) = uz;

    // Check hydro is finite
    for (int n=0; n<NHYDRO; ++n)
    if (!std::isfinite(phydro->w(n,k,j,i)))
      {
        std::cout << "WARNING: Interpolated hydro not finite, applying floors" << std::endl;
#if USETM
        peos->ApplyPrimitiveFloors(phydro->w, pscalars->r, k, j, i);
#else
        peos->ApplyPrimitiveFloors(phydro->w, k, j, i);
#endif
      }
    
  }

  if (pres_diff > 1e-3)
    std::cout << "WARNING: Interpolated pressure does not match eos. abs. rel. diff = "
              << pres_diff << std::endl;

  delete x; delete y; delete z;

  //---------------------------------------------------------------------------
  // Initialise conserved variables

  if(verbose)
    std::cout << "Initializing conservatives on current MeshBlock." << std::endl;

  // Prepare index bounds
  int il = is - NGHOST;
  int iu = ie + NGHOST;
  int jl = js;
  int ju = je;
  if (block_size.nx2 > 1) {
    jl -= NGHOST;
    ju += NGHOST;
  }
  int kl = ks;
  int ku = ke;
  if (block_size.nx3 > 1) {
    kl -= NGHOST;
    ku += NGHOST;
  }

  peos->PrimitiveToConserved(phydro->w,
		  pscalars->r,
		  pfield->bcc, phydro->u,
		  pscalars->s,
		  pcoord, il, iu, jl, ju, kl, ku);

  // --------------------------------------------------------------------------
  
#ifdef Z4C_ASSERT_FINITE
  pz4c->assert_is_finite_adm();
  pz4c->assert_is_finite_con();
  pz4c->assert_is_finite_mat();
  pz4c->assert_is_finite_z4c();
#endif
  
#if MAGNETIC_FIELDS_ENABLED

  // --------------------------------------------------------------------------
  // Initialize magnetic fields
  
  pfield->b.x1f.ZeroClear();
  pfield->b.x2f.ZeroClear();
  pfield->b.x3f.ZeroClear();
  pfield->bcc.ZeroClear();

  // Construct cell centred B field 
  //for(int k=pmb->ks-1; k<=pmb->ke+1; k++)
  //for(int j=pmb->js-1; j<=pmb->je+1; j++)
  //for(int i=pmb->is-1; i<=pmb->ie+1; i++)
  for (int k=0; k<ncells3; k++)
  for (int j=0; j<ncells2; j++)
  for (int i=0; i<ncells1; i++)
  {

    //TODO: check where XNS B is defined
    //TODO: coords, interp, cartesian components
    Real bccx= 0.0;
    Real bccy= 0.0;
    Real bccz= 0.0;
    
    pfield->bcc(0,k,j,i) = bccx;
    pfield->bcc(1,k,j,i) = bccy;
    pfield->bcc(2,k,j,i) = bccz;
    
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

#endif

  return;
}

