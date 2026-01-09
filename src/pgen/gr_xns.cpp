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
#include <iomanip>  

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
#include "../utils/lagrange_interp.hpp"
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
#endif

using namespace std;

namespace XNS {

#define CHECK_V2 (1) // just a check on v^2 computation
#define DEBUG (1) // to dump some data and info
  
  // Names of XNS 2D fields (HDF5 Datasets) 
  static constexpr char const * const XNS_dataset[] = {
    "alpha", "beta", "psi", 
    "rho", "pres", "vphi",
    "bpol", "btor",
    "b3", "bpolr", "bpolt", // B^\phi, B^r, B^\theta
    "btot",
    "chi",
    "epol",
    "jpol", "jtor",
  };
  // Indexes of XNS 2D fields
  enum{IXNS_alpha, IXNS_beta, IXNS_psi,
       IXNS_rho, IXNS_pres, IXNS_vphi,
       IXNS_bpol, IXNS_btor,
       IXNS_b3, IXNS_bpolr, IXNS_bpolt, // B^\phi, B^r, B^\theta
       IXNS_btot,
       IXNS_chi,
       IXNS_epol,
       IXNS_jpol, IXNS_jtor,
       NXNSVars,
  };

  // Indexes for unit conversion
  // (Conversion not needed, XNS works in G=Mo=c=1)
  enum{DIMLESS,
       LENGTH,
       MDENST,
       PRESS,
       VELOCITY,
       BFIELD,
       NUNITS,
  };
  
  class XNSData {
  public:

    int NTH, NR; // XNS grid points
    Real mb; // XNS Baryon mass
    
    // Unit conversion from XNS code to CGS
    Real b_units; // B-field
    Real r_units; // length
    Real rho_units; // mass density

    static const int matter_interp_order = 2;
    static const int metric_interp_order = 2*NGHOST-1;
    
  private:

    //AA xnsdata[NXNSVars];
    //AA xns_radius, xns_theta;
    AthenaArray<Real> xns_radius, xns_theta;
    AthenaArray<Real> xnsdata[NXNSVars];
  
    LagrangeInterpND<matter_interp_order, 2> * pinterp2_matter = nullptr;
    LagrangeInterpND<metric_interp_order, 2> * pinterp2_metric = nullptr;
    
  public:

    XNSData(){

    }
    
    ~XNSData() {

      for (int v = 0; v < NXNSVars; ++v) {
        xnsdata[v].DeleteAthenaArray();
      }
    }
    
    void ReadData(const std::string &h5filename) {

        // Open file
        hid_t plist = H5Pcreate(H5P_FILE_ACCESS);
        H5Pset_fapl_mpio(plist, MPI_COMM_WORLD, MPI_INFO_NULL);
        hid_t file = H5Fopen(h5filename.c_str(), H5F_ACC_RDONLY, plist);
        H5Pclose(plist);
        assert(file >= 0);

        // --------------------------------------------------
        // Read scalar attributes
        // --------------------------------------------------
        auto readIntAttr = [&](const char* name, int &val) {
            if (H5Aexists(file, name) > 0) {
                hid_t attr = H5Aopen(file, name, H5P_DEFAULT);
                H5Aread(attr, H5T_NATIVE_INT, &val);
                H5Aclose(attr);
            }
        };
        auto readRealAttr = [&](const char* name, Real &val) {
            if (H5Aexists(file, name) > 0) {
                hid_t attr = H5Aopen(file, name, H5P_DEFAULT);
                double tmp = 0.0;
                H5Aread(attr, H5T_NATIVE_DOUBLE, &tmp);
                val = static_cast<Real>(tmp);
                H5Aclose(attr);
            }
        };
        readIntAttr("NR", NR);
        readIntAttr("NTH", NTH);
        readRealAttr("bfield_unit", b_units);
        readRealAttr("length_unit", r_units);
        readRealAttr("massdens_unit", rho_units);

        mb = 1.0;

        // Dataset transfer property list for MPI
        hid_t dxpl = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_INDEPENDENT);

        // --------------------------------------------------
        // Read 1D coordinates
        // --------------------------------------------------
        auto read1D = [&](const char* name, AA &array, int N) {
            hid_t dset = H5Dopen(file, name, H5P_DEFAULT);
            assert(dset >= 0);
            std::vector<double> buffer(N);
            H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, dxpl, buffer.data());
            array.NewAthenaArray(N);
            for (int i = 0; i < N; ++i) array(i) = buffer[i];
            H5Dclose(dset);
        };
        read1D("R", xns_radius, NR);
        read1D("TH", xns_theta, NTH);

        // --------------------------------------------------
        // Allocate 2D XNS data arrays
        // --------------------------------------------------
        for (int v = 0; v < NXNSVars; ++v) {
          xnsdata[v].NewAthenaArray(NTH, NR);
          xnsdata[v].ZeroClear();
        }
        // --------------------------------------------------
        // Read 2D datasets
        // --------------------------------------------------
        for (int v = 0; v < NXNSVars; ++v) {
            hid_t dset = H5Dopen(file, XNS_dataset[v], H5P_DEFAULT);
            assert(dset >= 0);
            std::vector<double> dbuffer(NTH*NR);
            H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, dxpl, dbuffer.data());
            for (int i = 0; i < NTH; ++i)
                for (int j = 0; j < NR; ++j)
                    xnsdata[v](i,j) = dbuffer[i*NR + j]; //static_cast<Real>(NR-j) /
                                      //static_cast<Real>(NR*NTH);
            H5Dclose(dset);
        }

        // Cleanup
        H5Pclose(dxpl);
        H5Fclose(file);
    }

    void WriteXNSGridToFile() const {

      std::ofstream fout("xns_radius");
      fout << std::setprecision(16);
      fout << "# XNS grid\n";
      fout << "# NR = " << NR << std::endl;
      fout << "# Columns: radius\n";
      for (int i = 0; i < NR; ++i) {
	fout << xns_radius(i);
      }
      fout.close();
      
      std::ofstream fout("xns_theta");
      fout << std::setprecision(16);
      fout << "# XNS grid\n";
      fout << "# NTH = " << NTH << std::endl;
      fout << "# Columns: theta\n";
      for (int i = 0; i < NTH; ++i) {
	fout << xns_theta(i);
      }
      fout.close();
      
    }
    
    void WriteXNSDataToFile(int v,
			    const std::string& prefix = "xns_field_") const {
        std::stringstream fname;
        fname << prefix << v << ".txt";

        std::ofstream fout(fname.str().c_str());
        fout << std::setprecision(16);

        fout << "# XNS data dump\n";
        fout << "# field index v = " << v << "\n";
        fout << "# Columns: theta  radius  value\n";

        for (int i = 0; i < NTH; ++i) {
            for (int j = 0; j < NR; ++j) {
            fout << xns_theta(i) << " "
                << xns_radius(j) << " "
                << xnsdata[v](i,j) << "\n";
            }
            fout << "\n";  // separate theta slices
        }

        fout.close();
    }

    void PrepareInterp(Real xp, Real yp, Real zp, int order) {
      // Interpolator of 2D variable at (r,theta) <- (x,y,z) 

      const Real rp = std::sqrt(SQR(xp) + SQR(yp) + SQR(zp));
      const Real thetap = (rp>0.0)? std::acos(zp/rp) : 0.0; //TODO check acos range!

      Real origin[2];
      Real delta[2];
      int size[2];
      Real coord[2];
      
      origin[1] = xns_theta(0); 
      origin[0] = xns_radius(0);

      // NB Assumes uniform spacing!
      delta[1] = xns_theta(1)-xns_theta(0);
      delta[0] = xns_radius(1)-xns_radius(0);
      
      size[1] = NTH;
      size[0] = NR;
      
      coord[1] = thetap; //std::min(xns_theta(NTH-1),
            //std::max(xns_theta(0), thetap));
      coord[0] = rp; //std::min(xns_radius(NR-1),
            //std::max(xns_radius(0), rp));
      
      if (order == metric_interp_order) {
        pinterp2_metric =
          new LagrangeInterpND<metric_interp_order, 2>(origin, delta, size, coord);
      } else if (order == matter_interp_order) {
        pinterp2_matter =
          new LagrangeInterpND<matter_interp_order, 2>(origin, delta, size, coord);
      } else {
        std::stringstream msg;
        msg << "### FATAL ERROR in XNS pgen" << std::endl
            << "interpolation order =" << order << std::endl;
        ATHENA_ERROR(msg);
      }
    }
    
    void FreeInterp(int order) {
      if (order == metric_interp_order) {
        delete pinterp2_metric;
      } else if (order == matter_interp_order) {
        delete pinterp2_matter;
      }
    }

    Real Interp(int var, int order, int unit) {
      assert(unit >= 0 and unit < NUNITS);
      int uconv = 1.0;
      // XNS works in G=Mo=c=1, which is what we want.
      // No conversion needed.
      // if (unit == LENGHT) uconv = r_units; // Geom to Km
      // if (unit == MDENST) uconv = rho_units; // Geom to g
      // if (unit == BFIELD) uconv = b_units; // Geom to Gauss
        
      if (order == metric_interp_order) {
        return uconv * pinterp2_metric->eval(&(xnsdata[var](0,0))); 
      } else if (order == matter_interp_order) {
        return uconv * pinterp2_matter->eval(&(xnsdata[var](0,0))); 
      } else {
        std::stringstream msg;
        msg << "### FATAL ERROR in XNS pgen" << std::endl
            << "interpolation order =" << order << std::endl;
        ATHENA_ERROR(msg);
      }
    }
    
    Real CartesianMetric(Real beta, Real psi,
                         Real xp, Real yp, Real zp,
                         Real &betax, Real &betay, Real &betaz,
                         Real &gxx, Real &gxy, Real &gxz,
                         Real &gyy, Real &gyz, Real &gzz) {
      // Given the XNS metric functions beta and psi
      // returns Cartesian components of 3-metric and shift
      // and conformal factor psi^4
      
      Real psi4 = std::pow(psi,4);
      
      // beta^i and beta^2 = beta_i beta^i
      Real beta2 = CartesianVector(beta, psi4, xp,yp,zp,
                                   betax, betay, betaz);
      // gamma_ij
      gxx = psi4; 
      gxy = 0.0;
      gxz = 0.0;
      gyy = psi4;
      gyz = 0.0;
      gzz = psi4;
      
      // beta_i
      //beta_x = psi4 * betax;
      //beta_y = psi4 * betay;
      //beta_z = 0.0;
      
      return psi4;
    }
        
    Real CartesianVector(Real vr, Real vtheta, Real vphi,
                         Real psi4,
                         Real xp, Real yp, Real zp,
                         Real &vx, Real &vy, Real &vz) {
      // Given the components v^r, v^\theta, v^\phi and conf. fact.
      // returns Cartesian components of the 3-vector and modulus

      // Cylindrical radius and phi
      Real rcylp = 0.0;
      Real sinphi = 0.0;
      Real cosphi = 0.0;
      Real rcylp2 = SQR(xp) + SQR(yp);
      if (rcylp2>0.0) {
        rcylp = std::sqrt(rcylp2); // = r sin(theta)
        sinphi = yp/rcylp;
        cosphi = xp/rcylp;
      } else {
        rcylp = 0.0;
        sinphi = 0.0;
        cosphi = 0.0;
      }

      // Spherical radius and theta
      Real rp = 0.0;
      Real sintheta = 0.0;
      Real costheta = 0.0;
      Real rp2 = SQR(xp) + SQR(yp) + SQR(zp);
      if (rp2>0.0) {
        rp = std::sqrt(rp2); 
        sintheta = rcylp/rp;
        costheta = zp/rp;
      } else {
        rp = 0.0;
        sintheta = 0.0;
        costheta = 0.0;
      }

      // v^i
      vx = vr * sintheta * cosphi + vtheta * costheta * cosphi - vphi * sinphi;
      vy = vr * sintheta * sinphi + vtheta * costheta * sinphi + vphi * cosphi;
      vz = vr * costheta - vtheta * sintheta;

      // v_i v^i =  \psi^4 \delta_ij v^j v^i
      return psi4 *(SQR(vx) + SQR(vy) + SQR(vz));
    }

    Real CartesianVector(Real vphi,
                         Real psi4,
                         Real xp, Real yp, Real zp,
                         Real &vx, Real &vy, Real &vz) {
      // Given the non-zero component v^\phi and conf. fact.
      // returns Cartesian components of the 3-vector and modulus

      Real rcylp = 0.0;
      Real sinphi = 0.0;
      Real cosphi = 0.0;
      Real rcylp2 = SQR(xp) + SQR(yp);
      if (rcylp2>0.0) {
        rcylp = std::sqrt(rcylp2); // = r sin(theta)
        sinphi = yp/rcylp;
        cosphi = xp/rcylp;
      } else {
        rcylp = 0.0;
        sinphi = 0.0;
        cosphi = 0.0;
      }
      
      vx = - vphi * sinphi; 
      vy =   vphi * cosphi;
      vz = 0.0;
      return psi4 * SQR(vphi);
    } 
    
  }; // class XNSData

string h5_fname;
//extern XNSData *xns_data;
//XNSData *xns_data = nullptr;

}

using namespace XNS;

namespace {
  
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
    // Set XNS HDF5 filename 
    h5_fname = pin->GetOrAddString("problem", "filename", "xns.hdf5");
    //XNSData XNS;

    //XNS.ReadData(h5_fname);

    //XNS::xns_data = new XNS::XNSData();
    //XNS::xns_data->ReadData(h5_fname);

    //Save coordinates and fields as .txt files
    //XNS.WriteXNSGridToFile();

    // Example: dump only matter fields
    //XNS.WriteXNSDataToFile(IXNS_rho);
    //XNS.WriteXNSDataToFile(IXNS_pres);
    //XNS.WriteXNSDataToFile(IXNS_vphi);
    
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
  bool verbose = pin->GetOrAddBoolean("problem", "verbose", 0);

  // XNSData object and read XNS data
  XNSData XNS;
  XNS.ReadData(h5_fname);

XNSData XNS;
XNS.ReadData(h5_fname);
//XNSData &XNS = *(XNS::xns_data);

//XNS.WriteXNSDataToFile(IXNS_rho);
//XNS.WriteXNSDataToFile(IXNS_pres);
//XNS.WriteXNSDataToFile(IXNS_vphi);

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

  const int matter_interp_order = XNS.matter_interp_order;
  const int metric_interp_order = XNS.metric_interp_order;
  
  //---------------------------------------------------------------------------
  // Interpolate ADM metric
  
  if(verbose)
    std::cout << "Interpolating ADM metric on current MeshBlock." << std::endl;

  //pz4c->storage.adm.ZeroClear();

  for (int k=0; k<mbi->nn3; ++k)
  for (int j=0; j<mbi->nn2; ++j)
  for (int i=0; i<mbi->nn1; ++i)
  {
    // Coordinates
    Real xp = mbi->x1(i);
    Real yp = mbi->x2(j);
    Real zp = mbi->x3(k);
    
    // Interpolate XNS metric funs
    XNS.PrepareInterp(xp,yp,zp, metric_interp_order);

    Real alpha = XNS.Interp(IXNS_alpha, metric_interp_order, DIMLESS);
    Real beta = XNS.Interp(IXNS_beta, metric_interp_order, DIMLESS);
    Real psi = XNS.Interp(IXNS_psi, metric_interp_order, DIMLESS);

    XNS.FreeInterp(metric_interp_order);
    
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

    Real psi4 = XNS.CartesianMetric(beta, psi,
                                    xp,yp,zp,
                                    betax,betay,betaz,
                                    gxx,gxy,gxz,gyy,gyz,gzz);
    
    pz4c->storage.adm(Z4c::I_ADM_gxx,k,j,i) = gxx; // gamma_ij
    pz4c->storage.adm(Z4c::I_ADM_gxy,k,j,i) = gxy; // = psi^4 delta_ij
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

    pz4c->storage.adm(Z4c::I_ADM_psi4,k,j,i) = psi4;
  }

  //---------------------------------------------------------------------------
  // ADM-to-Z4c
  
  pz4c->ADMToZ4c(pz4c->storage.adm, pz4c->storage.u);
  pz4c->ADMToZ4c(pz4c->storage.adm, pz4c->storage.u1);

  //---------------------------------------------------------------------------
  // Interpolate primitives

  if(verbose)
    std::cout << "Interpolating primitives on current MeshBlock." << std::endl;

  Real pres_diff = 0.0;

#if USETM
  Real rho_min = pin->GetReal("hydro", "dfloor");
#endif

  // Open debug file once
  //std::ofstream fout("interp_debug.txt"); //txt
  //fout << std::setprecision(16);
  // Header
  //fout << "# xp    yp    zp    rp    thetap    rho    pres    vphi\n";
  //phydro->w.ZeroClear();
  //pscalars->r.ZeroClear();
  for (int k=0; k<ncells3; ++k)
  for (int j=0; j<ncells2; ++j)
  for (int i=0; i<ncells1; ++i)
  {
    // Coordinates
    Real xp = pcoord->x1v(i);
    Real yp = pcoord->x2v(j);
    Real zp = pcoord->x3v(k);

    // Interpolate fluid
    XNS.PrepareInterp(xp,yp,zp, matter_interp_order);

    Real rho = XNS.Interp(IXNS_rho, matter_interp_order, MDENST);
    Real pres = XNS.Interp(IXNS_pres, matter_interp_order, PRESS);
    Real vphi = XNS.Interp(IXNS_vphi, matter_interp_order, VELOCITY); // v^\phi 

    XNS.FreeInterp(matter_interp_order);

    //fout << xp << " " 
    //     << yp << " " 
    //     << zp << " " 
    //     << std::sqrt(xp*xp + yp*yp + zp*zp) << " " 
    //     << ((xp*xp + yp*yp + zp*zp > 0.0) ? std::acos(zp/std::sqrt(xp*xp + yp*yp + zp*zp)) : 0.0) << " " 
    //     << rho << " " 
    //     << pres << " " 
    //     << vphi << "\n";//txt above!


    // Interpolate metric & get Cartesian components
    XNS.PrepareInterp(xp,yp,zp, metric_interp_order);

    Real alpha = XNS.Interp(IXNS_alpha, metric_interp_order, DIMLESS);
    Real beta = XNS.Interp(IXNS_beta, metric_interp_order, DIMLESS);
    Real psi = XNS.Interp(IXNS_psi, metric_interp_order, DIMLESS);
    
    XNS.FreeInterp(metric_interp_order);

    // Metric
    Real betax = 0.0; // beta^i
    Real betay = 0.0;
    Real betaz = 0.0;

    Real gxx = 0.0; // gamma_ij
    Real gxy = 0.0;
    Real gxz = 0.0;
    Real gyy = 0.0;
    Real gyz = 0.0;
    Real gzz = 0.0;

    Real psi4 = XNS.CartesianMetric(beta, psi,
                                    xp,yp,zp,
                                    betax,betay,betaz,
                                    gxx,gxy,gxz,gyy,gyz,gzz);
    
    // Cartesian components v^i
    Real vx = 0.0; 
    Real vy = 0.0; 
    Real vz = 0.0;
    Real v2 = XNS.CartesianVector(vphi, psi4,
                                  xp,yp,zp,
                                  vx, vy, vz);
    
#if (CHECK_V2)
    // Check 
    Real v_x = psi4 * vx;
    Real v_y = psi4 * vy;
    Real v_z = psi4 * vz; 
    Real _v2 = v_x*vx + v_y*vy + v_z*vz;
    assert(std::abs(v2-_v2)<1e-12); // Some formulas are wrong

    v_x = gxx*vx + gxy*vy + gxz*vz;
    v_y = gxy*vx + gyy*vy + gyz*vz;
    v_z = gxz*vx + gyz*vy + gzz*vz; 
    _v2 = v_x*vx + v_y*vy + v_z*vz;
    assert(std::abs(v2-_v2)<1e-12); // Some formulas are wrong
#endif
    
    // Lorentz factor
    if (std::fabs(v2) < 1e-20) v2 = 0.0;
    assert(v2<1.0);
    Real W = 1.0/std::sqrt(1.0-v2);

    // u^i velocity
    Real ux = W * vx;
    Real uy = W * vy;
    Real uz = W * vz;
    
#if USETM
#if defined(USE_COMPOSE_EOS) || defined(USE_HYBRID_EOS)
    rho *= ceos->mb/XNS.mb; // adjust for baryon mass
#endif
    if (rho > rho_min) {
      Real pres_eos = ceos->GetPressure(rho);
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
      if (!std::isfinite(phydro->w(n,k,j,i))) {
        std::cout << "WARNING: Interpolated hydro not finite, applying floors" << std::endl;
#if USETM
        peos->ApplyPrimitiveFloors(phydro->w, pscalars->r, k, j, i);
#else
        peos->ApplyPrimitiveFloors(phydro->w, k, j, i);
#endif
      }
    
  }

  // Close file
  //fout.close();//txt above

  if (pres_diff > 1e-3)
    std::cout << "WARNING: Interpolated pressure does not match eos. abs. rel. diff = "
              << pres_diff << std::endl;

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

  // The following assumes B^i is defined at cc //TODO: check!
  
  pfield->b.x1f.ZeroClear();
  pfield->b.x2f.ZeroClear();
  pfield->b.x3f.ZeroClear();
  pfield->bcc.ZeroClear();
  
  // Construct cell centred B field 
  /* for(int k=pmb->ks-1; k<=pmb->ke+1; k++)
     for(int j=pmb->js-1; j<=pmb->je+1; j++)
     for(int i=pmb->is-1; i<=pmb->ie+1; i++)
  */
  for (int k=0; k<ncells3; k++)
  for (int j=0; j<ncells2; j++)
  for (int i=0; i<ncells1; i++)
  {
    // Coordinates
    Real xp = pcoord->x1v(i);
    Real yp = pcoord->x2v(j);
    Real zp = pcoord->x3v(k);

    // Interpolate B cc
    XNS.PrepareInterp(xp,yp,zp, matter_interp_order);
    
    Real Br = XNS.Interp(IXNS_bpolr, matter_interp_order, BFIELD);
    Real Btheta = XNS.Interp(IXNS_bpolt, matter_interp_order, BFIELD);
    Real Bphi = XNS.Interp(IXNS_b3, matter_interp_order, BFIELD);

    XNS.FreeInterp(matter_interp_order);

    //std::ofstream fout("interp_debugB.txt", std::ios::app);
    //fout << std::setprecision(16)
    //    << "xp=" << xp << " yp=" << yp << " zp=" << zp
    //    << " Br=" << Br
    //    << " Btheta=" << Btheta
    //    << " Bphi=" << Bphi
    //    << "\n";
    //fout.close();
    
    // Interpolate conf. fact.
    XNS.PrepareInterp(xp,yp,zp, metric_interp_order);

    Real psi = XNS.Interp(IXNS_psi, metric_interp_order, DIMLESS);
    Real psi4 = std::pow(psi, 4);

    XNS.FreeInterp(metric_interp_order);

    // Cartesian components
    Real bccx = 0.0;
    Real bccy = 0.0;
    Real bccz = 0.0;

    Real B2 = XNS.CartesianVector(Br, Btheta, Bphi,
                                  psi4, 
                                  xp,yp,zp,
                                  bccx, bccy, bccz);

    pfield->bcc(0,k,j,i) = bccx;
    pfield->bcc(1,k,j,i) = bccy;
    pfield->bcc(2,k,j,i) = bccz;

    //TODO: Must this be densitized ?
    Real sqrtdetgam = std::pow(psi4, 6); // = sqrt(det(gamma))
    pfield->bcc(0,k,j,i) *= sqrtdetgam;
    pfield->bcc(1,k,j,i) *= sqrtdetgam;
    pfield->bcc(2,k,j,i) *= sqrtdetgam;
    
  }

  // Initialise face centred field by averaging cc field
  for(int k=ks; k<=ke;   k++)
  for(int j=js; j<=je;   j++)
  for(int i=is; i<=ie+1; i++)
  {
    pfield->b.x1f(k,j,i) = 0.5*(pfield->bcc(0,k,j,i-1) +
                                pfield->bcc(0,k,j,i));
  }

  for(int k=ks; k<=ke;   k++)
  for(int j=js; j<=je+1; j++)
  for(int i=is; i<=ie;   i++)
  {
    pfield->b.x2f(k,j,i) = 0.5*(pfield->bcc(1,k,j-1,i) +
                                pfield->bcc(1,k,j,i));
  }

  for(int k=ks; k<=ke+1; k++)
  for(int j=js; j<=je;   j++)
  for(int i=is; i<=ie;   i++)
  {
    pfield->b.x3f(k,j,i) = 0.5*(pfield->bcc(2,k-1,j,i) +
                                pfield->bcc(2,k,j,i));
  }

#endif

  return;
}
