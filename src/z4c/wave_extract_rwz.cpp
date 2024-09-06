//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file wave_extract_rwz.cpp
//  \brief Implementation of metric-based extraction of Regge-Wheeler-Zerilli functions
//         Supports bitant symmetry

//TODO needs to be interfaced to the rest of GRA, in a way similarly to wave extract, AHF, ejecta, etc.
//TODO 2nd derivatives are not implemented, but they can be probably taken from interpolation (?)
//TODO add ADM integrals

#include <cstdio>
#include <stdexcept>
#include <sstream>
#include <unistd.h>
#include <cmath> // NAN

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#include "wave_extract_rwz.hpp"
#include "../globals.hpp"
#include "../parameter_input.hpp"
#include "../mesh/mesh.hpp"
#include "../utils/tensor.hpp"
//#include "../coordinates/coordinates.hpp"

//using namespace utils::tensor;

char const * const WaveExtractRWZ::ArealRadiusMethod[WaveExtractRWZ::NOptRadius] = {
  "areal", "areal_simple", "average_schw","schw_g00", "schw_gphph",
};

//----------------------------------------------------------------------------------------
//! \fn 
//  \brief class for RWZ waveform extraction
WaveExtractRWZ::WaveExtractRWZ(Mesh * pmesh, ParameterInput * pin, int n):
  pmesh(pmesh) {
  
  bitant = pin->GetOrAddBoolean("mesh", "bitant", false);
  verbose = pin->GetOrAddBoolan("rwz_extraction", "verbose", false);
  
  if (!(pin->GetBoolan("z4c", "store_metric_drvts"))) {
    std::stringstream msg;
    msg << "### FATAL ERROR in WaveExtractRWZ setup" << std::endl
	<< "Z4c 'store_metric_drvts' needs to be activated " << std::endl;
    ATHENA_ERROR(msg);
  }
  
  Nrad = n; 
  std::string n_str = std::to_string(n);
  
  lmax = pin->GetOrAddInteger("rwz_extraction", "lmax", 2);
  if ((lmax>8) || (lmax<2)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in WaveExtractRWZ setup" << std::endl
        << "lmax must be in [2,8] " << lmax << std::endl;
    ATHENA_ERROR(msg);
  }

  // Set method to compute areal radius
  string radius_method = pin->GetOrAddString("rwz_extraction", "method_areal_radius", "areal");
  int i;
  for (int i=0; i<NOptRadius; ++i) {
    if (radius_method==ArealRadiusMethod[i]) {
      method_areal_radius = i;
    }
  }
  if (i==NOptRadius) {
    std::stringstream msg;
    msg << "### FATAL ERROR in WaveExtractRWZ setup" << std::endl
        << "unknown method_areal_radius " << radius_method << std::endl;
    ATHENA_ERROR(msg);
  }
  
  // Get extraction radii
  std::string parname = "radius_" + n_str;
  Radius = pin->GetOrAddReal("rwz_extraction", parname, 10.0);

  // Center of the sphere
  parname = "center_x_";
  parname += n_str;
  center[0] = pin->GetOrAddReal("rwz_extraction", parname, 0.0);
  parname = "center_y_";
  parname += n_str;
  center[1] = pin->GetOrAddReal("rwz_extraction", parname, 0.0);
  parname = "center_z_";
  parname += n_str;
  center[2] = pin->GetOrAddReal("rwz_extraction", parname, 0.0);
  
  // (theta,phi) coordinate points 
  Ntheta = pin->GetOrAddInteger("rwz_extraction", "ntheta",60);
  Nphi = pin->GetOrAddInteger("rwz_extraction", "nphi",30);
  if ((Nphi+1)%2==0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in WaveExtractRWZ setup" << std::endl
        << "nphi must be even " << Nphi << std::endl;
    ATHENA_ERROR(msg);
  }

  th_grid.NewAthenaArray(Ntheta);
  for (int i = 0; i < Ntheta; ++i)
    th_grid(i) = coord_theta(i);

  ph_grid.NewAthenaArray(Nphi);
  for (int j = 0; j < Nphi; ++j)
    ph_grid(i) = coord_phi(j);

  //TODO Add more accurate integration schemes? 
  // Currently only Riemann sums: weights is not used.
  //string integral_method = pin->GetOrAddString("rwz_extraction", "method_integrals", "sum");
  //weights.NewAthenaArray(Ntheta,Nphi);
  //SetWeightsIntegral(integral_method);
  
  // Flag sphere points belonging to this rank
  havepoint.NewAthenaArray(Ntheta,Nphi);
 
  MeshBlock * pmb = pmesh->pblock;
  while (pmb != nullptr) {
    FlagSpherePointsContained(pmb);
    pmb = pmb->next; 
  }
  
  // 3+1 metric on the sphere
  gamma_dd.NewAthenaTensor(Ntheta,Nphi);
  dr_gamma_dd.NewAthenaTensor(Ntheta,Nphi);
  dot_gamma_dd.NewAthenaTensor(Ntheta,Nphi);

  beta_d.NewAthenaTensor(Ntheta,Nphi);
  dr_beta_d.NewAthenaTensor(Ntheta,Nphi);
  dot_beta_d.NewAthenaTensor(Ntheta,Nphi);

  beta2.NewAthenaTensor(Ntheta,Nphi);
  dr_beta2.NewAthenaTensor(Ntheta,Nphi);
  dot_beta2.NewAthenaTensor(Ntheta,Nphi);

  alpha.NewAthenaTensor(Ntheta,Nphi);
  dr_alpha.NewAthenaTensor(Ntheta,Nphi);
  dot_alpha.NewAthenaTensor(Ntheta,Nphi);

  // Background 2-metric
  g_dd.NewTensorPointwise();
  g_dr_dd.NewTensorPointwise();
  g_dot_dd.NewTensorPointwise();
  
  g_uu.NewTensorPointwise();
  g_dr_uu.NewTensorPointwise();
  g_dot_uu.NewTensorPointwise();
  
  Gamma_udd.NewTensorPointwise();
  Gamma_dyn_uddd.NewTensorPointwise();  
  
  // Number of spherical harmonics (with l = 2 ... lmax)
  lmpoints = MPoints(lmax); // lmax*(lmax + 2) - 3;
  
  // Spherical harmonics 
  Y.NewAthenaArray(Ntheta,Nphi,lmpoints,2);
  Yth.NewAthenaArray(Ntheta,Nphi,lmpoints,2);
  Yph.NewAthenaArray(Ntheta,Nphi,lmpoints,2);
  X.NewAthenaArray(Ntheta,Nphi,lmpoints,2);
  W.NewAthenaArray(Ntheta,Nphi,lmpoints,2);  
  
  ComputeSphericalHarmonics();

  // Allocate memory for reducing the multipoles
  // NVMultipoles complex multipole with lm indexes
  integrals_multipoles = new Real[2*NVMultipoles*lmpoints];

  // Even-parity Multipoles & dvrts
  h00.NewAthenaArray(lmpoints,2);
  h01.NewAthenaArray(lmpoints,2);
  h11.NewAthenaArray(lmpoints,2);
  h0.NewAthenaArray(lmpoints,2);
  h1.NewAthenaArray(lmpoints,2);
  G.NewAthenaArray(lmpoints,2);
  K.NewAthenaArray(lmpoints,2);

  h00_dr.NewAthenaArray(lmpoints,2);
  h01_dr.NewAthenaArray(lmpoints,2);
  h11_dr.NewAthenaArray(lmpoints,2);
  h0_dr.NewAthenaArray(lmpoints,2);
  h1_dr.NewAthenaArray(lmpoints,2);
  G_dr.NewAthenaArray(lmpoints,2);
  K_dr.NewAthenaArray(lmpoints,2);

  h00_dot.NewAthenaArray(lmpoints,2);
  h01_dot.NewAthenaArray(lmpoints,2);
  h11_dot.NewAthenaArray(lmpoints,2);
  h0_dot.NewAthenaArray(lmpoints,2);
  h1_dot.NewAthenaArray(lmpoints,2);
  G_dot.NewAthenaArray(lmpoints,2);
  K_dot.NewAthenaArray(lmpoints,2);
  
  // Odd-parity Multipoles & dvrts
  H0.NewAthenaArray(lmpoints,2);
  H1.NewAthenaArray(lmpoints,2);
  H.NewAthenaArray(lmpoints,2);

  H0_dr.NewAthenaArray(lmpoints,2);
  H1_dr.NewAthenaArray(lmpoints,2);
  H_dr.NewAthenaArray(lmpoints,2);

  H0_dot.NewAthenaArray(lmpoints,2);
  H1_dot.NewAthenaArray(lmpoints,2);
  H_dot.NewAthenaArray(lmpoints,2);

  // Gauge-invariant
  // NB two of these are stored as Tensors
  kappa_dd.NewTensorPointwise(lmpoints,2);
  kappa_d.NewTensorPointwise(lmpoints,2);
  kappa.NewAthenaArray(lmpoints,2);
  Tr_kappa_dd.NewAthenaArray(lmpoints,2);
  
  // Master functions multipoles
  Psie.NewAthenaArray(lmpoints,2);
  Psio.NewAthenaArray(lmpoints,2);
  Psie_dyn.NewAthenaArray(lmpoints,2);
  Psio_dyn.NewAthenaArray(lmpoints,2);
  Qplus.NewAthenaArray(lmpoints,2);
  Qstar.NewAthenaArray(lmpoints,2);
  
  // Set up stuff for output
#ifdef MPI_PARALLEL
  root = pin->GetOrAddInteger("rwz_extraction", "mpi_root", 0);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ioproc = (root == rank);
#else
  ioproc = true;
#endif

  outprec = pin->GetOrAddInteger("rwz_extraction", "output_digits", 6);
  
  if (ioproc) {
      // Baseline names
      ofbname[Iof_hlm] = pin->GetOrAddString("rwz_extraction", "filename_hlm", "wave_rwz");
      ofbname[Iof_Psie] = pin->GetOrAddString("rwz_extraction", "filename_psie", "wave_psie");
      ofbname[Iof_Psio] = pin->GetOrAddString("rwz_extraction", "filename_psio", "wave_psio");      
      // These are assigned, given the choice for Psie/o
      ofbname[Iof_Psie_dyn] = ofname[Iof_Psie];
      ofbname[Iof_Psie_dyn] += "_dyn";
      ofbname[Iof_Psio_dyn] = ofname[Iof_Psio];
      ofbname[Iof_Psio_dyn] += "_dyn";
      ofbname[Iof_Qplus] = pin->GetOrAddString("rwz_extraction", "filename_Qplus", "wave_Qplus");
      ofbname[Iof_Qstar] = pin->GetOrAddString("rwz_extraction", "filename_Qstar", "wave_Qstar");      
  }// if (ioproc)

}

//----------------------------------------------------------------------------------------

WaveExtractRWZ::~WaveExtractRWZ() {

  th_grid.DeleteAthenaArray();
  ph_grid.DeleteAthenaArray();
  //weights.DeleteAthenaArray();
  
  havepoint.DeleteAthenaArray();

  // Arrays on the sphere with tensor indexes 

  // 3+1 metric
  gamma_dd.DeleteAthenaTensor(); 
  dr_gamma_dd.DeleteAthenaTensor();
  dot_gamma_dd.DeleteAthenaTensor();

  beta_d.DeleteAthenaTensor();
  dr_beta_d.DeleteAthenaTensor();
  dot_beta_d.DeleteAthenaTensor();

  beta2.DeleteAthenaTensor();
  dr_beta2.DeleteAthenaTensor();
  dot_beta2.DeleteAthenaTensor();
  
  alpha.DeleteAthenaTensor();
  dr_alpha.DeleteAthenaTensor();
  dot_alpha.DeleteAthenaTensor();

  // 2-metric background (pointwise)
  g_dd.DeleteTensorPointwise();
  g_uu.DeleteTensorPointwise();
  g_dr_dd.DeleteTensorPointwise();
  g_dr_uu.DeleteTensorPointwise();
  g_dot_dd.DeleteTensorPointwise();
  g_dot_uu.DeleteTensorPointwise();
  Gamma_udd.DeleteTensorPointwise();
  Gamma_dyn_udd.DeleteTensorPointwise(); 
  
  // Arrays on the sphere with lm indexes (complex)

  // Spherical harmonics
  Y.DeleteAthenaArray();
  Yth.DeleteAthenaArray();
  Yph.DeleteAthenaArray();
  X.DeleteAthenaArray();
  W.DeleteAthenaArray();
   
  // Even-parity Multipoles & dvrts
  h00.DeleteAthenaArray();
  h01.DeleteAthenaArray();
  h11.DeleteAthenaArray();
  h0.DeleteAthenaArray();
  h1.DeleteAthenaArray();
  G.DeleteAthenaArray();
  K.DeleteAthenaArray();

  h00_dr.DeleteAthenaArray();
  h01_dr.DeleteAthenaArray();
  h11_dr.DeleteAthenaArray();
  h0_dr.DeleteAthenaArray();
  h1_dr.DeleteAthenaArray();
  G_dr.DeleteAthenaArray();
  K_dr.DeleteAthenaArray();

  h00_dot.DeleteAthenaArray();
  h01_dot.DeleteAthenaArray();
  h11_dot.DeleteAthenaArray();
  h0_dot.DeleteAthenaArray();
  h1_dot.DeleteAthenaArray();
  G_dot.DeleteAthenaArray();
  K_dot.DeleteAthenaArray();
  
  // Odd-parity Multipoles & dvrts
  H0.DeleteAthenaArray();
  H1.DeleteAthenaArray();
  H.DeleteAthenaArray();

  H0_dr.DeleteAthenaArray();
  H1_dr.DeleteAthenaArray();
  H_dr.DeleteAthenaArray();

  H0_dot.DeleteAthenaArray();
  H1_dot.DeleteAthenaArray();
  H_dot.DeleteAthenaArray();

  // Gauge-invariant
  kappa_dd.DeleteTensorPointwise();
  kappa_d.DeleteTensorPointwise();
  kappa.DeleteAthenaArray();
  Tr_kappa_dd.DeleteAthenaArray();
    
  // Master funs
  Psie.DeleteAthenaArray();
  Psio.DeleteAthenaArray();
  Psie_dyn.DeleteAthenaArray();
  Psio_dyn.DeleteAthenaArray();
  Qplus.DeleteAthenaArray();
  Qstar.DeleteAthenaArray();
  
  delete integrals_multipoles;

}

//----------------------------------------------------------------------------------------
//! \fn std::string WaveExtractRWZ::OutputFileName(std::string base)
//  \brief compute filenames from a basename and adding extraction radius index
std::string WaveExtractRWZ::OutputFileName(std::string base) {
  std::string fname = base +  "_r" + std::to_string(Nrad) + ".txt";
  return fname;
  //return move(fname);
}
 
//----------------------------------------------------------------------------------------
// \!fn WaveExtractRWZ::Write(int iter, Real time)
// \brief write output at given time and for given radius
//        output frequency is controlled by the RWZ trigger (see main.hpp)
void WaveExtractRWZ::Write(int iter, Real time) {  
  
  if (!ioproc) return;
  
  // The diagnostic file is special ...   
  ofname = OutputFileName(ofbname[Iof_diagnostic]);
  
  if (access(ofname.c_str(), F_OK) == 0) {
    // File exists
    outfile.open(ofname, std::ofstream::app);
    if (!outfile.is_open()) {
      std::stringstream msg;
      msg << "### FATAL ERROR in WaveExtract constructor" << std::endl;
      msg << "Could not open file '" << ofname << "' for writing!";
      throw std::runtime_error(msg.str().c_str());
    }
    outfile << iter << " "
	    << std::setprecision(outprec) << std::scientific << time << " "
	    << std::setprecision(outprec) << std::scientific << Schwarzschild_radius << " "
	    << std::setprecision(outprec) << std::scientific << Schwarzschild_mass << " "
	    << std::setprecision(outprec) << std::scientific << dt_rsch << " "
	    << std::setprecision(outprec) << std::scientific << norm_Delta_Gamma << " "
	    << std::setprecision(outprec) << std::scientific << norm_Tr_kappa_dd << " "
	    << std::endl;
    outfile.close();	 
  }
  
  else {
    // Create new file
    outfile.open(ofname, std::ofstream::out);
    if (!outfile.is_open()) {
      std::stringstream msg;
      msg << "### FATAL ERROR in WaveExtract constructor" << std::endl;
      msg << "Could not open file '" << ofname << "' for writing!";
      throw std::runtime_error(msg.str().c_str());
    }	  
    outfile << "# 1:iter " 
	    << "2:time "
	    << "3:SchwarzschildRradius "
      	    << "4:SchwarzschildMass "
	    << "5:SchwarzschildRadius_dot "
      	    << "6:NormDeltaGamma "
	    << "7:NormTracekappaAB "
	    << std::endl;
    outfile.close();       
  }
  
  // ... all other files contain complex wave multipoles as function of time
  std::vector<AthenaArray<Real>*> data;
  data.reserve(6);
  data.push_back(Psie);
  data.push_back(Psio);
  data.push_back(Psie_dyn);
  data.push_back(Psio_dyn);
  data.push_back(Qplus);
  data.push_back(Qstar);
  
  for (int i = Iof_monitor+1; i < Iof_idx_Num; ++i) {
    
    ofname = OutputFileName(ofbname[i]);	 
    if (access(ofname.c_str(), F_OK) == 0) {
      // File exists
      outfile.open(ofname, std::ofstream::app);
      if (!outfile.is_open()) {
	std::stringstream msg;
	msg << "### FATAL ERROR in WaveExtract constructor" << std::endl;
	msg << "Could not open file '" << ofname << "' for writing!";
	throw std::runtime_error(msg.str().c_str());
      }      

      if (i==Iof_hlm) {
	// This should be the last output since there is no storage for hlm:
	// they are computed normalizing the Psi. We also multiply by the coord. radius
	for (int l = 2; l <= lmax; ++l) {
	  const Real NRWZ = Radius * RWZnorm(l);
	  for (int m = -l; m <= l; ++m) {
	    const Real hlm_R = NRWZ *(Psie(lm,Re) + Psio(lm,Re));
	    const Real hlm_I = NRWZ *(Psie(lm,Im) + Psio(lm,Im));
	    outfile << iter << " "
		    << std::setprecision(outprec) << std::scientific << time <<  " "
		    << std::setprecision(outprec) << std::scientific << hlm_R << " "
		    << std::setprecision(outprec) << std::scientific << hlm_I << " "; 
	  }
	}
	outfile << std::endl;
      }      
      else {      
	for (int l = 2; l <= lmax; ++l) {
	  for (int m = -l; m <= l; ++m) {
	    outfile << iter << " "
		    << std::setprecision(outprec) << std::scientific << time <<  " "
		    << std::setprecision(outprec) << std::scientific << data(i-1)(lm,Re) << " "
		    << std::setprecision(outprec) << std::scientific << data(i-1)(lm,Im) << " "; 
	  }
	}
	outfile << std::endl;
      } // if i == Iof_hlm      

      outfile.close();      
    }
    
    else {
      // Create new file
      outfile.open(ofname, std::ofstream::out);
      if (!outfile.is_open()) {
	std::stringstream msg;
	msg << "### FATAL ERROR in WaveExtract constructor" << std::endl;
	msg << "Could not open file '" << ofname << "' for writing!";
	throw std::runtime_error(msg.str().c_str());
      }	  
      outfile << "# 1:iter 2:time";
      int idx = 3;
      for (int l = 2; l <= lmax; ++l) {
	for (int m = -l; m <= l; ++m) {
	  outfile << " " << idx++ << ":l=" << l << "-m=" << m << "-Re"
		  << " " << idx++ << ":l=" << l << "-m=" << m << "-Im";		  
	} // for l
      } // for m
      outfile << std::endl;
      outfile.close();
    }
    
  }// for i in Iof_*

  data.resize(0);  
  //data.clear();  
}

//----------------------------------------------------------------------------------------
// \!fn Real WaveExtractRWZ::coord_theta(const int i)
// \brief theta coordinate from index
Real WaveExtractRWZ::coord_theta(const int i) {
  Real dtheta = dth_grid();
  return dtheta*(0.5 + i);
}

//----------------------------------------------------------------------------------------
// \!fn Real WaveExtractRWZ::coord_phi(const int i)
// \brief phi coordinate from index
Real WaveExtractRWZ::coord_phi(const int j) {
  Real dphi = dph_grid();
  return dphi*(0.5 + j);
}

//----------------------------------------------------------------------------------------
// \!fn Real WaveExtractRWZ::dth_grid()
// \brief compute spacing dtheta 
Real WaveExtractRWZ::dth_grid() {
  return PI/Ntheta;
}

//----------------------------------------------------------------------------------------
// \!fn Real WaveExtractRWZ::dph_grid()
// \brief compute spacing dphi
Real WaveExtractRWZ::dph_grid() {
  return 2.0*PI/Nphi;
}

//----------------------------------------------------------------------------------------
// \!fn int WaveExtractRWZ::TPindex(const int i, const int j)
// \brief spherical grid single index (i,j) -> index
int WaveExtractRWZ::TPIndex(const int i, const int j) {
  return ((i)*Nphi + (j)); 
}

//----------------------------------------------------------------------------------------
// \!fn int WaveExtractRWZ::SetWeightsIntegral()
// \brief set the weights for the 2D integrals - UNFINISHED
void WaveExtractRWZ::SetWeightsIntegral(int method) {
  //weights.Fill(1.0); // method -> sum
}

//----------------------------------------------------------------------------------------
// \!fn void WaveExtractRWZ::FlagSpherePointsContained(MeshBlock * pmb)
// \brief flags points on the sphere belonging to this MB;
//        uses PointContained(x,y,z)) which works on physical points (not ghosts)
void WaveExtractRWZ::FlagSpherePointsContained(MeshBlock * pmb) {

  havepoint.ZeroClear(); // no points on this rank by default.
  
  // Center of the sphere
  const Real xc = center[0];
  const Real yc = center[1];
  const Real zc = center[2];
  
  for(int i=0; i<Ntheta; i++) {
    
    const Real theta = th_grid(i);
    const Real sinth = std::sin(theta);
    const Real costh = std::cos(theta);
    
    for(int j=0; j<Nphi; j++) {
      
      const Real phi   = ph_grid(j);
      const Real sinph = std::sin(phi);
      const Real cosph = std::cos(phi);
      
      // Global coordinates of the surface
      const Real x = xc + radius * sinth * cosph;
      const Real y = yc + radius * sinth * sinph;
      const Real z = zc + radius * costh;
      
      // Impose bitant symmetry below
      bool bitant_sym = ( bitant && z < 0 ) ? true : false;
      
      // Associate z -> -z if bitant
      if (bitant) z = std::abs(z);
      
      if (pmb->PointContained(x,y,z)) {
	// This sphere point is in this rank!
	havepoint(i,j) += 1; 
      } 
      
    }// for i theta
  }// for j phi

}

//----------------------------------------------------------------------------------------
// \!fn int WaveExtractRWZ::MPoints(const int l)
// \brief return number of multipoles up to given l
int WaveExtractRWZ::MPoints(const int l) {
  return (l*(l + 2) - 3);
}

//----------------------------------------------------------------------------------------
// \!fn int WaveExtractRWZ::MIndex(const int l, const int m)
// \brief return multipolar single index (l,m) -> k 
int WaveExtractRWZ::MIndex(const int l, const int m) {
  //return MPoints(l-1) + (m+l);
  return ((l-1)*((l-1)+2)-3) + (m+l);
}

//----------------------------------------------------------------------------------------
// \!fn Real WaveExtractRWZ::RWZnorm(const int l)
// \brief return RWZ normalization factor to strain multipoles
Real WaveExtractRWZ::RWZnorm(const int l) {
  return std::sqrt(Factorial(l+2)/Factorial(l-2));
}

//----------------------------------------------------------------------------------------
// \!fn Real WaveExtractRWZ::Factorial(const int n)
// \brief factorial function
static const double fact35[] = {
  1.,
  1.,
  2.,
  6.,
  24.,
  120.,
  720.,
  5040.,
  40320.,
  362880.,
  3628800., 
  39916800.,
  479001600.,
  6227020800.,
  87178291200.,
  1307674368000.,
  20922789888000.,
  355687428096000.,
  6402373705728000.,
  121645100408832000.,
  2432902008176640000.,
  51090942171709440000.,
  1124000727777607680000.,
  25852016738884976640000.,
  620448401733239439360000.,
  15511210043330985984000000.,
  403291461126605635584000000.,
  10888869450418352160768000000.,
  304888344611713860501504000000.,
  8841761993739701954543616000000.,
  265252859812191058636308480000000.,
  8222838654177922817725562880000000.,
  263130836933693530167218012160000000.,
  8683317618811886495518194401280000000.,
  295232799039604140847618609643520000000.,
  10333147966386144929666651337523200000000.};

Real WaveExtractRWZ::Factorial(const int n) {
  if (n < 0){
    std::stringstream msg;
    msg << "### FATAL ERROR in WaveExtractRWZ::Factorial" << std::endl
        << "factorial requires integer nonnegeative argument " << n << std::endl;
    ATHENA_ERROR(msg);
  } else if (n <= 35){
    return fact35[n];
  } else {
    return ((Real)n) * fact(n-1);
  }
}

//----------------------------------------------------------------------------------------
// \!fn Real WaveExtractRWZ::SphHarm_Plm(const int l, const int m, const Real x)
// \brief compute associated Legendre polynomial Plm(x).
//        m and l are integers satisfying 0 <= m <= l,
//        while x lies in the range -1 <= x <= 1
//

//TODO double check! Follow closely the cactus implementation
// https://bitbucket.org/einsteintoolkit/einsteinanalysis/src/b7d79de8b744005b5513ee6d76822ad7db33a4a8/Extract/src/D2_extract.F#lines-800

Real WaveExtractRWZ::SphHarm_Plm(const int l, const int m, const Real x) {

  Real pmm = 1.0;
    
  if (m>0) {
    Real somx2 = std::sqrt((1.0-x)*(1.0+x));
    Real fact = 1.0;
    for (int i=1; i<=m; ++i) {
      ppm  = -pmm*fact*somx2;
      fact += 2.0;
    }
  }
  else {
    std::stringstream msg;
    msg << "### FATAL ERROR in WaveExtractRWZ::SphHarm_Plm" << std::endl
        << "SphHarm_Plm requires nonnegeative m argument " << n << std::endl;
    ATHENA_ERROR(msg);
  }
  
  if (l == m) {
    return pmm;
  }
  
  Real pmmp1 = x*(2.0*m + 1.0)*pmm;
  
  if (l == (m+1)) { 
    return pmmp1;
  }
  
  for (int i=m+2; i<=l; ++i) {
    Real pll = ( x* ((Real)(2*i-1)) * pmmp1 - ((Real)(i+m-1)) * pmm )/((Real)(i-m));
    pmm = pmmp1;
    pmmp1 = pll;
  }
  return pll;    
}

//----------------------------------------------------------------------------------------
// \!fn void WaveExtractRWZ::SphHarm_Ylm(const int l, const int m, const Real theta, const Real phi,
//             				 Real * YlmR, Real * YlmI)
// \brief compute scalar spherical harmonics
//
//               a    ( 2 l + 1 (l-|m|)! )                      i m phi
//     Ylm = (-1) SQRT( ------- -------  ) P_l|m| (cos(theta)) e
//                    (   4 Pi  (l+|m|)! ) 
//
// where
//
//      a = m/2 (sign(m)+1)  

// TODO double check.  Follow closely the cactus implementation
// https://bitbucket.org/einsteintoolkit/einsteinanalysis/src/b7d79de8b744005b5513ee6d76822ad7db33a4a8/Extract/src/D2_extract.F#lines-729

void WaveExtractRWZ::SphHarm_Ylm(const int l, const int m, const Real theta, const Real phi,
				 Real * YlmR, Real * YlmI) {
  
  const int abs_m = std::abs(m);
  const Real fact_norm = Factorial(l+abs_m)/Factorial(l-abs_m);
    
  //TODO Old code, test and remove -----
  Real fac = 1.0;  
  for (int i=(l-abs_m+1); i<=(l+abs_m); ++i) {
    fac *= (Real)(i);
  }  
  assert(std::fabs(fac - fact_norm)>1e-10);
  //fac = 1.0/fac //divide once below
  // -------------
  
  const Real a = std::sqrt((Real)(2*l+1)/(4.0*PI*fact_norm));
  const int mfac = (m>0)? std::pow(-1.0,m) : 1.0; //FIXME: this is the original, but it should be:
  //const int mfac = (m==0)? 1.0 : std::pow(-1.0,m); 
  const Real Plm = mfac * a * SphHarm_Plm(l,abs_m,std::cos(theta));

  *YlmR = Plm * std::cos((Real)(m)*phi);
  *YlmI = Plm * std::sin((Real)(m)*phi);
  
}

//----------------------------------------------------------------------------------------
// \!fn void WaveExtractRWZ:::SphHarm_Ylm_a(const int l, const int m, const Real theta, const Real phi,
//       				    Real * YthR, Real * YthI, Real * YphR, Real * YphI, 
//             				    Real * XR, Real * XI, Real * WR, Real * WI)
// \brief compute vector spherical harmonics basis and functions Xlm and Wlm
//

//TODO DOUBLE check! e.g. conventions for X = 2 Y4  and W = Y3 ! NB Follow closely the cactus implementation
// https://bitbucket.org/einsteintoolkit/einsteinanalysis/src/b7d79de8b744005b5513ee6d76822ad7db33a4a8/Extract/src/D2_extract.F#lines-871

void WaveExtractRWZ::SphHarm_Ylm_a(const int l_, const int m_, const Real theta, const Real phi,
				   Real * YthR, Real * YthI, Real * YphR, Real * YphI,
				   Real * XR, Real * XI, Real * WR, Real * WI) {
  
  const Real l = (Real)l_;
  const Real m = (Real)m_;

  const Real div_sin_theta = 1.0/(std::sin(theta));
  const Real cot_theta = std::cos(theta) * div_sin_theta;
  
  const Real a = -(l+1.0) * cot_theta;
  const Real b = std::sqrt((SQR(l+1.0)-SQR(m))*(l+0.5)/(l+1.5)) * div_sin_theta;

  Real YR,YI;
  SphHarm_Ylm_a(l+1,m,theta,phi,&YR,&YI);
  
  Real YplusR,YplusI;
  SphHarm_Ylm_a(l+1,m,theta,phi,&YplusR,&YplusI);

  const Real _YthR = a * YR + b * YplusR;
  const Real _YthI = a * YI + b * YplusI;

  const Real c = - 2.0*cot_theta;
  const Real d = (2.0*SQR(m*div_sin_theta) - l*(l+1.0));
  
  *YthR = _YthR;
  *YthI = _YthI;
  
  *YphR = - m * YR;
  *YphI =   m * YI;

  *WR =  c * YthR + d * YR;
  *WI =  c * YthI + d * YI;

  *XR = 2.0 * m * (cot_theta*YI - _YthI)
  *XI = 2.0 * m * (_YthR - cot_theta*YR)    
}

//----------------------------------------------------------------------------------------
// \!fn void WaveExtractRWZ::ComputeSphericalHarmonics()
// \brief compute spherical harmonics on the (theta,phi) grid
void WaveExtractRWZ::ComputeSphericalHarmonics() {

  Y.ZeroClear();
  Yth.ZeroClear();
  Yph.ZeroClear();
  X.ZeroClear();
  W.ZeroClear();  

  Real YlmR, YlmI;
  Real Ylm_thR, Ylm_thI;
  Real Ylm_phR, Ylm_phI;
  Real XlmR, XlmI;
  Real WlmR, WlmI;
      
  for(int i=0; i<Ntheta; ++i) {
    
    const Real theta = th_grid(i);
    const Real sinth = std::sin(theta);
    const Real costh = std::cos(theta);
    
    for(int j=0; j<Nphi; ++j) {
      
      const Real phi   = ph_grid(j);
      const Real sinph = std::sin(phi);
      const Real cosph = std::cos(phi);

      for(int l=2; l<=lmax; ++l) {
	for(int m=-l; m<=l; ++m) {
	  const int lm = MIndex(l,m);
	  
	  SphHarm_Ylm(l,m, theta,phi, &YlmR, &YlmI);

	  Y(i,j,lm,Re) = YlmR;
	  Y(i,j,lm,Im) = YlmI;

	  SphHarm_Ylm_a(l,m, theta,phi,
			&Ylm_thR, &Ylm_thI,
			&Ylm_phR, &Ylm_phI,
			&XlmR, &XlmI,
			&WlmR, &WlmI);

	  Yth(i,j,lm,Re) = Ylm_thR;
	  Yth(i,j,lm,Im) = Ylm_thI;
	  
	  Yph(i,j,lm,Re) = Ylm_phR;
	  Yph(i,j,lm,Im) = Ylm_phI;
	  
	  X(i,j,lm,Re) = XlmR;
	  X(i,j,lm,Im) = XlmI;

	  W(i,j,lm,Re) = WlmR;
	  W(i,j,lm,Im) = WlmI;
	  
	}// for m
      }// for m
      
    }// for j ph    
  }// for i th
    
}

//----------------------------------------------------------------------------------------
// \!fn void WaveExtractRWZ::MetricToSphere()
// \brief run over MBs of this rank and get the ADM metric in spherical coordinates on the sphere
void WaveExtractRWZ::MetricToSphere() {
 
  MeshBlock * pmb = pmesh->pblock;
  while (pmb != nullptr) {
    InterpMetricToSphere(pmb);
    pmb = pmb->next; 
  }
  
}

//----------------------------------------------------------------------------------------
// \!fn void WaveExtractRWZ::InterpMetricToSphere(MeshBlock * pmb)
// \brief interpolate the ADM metric and its drvts on the sphere
//        transform Cartesian to spherical coordinates
//
// This assumes there is a special storage with the spatial drvts of
// ADM metric and lapse and shift
// These derivatives are computed elsewhere, when the z4c parameter
// "store_metric_drvts" is turned on

//TODO 2nd drvts are missing, but they can be taken from the interpolation.

void WaveExtractRWZ::InterpMetricToSphere(MeshBlock * pmb)
{
  Z4c *pz4c = pmb->pz4c;

  // 3+1 metric 
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> adm_g_dd;      
  adm_g_dd.InitWithShallowSlice(pz4c->storage.adm, Z4c::I_ADM_gxx);

  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> adm_beta_u;    
  adm_g_dd.InitWithShallowSlice(pz4c->storage.Z4c, Z4c::I_Z4c_betax);
  
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> adm_alpha;    
  adm_g_dd.InitWithShallowSlice(pz4c->storage.Z4c, Z4c::I_Z4c_alpha);

  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 3> adm_dg_ddd;      
  adm_dg_dd.InitWithShallowSlice(pz4c->storage.aux, Z4c::I_AUX_dgxx_x); 

  AthenaTensor<Real, TensorSymm::NONE, NDIM, 2> adm_dbeta_du; // beta^j,i = d/dx^i beta^j
  adm_dbeta_du.InitWithShallowSlice(pz4c->storage.aux, Z4c::I_AUX_dbetax_x); 

  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> adm_dalpha_d;      
  adm_dalpha_d.InitWithShallowSlice(pz4c->storage.aux, Z4c::I_AUX_dalpha_x); 
  
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 2> adm_beta_dot_u;      
  adm_beta_dot.InitWithShallowSlice(pz4c->storage.rhs, Z4c::I_Z4c_betax);
  
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 2> adm_alpha_dot;      
  adm_alpha_dot.InitWithShallowSlice(pz4c->storage.rhs, Z4c::I_Z4c_alpha);

  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> adm_g_dot_dd;      
  adm_g_dot_dd.InitWithShallowSlice(pz4c->storage.adm, Z4c::I_ADM_Kxx);

  // adm_g_dot_dd stores Kab, get the time drvt
  // of the ADM metric from Kab, lapse and shift
  //FIXME Lie drvt
  ILOOP3(k,j,i) {
    for(int a = 0; a < NDIM; ++a)
      for(int b = a; b < NDIM; ++b) {
	const Real Kab = adm_g_dot_dd(a,b,k,j,i);
	adm_g_dot_dd(a,b,k,j,i) = - 2.0*z4c_alpha(k,j,i)*Kab
	  + z4c_dbeta_dd(a,b,k,j,i) + z4c_dbeta_dd(b,a,k,j,i);
      }
  }
  
  // Center of the sphere
  const Real xc = center[0];
  const Real yc = center[1];
  const Real zc = center[2];

  const Real r = Radius;
  const Real div_r = 1.0/(Radius);     
  Real n[3];
  n[0] = (x-center[0])*div_r;
  n[1] = (y-center[1])*div_r;
  n[2] = (z-center[2])*div_r;
  
  // For interp
  LagrangeInterpND<metric_interp_order, 3> * pinterp3 = nullptr;
  Real origin[NDIM];
  Real delta[NDIM];
  int size[NDIM];
  Real coord[NDIM];

  origin[0] = pz4c->mbi.x1(0);
  size[0]   = pz4c->mbi.nn1;
  delta[0]  = pz4c->mbi.dx1(0);
  
  origin[1] = pz4c->mbi.x2(0);
  size[1]   = pz4c->mbi.nn2;
  delta[1]  = pz4c->mbi.dx2(0);
  
  origin[2] = pz4c->mbi.x3(0);
  size[2]   = pz4c->mbi.nn3;
  delta[2]  = pz4c->mbi.dx3(0);
  
  // Pointwise tensors: Cartesian components
  TensorPointwise<Real, TensorSymm::SYM2, 0, 2> Cgamma_dd;
  TensorPointwise<Real, TensorSymm::NONE, 0, 1> Cbeta_u;
  TensorPointwise<Real, TensorSymm::NONE, 0, 1> Cbeta_d;
  TensorPointwise<Real, TensorSymm::NONE, 0, 0> Calpha;

  TensorPointwise<Real, TensorSymm::SYM2, 0, 3> Cgamma_der_ddd;
  TensorPointwise<Real, TensorSymm::NONE, 0, 2> Cbeta_der_ud;
  TensorPointwise<Real, TensorSymm::NONE, 0, 1> Calpha_der_d;

  TensorPointwise<Real, TensorSymm::SYM2, 0, 2> Cgamma_dot_dd;
  TensorPointwise<Real, TensorSymm::NONE, 0, 2> Cbeta_dot_u;
  TensorPointwise<Real, TensorSymm::NONE, 0, 1> Calpha_dot_d;

  TensorPointwise<Real, TensorSymm::SYM2, 0, 2> dr_Cgamma_dd; // Radial drvts of Cartesian comp.
  TensorPointwise<Real, TensorSymm::NONE, 0, 1> dr_Cbeta_u;
  TensorPointwise<Real, TensorSymm::NONE, 0, 1> dr_Cbeta_d;  
  
  Cgamma_dd.NewTensorPointwise();
  Cbeta_u.NewTensorPointwise();
  Cbeta_d.NewTensorPointwise();
  Calpha.NewTensorPointwise();
  
  Cgamma_der_ddd.NewTensorPointwise();
  Cbeta_der_ud.NewTensorPointwise();
  Calpha_der_d.NewTensorPointwise(); 

  Cgamma_dot_ddd.NewTensorPointwise();
  Cbeta_dot_ud.NewTensorPointwise();
  Calpha_dot_d.NewTensorPointwise(); 

  dr_Cgamma_dd.NewTensorPointwise();
  dr_Cbeta_u.NewTensorPointwise();
  dr_Cbeta_d.NewTensorPointwise();
    
  // Cartesian-to-Spherical Jacobian
  TensorPointwise<Real, TensorSymm::NONE, 0, 2> Jac;
  Jac.NewTensorPointwise();
  
  
  
  for (int i=0; i<Ntheta; i++) {
    
    const Real theta = th_grid(i);
    const Real sinth = std::sin(theta);
    const Real costh = std::cos(theta);
    const Real sinth2 = SQR(sinth);
    const Real costh2 = SQR(costh);
	  
    for(int j=0; j<Nphi; j++){

      if (!havepoint(i,j)) continue;
      
      const Real phi   = ph_grid(j);
      const Real sinph = std::sin(phi);
      const Real cosph = std::cos(phi);
      const Real sinph2 = SQR(sinph);
      const Real cosph2 = SQR(costh);
      
      // Global coordinates of the surface
      const Real x = xc + Radius * sinth * cosph;
      const Real y = yc + Radius * sinth * sinph;
      const Real z = zc + Radius * costh;
      
      coord[0]  = x;
      coord[1]  = y;
      coord[2]  = z;
      
      // Impose bitant symmetry below
      bool bitant_sym = ( bitant && z < 0 ) ? true : false;
      // Associate z -> -z if bitant
      if (bitant) z = std::abs(z);


      
      // Interpolate Cartesian components at point (theta_i,phi_j)
      // ----------------------------------------

      pinterp3 = new LagrangeInterpND<metric_interp_order, 3>(origin, delta, size, coord);

      // 3-metric
      // With bitant wrt z=0, pick a (-) sign every time a z component is encountered.
      for(int a = 0; a < NDIM; ++a)
      for(int b = a; b < NDIM; ++b) {
        int bitant_z_fac = 1;
        if (bitant_sym) {
          if (a == 2) bitant_z_fac *= -1;
          if (b == 2) bitant_z_fac *= -1;
        }
        Cgamma_dd(a,b) = pinterp3->eval(&(adm_g_dd(a,b,0,0,0)))*bitant_z_fac;
	Cgamma_dot_dd(a,b) = pinterp3->eval(&(adm_g_dot_dd(a,b,0,0,0)))*bitant_z_fac;
      }

      // Shift (up)
      for(int a = 0; a < NDIM; ++a) {
        int bitant_z_fac = 1;
        if ((bitant_sym) && (a == 2)) bitant_z_fac *= -1;
        Cbeta_u(a) = pinterp3->eval(&(adm_beta_u(a,0,0,0)))*bitant_z_fac;
	Cbeta_dot_u(a) = pinterp3->eval(&(adm_beta_dot_u(a,0,0,0)))*bitant_z_fac;
      }

      // lapse 
      Calpha() = pinterp3->eval(&(adm_alpha(0,0,0)));
      Calpha_dot() = pinterp3->eval(&(adm_alpha_dot(0,0,0)));
      
      // 3-metric spatial drvts
      // NB g_ddd(a,b,c) means d/dx^c g_{ab}
      for(int a = 0; a < NDIM; ++a)
	for(int b = a; b < NDIM; ++b)
	  for(int c = 0; c < NDIM; ++c) {
	    int bitant_z_fac = 1;
	    if (bitant_sym) {
	      if (a == 2) bitant_z_fac *= -1;
	      if (b == 2) bitant_z_fac *= -1;
	      if (c == 2) bitant_z_fac *= -1;
	    }
	    Cgamma_der_ddd(c,a,b) = pinterp3->eval(&(adm_dg_ddd(c,a,b,0,0,0)))*bitant_z_fac;
	  }
      
      // shift (up) spatial drvts
      for(int a = 0; a < NDIM; ++a) 
	for(int b = 0; b < NDIM; ++b) {
	  int bitant_z_fac = 1;
	  if (bitant_sym) {
	    if (a == 2) bitant_z_fac *= -1;
	    if (b == 2) bitant_z_fac *= -1;
	  }
	  Cbeta_der_ud(a,b) = pinterp3->eval(&(adm_dbeta_du(a,b,0,0,0)))*bitant_z_fac;
	}      
      
      // lapse spatial drvts
      for(int a = 0; a < NDIM; ++a) {
        int bitant_z_fac = 1;
        if ((bitant_sym) && (a == 2)) bitant_z_fac *= -1;
	Calpha_der_d(a) = pinterp3->eval(&(adm_alpha_d(a,0,0,0)));
      }
            
      delete pinterp3;
      

      
      // Auxiliary Cartesian tensor components
      // ----------------------------------------
      
      // Shift (down) 
      for(int a = 0; a < NDIM; ++a) {
	Cbeta_d(a) = 0.0;
	for(int b = 0; b < NDIM; ++b)  {
	  Cbeta_d(a) += Cgamma_dd(a,b) * Cbeta_u(b);
	}
      }      

      // Shift (down) spatial drvts
      for(int a = 0; a < NDIM; ++a) 
	for(int b = 0; b < NDIM; ++b)  {
	  Cbeta_dd(a,b) = 0.0;
	  for(int c = 0; c < NDIM; ++c)  {
	    Cbeta_dd(a,b) += Cgamma_ddd(a,c,b) * Cbeta_du(c,b) + Cgamma_dd(a,c) * Cbeta_du(c,b);
	  }
	}      

      // Radial drvts of Cartesian gamma_{ij} 
      for (int a = 0; a < NDIM; ++a) 
	for (int b = 0; b < NDIM; ++b) {
	  dr_Cgamma_dd(a,b) = 0.0;
	  for (int c = 0; c < NDIM; ++c)
	    dr_Cgamma_dd(a,b) += n[c] * Cgamma_der_ddd(c,a,b);
	}

      // Radial drvts of beta^i 
      for (int a = 0; a < NDIM; ++a) {
	dr_Cbeta_u(a) = 0.0;    
	for (int b = 0; b < 3; ++b) 
	  dr_Cbeta_u(a) += n[b] * Cbeta_der_ud(b,a); 
      }
      
      // Radial drvts of beta_i
      for (int a = 0; a < NDIM; ++a) {
	dr_Cbeta_d(a) = 0.0;    
	for (int b = 0; b < NDIM; ++b) {
	  dr_Cbeta_d(a) += dr_Cgamma_dd(a,b) * Cbeta_u(b) + Cgamma(a,b) * dr_Cbeta_u(b);
	}
      }


      
      // Lapse & drvts
      // ----------------------------------------      
      
      // lapse & time drvts
      alpha(i,j) = Calpha();
      dot_alpha(i,j) = Calpha_dot();

      // Radial drvts of lapse
      dr_alpha(i,j) = 0.0;
      for (int a = 0; a < NDIM; ++a) {
	dr_alpha(i,j) += n[a] * Calpha_der_d[a];
      }


      
      // beta^2 & drvts
      // ----------------------------------------      
      
      beta2(i,j) = 0.0;
      for (int a=0; a<NDIM; ++a)
	beta2(i,j) += Cbeta_u(a)*Cbeta_d(a);
      
      dot_beta2(i,j) = 0.0;
      for (int a=0; a<3; ++a)
	for (int b=0; b<3; ++b) {
	  dot_beta2(i,j) +=
	    Cgamma_dot_dd(a,b) * Cbeta_u(a) * Cbeta_u(b)
	    Cgamma_dd(a,b) * Cbeta_dot_u(a) * Cbeta_u(b)
	    Cgamma_dd(a,b) * Cbeta_u(a,i,j) * Cbeta_dot_u(b);
	}
      
      dr_beta2(i,j) = 0.0;
      for (int a=0; a<NDIM; ++a)
	for (int b=0; b<NDIM; ++b) {
	  dr_beta2(i,j) +=
	    dr_Cgamma_dd(a,b) * Cbeta_u(a) * Cbeta_u(b)
	    Cgamma_dd(a,b) * dr_beta_u(a) * beta_u(b,)
	    Cgamma_dd(a,b) * beta_u(a) * dr_beta_u(b);
	}
      

      
            
      // Transform tensors to spherical coordinates 
      // ----------------------------------------
      
      // Jacobian
      Jac(0,0) = sinth * cosph;
      Jac(0,1) = r * costh * cosph;
      Jac(0,2) = - r * sinth * sinph;
      Jac(1,0) = sinth * sinph;
      Jac(1,1) = r * costh * sinph;
      Jac(1,2) = r * sinth * cosph;
      Jac(2,0) = costh;
      Jac(2,1) = - r* sinth;
      Jac(2,2) = 0.0;

      // 3-metric & time drvts  
      for (int a = 0; a < NDIM; ++a)
	for (int b = 0; b < NDIM; ++b) {
	  gamma_dd(a,b, i,j) = 0.0;
	  dot_gamma_dd(a,b, i,j) = 0.0;
	  for (int c = 0; c < NDIM; ++c)
	    for (int d = 0; d < NDIM; ++d) {
	      //TODO check indexes	  
	      gamma_dd(a,b, i,j) += Jac(a,c) * Cgamma_dd(c,d) *  Jac(d,b);
	      dot_gamma_dd(a,b) += Jac(a,c) * Cgamma_dot_dd(c,d) *  Jac(d,b);
	    }
	}

      // shift down (beta_i) & time drvts
      for (int a = 0; a < NDIM; ++a) {
	beta_d(a, i,j) = 0.0;
	beta_dot_u(a, i.j) = 0.0;
	for (int c = 0; c < NDIM; ++c) {
	  //TODO check indexes
	  beta_d(a, i,j) += Jac(a,c) * Cbeta_d(c);
	  beta_dot_u(a, i,j) += Jac(a,c) * Cbeta_dot_u(c);
	}
      }
      
      // Spherical components of radial drvts of gamma_{ij}
      //FIXME: code the following cleanly (with appropriate Jacobians)
      dr_gamma_dd(0,0, i,j) =
	sinth2 * ( cosph2 * dr_Cgamma_dd(0,0)
		   + sinph2 * dr_Cgamma_dd(1,1) )
	+ costh2 * dr_Cgamma_dd(2,2)
	+ 2.0 * sinth * ( sinth * cosph * sinph * dr_Cgamma_dd(0,1)
			  + costh * cosph * dr_Cgamma_dd(0,2)
			  + costh * sinph * dr_Cgamma_dd(1,2) );
      
      dr_gamma_dd(0,1, i,j) = div_r * gamma_dd(0,1, i,j)
	+ r * ( sinth * cosph2 * costh * dr_Cgamma_dd(0,0)
		+ 2.0 * sinth * costh * sinph * cosph * dr_Cgamma_dd(0,1)
		+ cosph * (costh2-sinth2) * dr_Cgamma_dd(0,2)
		+ sinth * sinph2 * costh * dr_Cgamma_dd(1,1)
		+ sinph * (costh2-sinth2) * dr_Cgamma_dd(1,2)
		- costh * sinth * dr_Cgamma_dd(2,2) );
      
      dr_gamma_dd(0,2, i,j) = div_r * gamma_dd(0,2, i,j)
	+ r * sinth * ( - sinth * sinph * cosph * dr_Cgamma_dd(0,0)
			- sinth * (sinph2-cosph2) * dr_Cgamma_dd(0,1) 
			- sinph * costh * dr_Cgamma_dd(0,2) 
			+ sinth * sinph * cosph * dr_Cgamma_dd(1,1)
			+ costh * cosph * * dr_Cgamma_dd(1,2) );
      
      dr_gamma_dd(1,1, i,j) = 2.0 * div_r * gamma_dd(1,1, i,j)
	+ r2 * ( costh2 * ( cosph2 * dr_Cgamma_dd(0,0)
			    + sinph2 * dr_Cgamma_dd(1,1) )
		 + sinth2 * dr_Cgamma_dd(2,2)
		 + 2.0 * costh * ( costh * sinph * cosph * dr_Cgamma_dd(0,1)
				   sinth * cosph * dr_Cgamma_dd(0,2)
				   sinth * sinph * dr_Cgamma_dd(1,2) ) );
      
      dr_gamma_dd(1,2, i,j) = 2.0 * div_r *  gamma_dd(1,2, i,j)
	+ r2 * sinth * ( - cosph * sinph * costh * dr_Cgamma_dd(0,0)
			 - costh * (sinph2-cosph2) * dr_Cgamma_dd(0,1)
			 + sinth * sinph * dr_Cgamma_dd(0,2)
			 + cosph * sinph * costh * dr_Cgamma_dd(1,1)
			 - sinth * cosph * dr_Cgamma_dd(1,2) );
      
      dr_gamma_dd(2,2, i,j) = 2.0 * div_r *  gamma_dd(1,2, i,j)
	+ r2 * sinth2 * ( sinph2 * dr_Cgamma_dd(0,0)
			  - 2.0 * cosph * sinph * dr_Cgamma_dd(0,1)
			  + cosph2 * dr_Cgamma_dd(1,1) );
      
      // Spherical components of radial drvts of beta_i
      //FIXME: code this cleanly
      dr_beta_d(0,i,j) = sinth * ( cosph * dr_Cbeta_d(0) + sinph * dr_Cbeta_d(1) )
	+ costh * dr_Cbeta_d(2);
      dr_beta_d(1,i,j) = div_r * beta_d(1,i,j) + r*( costh * ( dr_Cbeta_d(0) + sinph * dr_Cbeta_d(1) )
						     - sinth * dr_Cbeta_d(2) );
      dr_beta_d(2,i,j) = div_r * beta_d(2,i,j) + r * sinth * ( cosph * dr_Cbeta_d(1)
							       - sinth * dr_Cbeta_d(0) );

      
    } // phi loop
  } // theta loop
  
  adm_g_dd.DeleteTensorPointwise();
  adm_beta_u.DeleteTensorPointwise();
  adm_alpha.DeleteTensorPointwise();
  
  adm_dg_ddd.DeleteTensorPointwise();
  adm_dbeta_du.DeleteTensorPointwise();
  adm_dalpha_d.DeleteTensorPointwise();

  adm_g_dot_dd.DeleteTensorPointwise();
  adm_beta_dot_u.DeleteTensorPointwise();
  adm_alpha_dot.DeleteTensorPointwise();
    
  Jac.DeleteTensorPointwise();

  Cgamma_dd.DeleteTensorPointwise();
  Cbeta_u.DeleteTensorPointwise();
  Cbeta_d.DeleteTensorPointwise();
  Calpha.DeleteTensorPointwise();
  
  Cgamma_der_ddd.DeleteTensorPointwise();
  Cbeta_der_ud.DeleteTensorPointwise();
  Calpha_der_d.DeleteTensorPointwise(); 

  Cgamma_dot_ddd.DeleteTensorPointwise();
  Cbeta_dot_u.DeleteTensorPointwise();
  Calpha_dot_d.DeleteTensorPointwise(); 

  dr_Cgamma_dd.DeleteTensorPointwise();
  dr_Cbeta_u.DeleteTensorPointwise();
  dr_Cbeta_d.DeleteTensorPointwise();
  
}

//----------------------------------------------------------------------------------------
// \!fn void WaveExtractRWZ::TransformMetricCarToSph(...)
// \brief transform metric on the sphere from (x,y,z) to (R,theta,phi) at a (theta,phi) point
//void WaveExtractRWZ::TransformMetricCarToSph(x,y,z, theta,phi) {
//
//}

//----------------------------------------------------------------------------------------
// \!fn void WaveExtractRWZ::BackgroundReduce()
// \brief compute the background spherical metric, areal radius, mass, etc
//        performs local sums and then MPI reduce
void WaveExtractRWZ::BackgroundReduce() {

  const Real dthdph = dth_grid() * dph_grid();  
  
  // Zeros the integrals
  for (int i=0; i<NVBackground; i++) 
    integrals_background[i] = 0.0;
     
  for (int i=0; i<Ntheta; i++) {
    
    const Real theta = th_grid(i);
    const Real sinth = std::sin(theta);
    const Real costh = std::cos(theta);
    const Real sinth2 = SQR(sinth);
    const Real costh2 = SQR(costh);
    const Real div_sinth = 1.0/sinth;
    const Real div_sinth2 = SQR(div_sinth);
    
    const Real vol = dthdph * sinth;
    
    for(int j=0; j<Nphi; j++){

      if (!havepoint(i,j)) continue;
      
      const Real phi   = ph_grid(j);
      const Real sinph = std::sin(phi);
      const Real cosph = std::cos(phi);
      const Real sinph2 = SQR(sinph);
      const Real cosph2 = SQR(cosph);

      Real int_r2 = 0.0;
      Real int_drsch_dri = 0.0;
      Real int_d2rsch_dri2 = 0.0;
      Real int_r2dot = 0.0;
      Real int_drsch_dri_dot = 0.0; //TODO
      
      // NB These integrals will be all normalized with 1/(4 Pi)
      
      if (method_areal_radius==areal) {
	
	const Real aux_r2 = std::sqrt(gamma_dd(1,1,i,j)*gamma_dd(2,2,i,j) - SQR(gamma_dd(1,0,i,j)));
	const Real div_ aux_r2 = 1.0/ aux_r2;
	const Real aux_r2_d =
	  dr_gamma_dd(1,1,i,j) * gamma_dd(2,2,i,j) 
	  + gamma_dd(1,1,i,j) * dr_gamma_dd(2,2,i,j)
	  - 2.0*gamma_dd(1,0,i,j) * dr_gamma_dd(1,0,i,j);
	const Real aux_r2_d2 =
	  dr2_gamma_dd(1,1,i,j) * gamma_dd(2,2,i,j) 
	  + 2.0*dr_gamma_dd(1,1,i,j) * dr_gamma_dd(2,2,i,j)
	  + gamma_dd(1,1,i,j) * dr2_gamma_dd(2,2,i,j) 
	  - 2.0*SQR(dr_gamma_dd(1,0,i,j))
	  - 2.0*gamma_dd(1,0,i,j) * dr2_gamma_dd(1,0,i,j);
	const Real aux_r2_dot =
	  dot_gamma_dd(1,1,i,j) * gamma_dd(2,2,i,j) 
	  + gamma_dd(1,1,i,j) * dot_gamma_dd(2,2,i,j)
	  - 2.0*gamma_dd(1,0,i,j) * dot_gamma_dd(1,0,i,j);
	
	int_r2 = aux_r2 * dthpdh;
	int_drsch_dri = 0.5*aux_r2_d * div_aux_r2 * dthpdh;
	int_d2rsch_dri2 = ( 0.5*aux_r2_d2 * div_aux_r2
			    - 0.25*SQR(aux_r2_d) * std::pow(div_aux_r2,3) ) * dthpdh;	

	int_r2dot = aux_r2_dot * div_aux_r2 * dthpdh;
	
      } else if (method_areal_radius==areal_simple) {

	const Real aux_r2 = std::sqrt(gamma_dd(1,1,i,j)*gamma_dd(2,2,i,j));
	const Real div_ aux_r2 = 1.0/ aux_r2;
	const Real aux_r2_d =
	  dr_gamma_dd(1,1,i,j)*gamma_dd(2,2,i,j) 
	  + gamma_dd(1,1,i,j) * dr_gamma_dd(2,2,i,j)
	  const Real aux_r2_d2 =
	  dr2_gamma_dd(1,1,i,j) * gamma_dd(2,2,i,j) 
	  + 2.0*dr_gamma_dd(1,1,i,j) * dr_gamma_dd(2,2,i,j)
	  + gamma_dd(1,1,i,j) * dr2_gamma_dd(2,2,i,j);
	const Real aux_r2_dot =
	  dot_gamma_dd(1,1,i,j) * gamma_dd(2,2,i,j) 
	  + gamma_dd(1,1,i,j) * dot_gamma_dd(2,2,i,j);
	
	int_r2 = aux_r2 * dthpdh;
	int_drsch_dri = 0.5*aux_r2_d * div_aux_r2 * dthpdh;
	int_d2rsch_dri2 = ( 0.5*aux_r2_d2 * div_aux_r2
			    - 0.25*SQR(aux_r2_d) * std::pow(div_aux_r2,3) ) * dthpdh;	

	int_r2dot = aux_r2_dot * div_aux_r2 * dthpdh;
	
      } else if (method_areal_radius==average_schw) {

	int_r2 = 0.5*( gamma_dd(1,1,i,j) + gamma_dd(2,2,i,j)*div_sinth2 ) * vol;
	int_drsch_dri = 0.25*( dr_gamma_dd(1,1,i,j) + dr_gamma_dd(2,2,i,j)*div_sinth2 ) * vol;
	int_d2rsch_dri2 = 0.25*( dr2_gamma_dd(1,1,i,j) + dr2_gamma_dd(2,2,i,j)*div_sinth2 ) * vol;

	int_r2dot = 0.25*( dot_gamma_dd(1,1,i,j) + dot_gamma_dd(2,2,i,j)*div_sinth2 ) * vol; 	
	
      } else if (method_areal_radius==schw_gthth) {

	int_r2 = gamma_dd(1,1,i,j) * vol;
	int_drsch_dri = dr_gamma_dd(1,1,i,j) * vol;
	int_d2rsch_dri2 = dr2_gamma_dd(1,1,i,j) * vol;

	int_r2dot = dot_gamma_dd(1,1,i,j) * vol;
	
      } else if (method_areal_radius==schw_gphph) {

	int_r2 = gamma_dd(2,2,i,j) * div_sinth2 * vol;
	int_drsch_dri = dr_gamma_dd(2,2,i,j) * div_sinth2 * vol;
	int_d2rsch_dri2 = dr2_gamma_dd(2,2,i,j) * div_sinth2 * vol;

	int_r2dot = dot_gamma_dd(2,2,i,j) * div_sinth2 * vol;
	
      }
	           
      // Local sums
      // ----------

      // Schwarzschild radius & Jacobians
      integrals_background[Irsch2] += int_r2;
      integrals_background[Idrsch_dri] += int_drsch_dri;
      integrals_background[Id2rsch_dri2] += int_d2rsch_dri2;
      
      integrals_background[Idot_rsch2] += int_r2dot;      
      integrals_background[Idrsch_dri_dot] += int_drsch_dri_dot;

      // 2-metric & drvts
      integrals_background[Ig00] -= vol * (SQR(alpha(i,j)) - beta2(i,j));
      integrals_background[Ig0r] += vol * beta_d(0,i,j);
      integrals_background[Igrr] += vol * gamma_dd(0,0,i,j);

      integrals_background[Idr_g00] -= vol * (2.0*alpha(i,j)*dr_alpha(i,j) - dr_beta2(i,j));
      integrals_background[Idr_g0r] += vol * dr_beta_d(0,i,j);
      integrals_background[Idr_grr] += vol * dr_gamma_dd(0,0,i,j);
      
      integrals_background[Idot_g00] -= vol * (2.0*alpha(i,j)*dot_alpha(i,j) - dot_beta2(i,j));
      integrals_background[Idot_g0r] += vol * dot_beta_d(0,i,j);
      integrals_background[Idot_grr] += vol * dot_gamma_dd(0,0,i,j);

      //TODO check if the following components are needed at all!
      integrals_background[Igrt] += vol * gamma_dd(1,2,i,j);
      integrals_background[Igtt] += vol * gamma_dd(1,1,i,j);
      integrals_background[Igpp] += vol * gamma_dd(2,2,i,j);
      
      integrals_background[Idr_gtt] += vol * dr_gamma_dd(1,1,i,j);
      
    }// j
  }// i
  
#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, integrals_background, NVBackground,
		MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

  // Normalization
  const Real div_4PI = 1.0/(4.0*PI);
  for (int i=0; i<NVBackground; i++) 
    integrals_background[i] *= div_4PI;

  // Check
  const Real rsch2 = integrals_background[Irsch2];
  if (!(std::isfinite(r2)) || (rsch2<=1e-20)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in WaveExtractRWZ::BackgroundReduce" << std::endl
        << "Squared Schwarzschild radius is not finite or negative " << r2 << std::endl;
    ATHENA_ERROR(msg);
  }
  
  // All the data is here, time to finalize the background computation
  // -----------------------------------------------------------------

  rsch = std::sqrt(rsch2);
  const Real div_rsch = 1.0/rsch;
  
  // Idot_rsch2 is always d/dt(r^2) = d/dt(Integral)
  //  -> rdot = 1/(2r) d/dt(Integral)
  dot_rsch = 0.5 * div_rsch * integrals_background[Idot_rsch2];
  dot2_rsch = 0.0; //TODO we do not have 2nd drvts ATM
    
  drsch_dri = integrals_background[Idrsch_dri];
  d2rsch_dri2 = integrals_background[Id2rsch_dri2];
  
  dot_rsch = integrals_background[Idot_rsch2];
  drsch_dri_dot = integrals_background[Idrsch_dri_dot];

  Real g00 = integrals_background[Ig00];
  Real g0r = integrals_background[Ig0r];
  Real grr = integrals_background[Igrr];
  Real grt = integrals_background[Igrt];
  Real gtt = integrals_background[Igtt];
  Real gpp = integrals_background[Igpp];
  
  Real dr_g00 = integrals_background[Idr_g00];
  Real dr_g0r = integrals_background[Idr_g0r];
  Real dr_grr = integrals_background[Idr_grr];
  Real dr_gtt = integrals_background[Idr_gtt];
  
  Real dot_g00 = integrals_background[Idot_g00];
  Real dot_g0r = integrals_background[Idot_g0r];
  Real dot_grr = integrals_background[Idot_grr];

  // Update all quantities using the transformation of the metric components
  // from the isotropic radius R to the Schwarzschild radius r
  // Note variables are overwritten here, order matters!

  // dr_schwarzschild/dr_isotropic & Time derivatives of Schwarzschild radius  
  // Some integrals must be corrected for extra terms

  //if (method_areal_radius==areal) { }
  // else if (method_areal_radius==areal_simple) {}
  // else if (method_areal_radius==schw_gthth) {}
  //else if (method_areal_radius==schw_gphph) {}
  if (method_areal_radius==average_schw) {

    drsch_dri *= div_rsch;
    dri_drsch = 1.0/drsch_dri;
    
    d2rsch_dri2 = - div_rsch * ( SQR(drsch_dri) + d2rsch_dri2 );
    d2ri_drsch2 = - std::pow(dri_drsch,3)*d2rsch_dri2;

    drsch_dri_dot = div_rsch * ( - dot_rsch * drsch_dri + drsch_dri_dot );
    dri_drsch_dot = - SQR(dri_drsch) * drsch_dri_dot;
    
  } else {

    //TODO check other cases, implement general formulas or specify case-by-case
    
    dri_drsch = 1.0/drsch_dri;
    
    d2ri_drsch2 = - std::pow(dri_drsch,3) * d2rsch_dri2; // ?
    
  }
  
  // time derivatives of g0r and grr 
  dot_g0r = dri_drsch_dot * g0r + dri_drsch * dot_g0r;
  dot_grr = 2.0 * dri_drsch * dri_drsch_dot*grr + SQR(dri_drsch) * dot_grr;
    
  // Metric components, first drvt
  dr_g00  *= dri_drsch;
  dr_g0r   = d2ri_drsch2 * g0r + SQR(dri_drsch) * dr_g0r;
  dr_grr   = 2.0 * dri_drsch * d2ri_drsch2 * grr  + std::pow(dri_drsch,3) * dr_grr;

  g0r     *= dri_drsch ;
  grr     *= SQR(dri_drsch);
  
  // Inverse metric & Christoffel's symbols of the background metric 
  
  // Determinant
  const Real detg  = g00 * grr - SQR(g0r);
  const Real div_detg = (std::fabs(detg)<1e-12) ? 1.0 : 1.0/detg;
  const Real div_detg2 = SQR(div_detg);

  // Inverse matrix
  const Real g00_uu   =  grr*div_detg;
  const Real g0r_uu   = -g0r*div_detg;
  const Real grr_uu   =  g00*div_detg;
  
  // Derivatives of the inverse metric
  const Real dr_detg = div_detg2*(dr_g00*grr + g00*dr_grr - 2.0*g0r*dr_g0r);
  const Real dr_g00_uu =  dr_grr*div_detg - grr*dr_detg;
  const Real dr_g0r_uu = -dr_g0r*div_detg + g0r*dr_detg;
  const Real dr_grr_uu =  dr_g00*div_detg - g00*dr_detg;

  const Real dot_g0r_uu = -dot_g0r*div_detg + g0r*div_detg2*(dot_g00*grr + g00*dot_grr - 2.0*g0r*dot_g0r);
  
  // G stands for Gamma, these are the Christoffel symbols. 
  // Assuming time independence: 
  const Real G000 = -0.5*g0r_uu*dr_g00;
  const Real Gr00 = -0.5*grr_uu*dr_g00;
  const Real G00r =  0.5*g00_uu*dr_g00;
  const Real Gr0r =  0.5*g0r_uu*dr_g00;
  const Real G0rr =  g00_uu*dr_g0r + 0.5*g0r_uu*dr_grr;
  const Real Grrr =  g0r_uu*dr_g0r + 0.5*grr_uu*dr_grr;

  // Generic (time-dependent)
  // *_dyn stands for "dynamic": it uses time derivatives of the bakgrund metric 
  const Real G000_dyn =  0.5*g00_uu*dot_g00 + g0r_uu*dot_g0r - 0.5*g0r_uu*dr_g00;
  const Real Gr00_dyn =  grr_uu*dot_g0r - 0.5*grr_uu*dr_g00 + 0.5*g0r_uu*dot_g00;
  const Real G00r_dyn =  0.5*g00_uu*dr_g00  + 0.5*g0r_uu*dot_grr;
  const Real Gr0r_dyn =  0.5*g0r_uu*dr_g00  + 0.5*grr_uu*dot_grr;
  const Real G0rr_dyn =  g00_uu*dr_g0r - 0.5*g00_uu*dot_grr + 0.5*g0r_uu*dr_grr;
  const Real Grrr_dyn =  g0r_uu*dr_g0r - 0.5*g0r_uu*dot_grr + 0.5*grr_uu*dr_grr;

  // Compute differences
  const Real Delta_G000 = G000 - G000_dyn;
  const Real Delta_Gr00 = Gr00 - Gr00_dyn;
  const Real Delta_G00r = G00r - G00r_dyn;
  const Real Delta_Gr0r = Gr0r - Gr0r_dyn;
  const Real Delta_G0rr = G0rr - G0rr_dyn;
  const Real Delta_Grrr = Grrr - Grrr_dyn;

  norm_Delta_Gamma = SQR(Delta_G000)+SQR(Delta_G00r)+SQR(Delta_G0rr);
  norm_Delta_Gamma += SQR(Delta_Gr00)+SQR(Delta_Gr0r)+SQR(Delta_Grrr);
  norm_Delta_Gamma = std::sqrt(norm_Delta_Gamma);
  
  // The background is now in the correct coordinates.
  // Store everything.

  Schwarzschild_radius = rsch;
  Schwarzschild_mass = 0.5*rsch*(1 - grr_uu);  

  g_dd(0,0) = g00;
  g_dd(0,1) = g0r;
  g_dd(1,1) = grr;

  g_dr_dd(0,0) = dr_g00;
  g_dr_dd(0,1) = dr_g0r;
  g_dr_dd(1,1) = dr_grr;

  g_dot_dd(0,0) = dot_g00; 
  g_dot_dd(0,1) = dot_g0r;
  g_dot_dd(1,1) = dot_grr; 
  
  g_uu(0,0) = g00_uu;
  g_uu(0,1) = g0r_uu;
  g_uu(1,1) = grr_uu;

  g_dr_uu(0,0) = dr_g00_uu;
  g_dr_uu(0,1) = dr_g0r_uu;
  g_dr_uu(1,1) = dr_grr_uu;

  g_dot_uu(0,0) = 0.0; //dot_g00_uu; //TODO needed ? should be computed ?
  g_dot_uu(0,1) = dot_g0r_uu; 
  g_dot_uu(1,1) = 0.0; //dot_grr_uu; //TODO needed ? should be computed ?
  
  Gamma_udd(0,0,0) = G000;
  Gamma_udd(0,0,1) = G00r;
  Gamma_udd(0,1,1) = G0rr;
  Gamma_udd(1,0,0) = Gr00;
  Gamma_udd(1,0,1) = Gr0r;
  Gamma_udd(1,1,1) = Grrr;

  Gamma_dyn_udd(0,0,0) = G000_dyn;
  Gamma_dyn_udd(0,0,1) = Gamma_dyn_udd(0,1,0) = G00r_dyn; //CHECK is this fine/necessary with SYM2?
  Gamma_dyn_udd(0,1,1) = G0rr_dyn;
  Gamma_dyn_udd(1,0,0) = Gr00_dyn;
  Gamma_dyn_udd(1,0,1) = Gamma_dyn_udd(1,0,1) = Gr0r_dyn;
  Gamma_dyn_udd(1,1,1) = Grrr_dyn;
    
}

//----------------------------------------------------------------------------------------
// \!fn void WaveExtractRWZ::MultipoleReduce()
// \brief compute the multipoles 
//        performs local sums and then MPI reduce
void WaveExtractRWZ::MultipoleReduce() {

  const int lmpoints_x2 = lmpoints*2;
  const Real dthdph = dth_grid() * dph_grid();
  const Real r = Schwarzschild_radius;
  const Real div_r = 1.0/r;
  const Real div_r2 = SQR(div_r);
  const Real div_r3 = div_r2 * div_r;
    
  // Zeros the integrals
  for (int i=0; i<NVBackground*2*lmpoints; i++)    
    integrals_multipoles[i] = 0.0;
      
  for (int i=0; i<Ntheta; i++) {
    
    const Real theta = th_grid(i);
    const Real sinth = std::sin(theta);
    const Real costh = std::cos(theta);
    const Real sinth2 = SQR(sinth);
    const Real costh2 = SQR(costh);
    const Real div_sinth = 1.0/sinth;
    const Real div_sinth2 = SQR(div_sinth);

    const Real vol = dthdph * sinth;

    const Real lambda = l*(l+1);
    const Real div_lambda = 1.0/(lambda);
    const Real div_lambda_lambda_2 = 1.0/(lambda*(lambda-2.0));

    
    for(int j=0; j<Nphi; j++){

      if (!havepoint(i,j)) continue;
      
      const Real phi   = ph_grid(j);
      const Real sinph = std::sin(phi);
      const Real cosph = std::cos(phi);
      const Real sinph2 = SQR(sinph);
      const Real cosph2 = SQR(cosph);
      
      for(int l=2; l<=lmax; ++l) {
	for(int m=-l; m<=l; ++m) {
	  const int lm = MIndex(l,m);
	  
	  // Local sums
	  // ----------

	  for(int c=0; c<RealImag; ++c) {
	    const int s = (c==0)? 1.0 : -1.0; // conjugate
	    const Real sY   = s * Y(lm,c);
	    const Real sYth = s * Yth(lm,c);
	    const Real sYph = s * Yph(lm,c);
	    const Real sX   = s * X(lm,c);
	    const Real sW   = s * W(lm,c);

	    
	    // even parity
	    
	    integrals_multipoles[lmpoints_x2 * Ih00 + lm + c] -=
	      vol * (SQR(alpha(i,j)) - beta2(i,j)) * sYlm ;
	    
	    integrals_multipoles[lmpoints_x2 * Ih01 + lm + c] += 
	      vol * beta_d(0,i,j) * sY;

	    integrals_multipoles[lmpoints_x2 * Ih11 + lm + c] += 
	      vol * gamma_dd(1,1,i,j) * sY;

	    
	    integrals_multipoles[lmpoints_x2 * Ih0 + lm + c] += 
	      vol * div_lambda * (beta_d(1,i,j) * sYth + beta_d(2,i,j) * div_sinth2 * sYph );

	    integrals_multipoles[lmpoints_x2 * Ih1 + lm + c] += 
	      vol * div_lambda * (gamma_dd(0,1,i,j) * sYth + gamma_dd(0,2,i,j) * div_sinth2 * sYph );
	    
	    
	    integrals_multipoles[lmpoints_x2 * IK + lm + c] += 
	      vol * 0.5 * div_r2 * (gamma_dd(1,1,i,j) + gamma_dd(2,2,i,j) * div_sinth2) * sY;

	    integrals_multipoles[lmpoints_x2 * IG + lm + c] += 
	      vol * 0.5 * div_r2 * div_lambda_lambda_2
	      * ( (gamma_dd(1,1,i,j) + gamma_dd(2,2,i,j) * div_sinth2) * W
		  + 2.0*gamma_dd(1,2,i,j) * X );

	    
	    // even parity radial drvts

	    integrals_multipoles[lmpoints_x2 * Ih00_dr + lm + c] -=
	      vol * (2.0*alpha(i,j)*dr_alpha(i,j) - dr_beta2(i,j)) * sYlm ;
	    
	    integrals_multipoles[lmpoints_x2 * Ih01_dr + lm + c] += 
	      vol * beta_dr_d(0,i,j) * sY;

	    integrals_multipoles[lmpoints_x2 * Ih11_dr + lm + c] += 
	      vol * dr_gammadd(1,1,i,j) * sY;

	    
	    integrals_multipoles[lmpoints_x2 * Ih0_dr + lm + c] += 
	      vol * div_lambda * (dr_beta_d(1,i,j) * sYth
				  + dr_beta_d(2,i,j) * div_sinth2 * sYph );

	    integrals_multipoles[lmpoints_x2 * Ih1_dr + lm + c] += 
	      vol * div_lambda * (dr_gamma_dd(0,1,i,j) * sYth
				  + dr_gamma_dd(0,2,i,j) * div_sinth2 * sYph );
	    
	    
	    integrals_multipoles[lmpoints_x2 * IK_dr + lm + c] += 
	      vol * 0.5 * ( div_r2 * (dr_gamma_dd(1,1,i,j) + dr_gamma_dd(2,2,i,j) * div_sinth2) * sY
			    -2.0*div_r3 * (gamma_dd(1,1,i,j) + gamma_dd(2,2,i,j) * div_sinth2) * sY );
	    
	    integrals_multipoles[lmpoints_x2 * IG_dr + lm + c] += 
	      vol * 0.5 * div_lambda_lambda_2 *
	      ( div_r2 * ( (dr_gamma_dd(1,1,i,j) + dr_gamma_dd(2,2,i,j) * div_sinth2) * W
			   + 2.0*dr_gamma_dd(1,2,i,j) * X )
		-2.0*div_r3 * ( ( (gamma_dd(1,1,i,j) + gamma_dd(2,2,i,j) * div_sinth2) * W
				  + 2.0*gamma_dd(1,2,i,j) * X ) ) );


	    // even parity time derivatives
	    // NB assumes rdot=0
	    
	    integrals_multipoles[lmpoints_x2 * Ih00_dt + lm + c] -=
	      vol * (2.0*alpha(i,j)*dot_alpha(i,j) - dot_beta2(i,j)) * sYlm ;
	    
	    integrals_multipoles[lmpoints_x2 * Ih01_dt + lm + c] += 
	      vol * dot_beta_d(0,i,j) * sY;

	    integrals_multipoles[lmpoints_x2 * Ih11_dt + lm + c] += 
	      vol * dot_gamma_dd(1,1,i,j) * sY;

	    
	    integrals_multipoles[lmpoints_x2 * Ih0_dt + lm + c] += 
	      vol * div_lambda * (dot_beta_d(1,i,j) * sYth
				  + dot_beta_d(2,i,j) * div_sinth2 * sYph );

	    integrals_multipoles[lmpoints_x2 * Ih1_dt + lm + c] += 
	      vol * div_lambda * (dot_gamma_dd(0,1,i,j) * sYth
				  + dot_gamma_dd(0,2,i,j) * div_sinth2 * sYph );
	    
	    
	    integrals_multipoles[lmpoints_x2 * IK_dt + lm + c] += 
	      vol * 0.5 * div_r2 * (dot_gamma_dd(1,1,i,j)
				    + dot_gamma_dd(2,2,i,j) * div_sinth2) * sY;

	    integrals_multipoles[lmpoints_x2 * IG_dt + lm + c] += 
	      vol * 0.5 * div_r2 * div_lambda_lambda_2
	      * ( (dot_gamma_dd(1,1,i,j) + dot_gamma_dd(2,2,i,j) * div_sinth2) * W
		  + 2.0*dot_gamma_dd(1,2,i,j) * X );

	    
	    // odd parity

	    integrals_multipoles[lmpoints_x2 * IH0 + lm + c] += 
	      vol * div_lambda * div_sinth * ( - beta_d(1,i,j)*sYph + beta_d(2,i,j)*sYth );

	    integrals_multipoles[lmpoints_x2 * IH1 + lm + c] += 
	      vol * div_lambda * div_sinth * ( - gamma_dd(0,1,i,j)*sYph + gamma_dd(0,2,i,j)*sYth );

	    
	    integrals_multipoles[lmpoints_x2 * IH1 + lm + c] += 
	      vol * div_lambda_lambda_2 
	      * ( div_sinth * ( - gamma_dd(1,1,i,j) + gamma_dd(2,2,i,j) * div_sinth2 ) *X
		  + 2.0* gamma_dd(1,2,i,j) * div_sinth2 * div_sinth * W ); //CHECK 1/sin^3 or 1/sin ?

	    
	    // odd parity radial drvts
	    
	    integrals_multipoles[lmpoints_x2 * IH0_dr + lm + c] += 
	      vol * div_lambda * div_sinth * ( - dr_beta_d(1,i,j)*sYph
					       + dr_beta_d(2,i,j)*sYth );

	    integrals_multipoles[lmpoints_x2 * IH1_dr + lm + c] += 
	      vol * div_lambda * div_sinth * ( - dr_gamma_dd(0,1,i,j)*sYph
					       + dr_gamma_dd(0,2,i,j)*sYth );

	    
	    integrals_multipoles[lmpoints_x2 * IH1_dr + lm + c] += 
	      vol * div_lambda_lambda_2 
	      * ( div_sinth * ( - dr_gamma_dd(1,1,i,j) + dr_gamma_dd(2,2,i,j) * div_sinth2 ) *X
		  + 2.0* dr_gamma_dd(1,2,i,j) * div_sinth2 * div_sinth * W ); //CHECK 1/sin^3 or 1/sin ?

	    
	    // odd parity time drvts

	    integrals_multipoles[lmpoints_x2 * IH0_dt + lm + c] += 
	      vol * div_lambda * div_sinth * ( - dot_beta_d(1,i,j)*sYph
					       + dot_beta_d(2,i,j)*sYth );

	    integrals_multipoles[lmpoints_x2 * IH1_dt + lm + c] += 
	      vol * div_lambda * div_sinth * ( - dot_gamma_dd(0,1,i,j)*sYph
					       + dot_gamma_dd(0,2,i,j)*sYth );

	    
	    integrals_multipoles[lmpoints_x2 * IH1_dt + lm + c] += 
	      vol * div_lambda_lambda_2 
	      * ( div_sinth * ( - dot_gamma_dd(1,1,i,j) + dot_gamma_dd(2,2,i,j) * div_sinth2 ) *X
		  + 2.0* dot_gamma_dd(1,2,i,j) * div_sinth2 * div_sinth * W ); //CHECK 1/sin^3 or 1/sin ?
	    
	  }// for c Real/Img
	  
	}// for m
      }// for l
      
    }// j
  }// i

#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, integrals_multipoles, NVMultipoles*lmpoints_x2,
		MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

  // Everything is here, store the multipoles.
  for(int l=2; l<=lmax; ++l) {
    for(int m=-l; m<=l; ++m) {
      const int lm = MIndex(l,m);      
      for(int c=0; c<RealImag; ++c) {

	// even
	h00(lm,c) = integrals_multipoles[lmpoints_x2 *Ih00 + lm + c];
	h01(lm,c) = integrals_multipoles[lmpoints_x2 *Ih01 + lm + c];
	h11(lm,c) = integrals_multipoles[lmpoints_x2 *Ih11 + lm + c];

	h00_dr(lm,c) = integrals_multipoles[lmpoints_x2 *Ih00_dr + lm + c];
	h01_dr(lm,c) = integrals_multipoles[lmpoints_x2 *Ih01_dr + lm + c];
	h11_dr(lm,c) = integrals_multipoles[lmpoints_x2 *Ih11_dr + lm + c];

	h00_dot(lm,c) = integrals_multipoles[lmpoints_x2 *Ih00_dot + lm + c];
	h01_dot(lm,c) = integrals_multipoles[lmpoints_x2 *Ih01_dot + lm + c];
	h11_dot(lm,c) = integrals_multipoles[lmpoints_x2 *Ih11_dot + lm + c];

	
	h0(lm,c) = integrals_multipoles[lmpoints_x2 *Ih0 + lm + c];
	h1(lm,c) = integrals_multipoles[lmpoints_x2 *Ih1 + lm + c];

	h0_dr(lm,c) = integrals_multipoles[lmpoints_x2 *Ih0_dr + lm + c];
	h1_dr(lm,c) = integrals_multipoles[lmpoints_x2 *Ih1_dr + lm + c];

	h0_dot(lm,c) = integrals_multipoles[lmpoints_x2 *Ih0_dot + lm + c];
	h1_dot(lm,c) = integrals_multipoles[lmpoints_x2 *Ih1_dot + lm + c];

	
	G(lm,c) = integrals_multipoles[lmpoints_x2 *IG + lm + c];
	K(lm,c) = integrals_multipoles[lmpoints_x2 *IK + lm + c];

	G_dr(lm,c) = integrals_multipoles[lmpoints_x2 *IG_dr + lm + c];
	K_dr(lm,c) = integrals_multipoles[lmpoints_x2 *IK_dr + lm + c];

	G_dot(lm,c) = integrals_multipoles[lmpoints_x2 *IG_dot + lm + c];
	K_dot(lm,c) = integrals_multipoles[lmpoints_x2 *IK_dot + lm + c];

	
	// odd
	H0(lm,c) = integrals_multipoles[lmpoints_x2 *IH0 + lm + c];
	H1(lm,c) = integrals_multipoles[lmpoints_x2 *IH1 + lm + c];

	H0_dr(lm,c) = integrals_multipoles[lmpoints_x2 *IH0_dr + lm + c];
	H1_dr(lm,c) = integrals_multipoles[lmpoints_x2 *IH1_dr + lm + c];

	H0_dot(lm,c) = integrals_multipoles[lmpoints_x2 *IH0_dot + lm + c];
	H1_dot(lm,c) = integrals_multipoles[lmpoints_x2 *IH1_dot + lm + c];


	H(lm,c) = integrals_multipoles[lmpoints_x2 *IH + lm + c];

	H_dr(lm,c) = integrals_multipoles[lmpoints_x2 *IH_dr + lm + c];

	H_dot(lm,c) = integrals_multipoles[lmpoints_x2 *IH_dot + lm + c];
	
      }// real&imag      
    }// for m
  }// for l
  
  // Compute the various gauge-invariant functions
  MultipolesGaugeInvariant();
  MasterFuns();
  
}

//----------------------------------------------------------------------------------------
// \!fn void WaveExtractRWZ::MasterFuns()
// \brief compute even and odd parity master functions
void WaveExtractRWZ::MasterFuns() {

  const Real M = Schwarzschild_mass;
  const Real r = Schwarzschild_radius;
  const Real r2 = SQR(r);
  const Real div_r = 1.0/r;

  const Real S = -(1.0-2.0*M*div_r);
  const Real abs_detg = std::fabs(g_dd(0,0) * g_dd(1,1) - SQR(g_dd(0,1)));
  const Real div_sqrtdetg = 1.0/(std:sqrt(abs_detg));
  
  for(int l=2; l<=lmax; ++l) {
    const Real lambda = l*(l+1);
    const Real div_lambda_2 = 1.0/(lambda-2.0);
    const Real fac_Psie = (2.0*r)/((lambda-2)*r + 6.0*M);
    const Real r_div_lambda = r/lambda;
    const Real Lam  = (l-1)*(l-2) + 6*M*div_r;
    const Real Qpnorm = std::sqrt(2*(l-1)*(l-2)/lambda)/Lam;
    
    for(int m=-l; m<=l; ++m) {
      const int lm = MIndex(l,m);
      for(int c=0; c<RealImag; ++c) {
	  
	// Even parity in Schwarschild coordinates
	
	const Real Qplus_ = Qpnorm*( lambda*S*( r2*G_dr(lm,c) -2*h1(lm,c) )
				     + 2.0*r*S*( S*h11(lm,c) - r*K_dr(lm,c) )
				     + r*Lam*K(lm,c) );
	
	Psie_sch(lm,c) = Qplus_/(std::sqrt(2.0*lambda(lambda-2.0)));
	Qplus(lm,c) = Qplus_;
	
	// Odd parity in Schwarzschild coordinates
	
	const Real Psio_sch_ = ( r*(H1_dot(lm,c) - H0_dr(lm,c)) + 2.0*H0(lm,c) )*div_lambda_2;
	Psio_sch(lm,c) = Psio_sch_;	
	
	const Real Qstar_ = - div_r*S*( H1(lm,c) - H_dr(lm,c)*div_r + 2.0*H(lm,c)*SQR(div_r) );
	Qstar(lm,c) = Qstar_;
	
	// Even parity in general coordinates (static)
	
	const Real term1_K = K(lm,c); 
	const Real term1_hG = (- 2.0*div_r)*( g_uu(1,1)*( h1(lm,c) - 0.5*r2*G_dr(lm,c) )
					      + g_uu(0,1)*( h0(lm,c) - 0.5*r2*G_dot(lm,c) ) );	
	
	const Real term_hAB = SQR(g_uu(0,1))*h00(lm,c)
	  + 2.0*g_uu(0,1)*g_uu(1,1)*h01(lm,c)
	  + SQR(g_uu(1,1))*h11(lm,c);
	const Real term2_K = - r*( g_uu(1,1)*K_dr(lm,c) + g_uu(0,1)*K_dot(lm,c) );	
	
	Real coef_h0 = - r*std::pow(g_uu(0,1),3)*g_dr_dd(0,0)
	  + 2.0*r*g_uu(1,1)*( g_uu(0,0)*g_uu(1,1)*g_dr(0,1) )
	  + g_uu(0,1)*g_uu(1,1)*( - 2.0 + 2.0*r*g_uu(0,0)*g_dr_dd(0,0)
				  + r*g_uu(1,10*g_dr_dd(1,1) ) );
	
	coef_h0 *= div_r;
	
	Real coef_h1 = 2.0*SQR(g_uu(1,1))*(-1.0 + r*g_uu(0,1)*g_dr_dd(0,1))
	+ r*std::pow(g_uu(1,1),3)*g_dr_dd(1,1)
	  + r*g_uu(1,1)*SQR(g_uu(0,1))*g_dr_dd(0,0)
	  + 2.0*r*g_uu(1,1)*g_dr_uu(1,1);

	coef_h1 *= div_r;
	
	Real coef_G_dr = 2.0*g_uu(1,1)*(-1.0 + r*g_uu(0,1)*g_dr_dd(0,1))
	  + r*SQR(g_uu(1,1))*g_dr_dd(1,1)
	  + r*SQR(g_uu(0,1))*g_dr_dd(0,0)
	  + 2.0*r*g_dr_uu*(1,1);

	coef_G_dr *= -0.5*r*g_uu(1,1);
	
	Real coef_G_dot = r*std::pow(g_uu(0,1),3)*g_dr_dd(0,0)
	  - 2*r*g_uu(1,1)*( g_uu(0,0)*g_uu(1,1)*g_dr_dd(0,1) + g_dr_uu(0,1) )
	  - g_uu(0,1)*g_uu(1,1)*(-2.0 + 2.0*r*g_uu(0,0)*g_dr_dd(0,0) + r*g_uu(1,1)*g_dr_dd(1,1) );

	coef_G_dot *= 0.5*r;
	
	Psie(lm,c) = term1_K;
	Psie(lm,c) += term1_hG;
	Psie(lm,c) += fac_Psie*( term_hAB +
				 coef_h0 * h0(lm,c) +
				 coef_h1 * h1(lm,c) +
				 coef_G_dr * G_dr(lm,c) +
				 coef_G_dot * G_dot(lm,c) +
				 term2_K );
	Psie(lm,c) *= r_div_lambda;
	
	// Odd parity in general coordinates (static)
	
	Psio(lm,c) = Psio_sch_ * div_sqrtdetg;
	
	// Even parity in general coordinates (dynamic)
	
	Real coef_h0_t = 2.0*r*std::pow(g_uu(0,1),3)*g_dot_dd(0,1)
	  + 2.0*r*g_uu(0,1)*g_dot_uu(0,1) //CHECK g_dot_uu(0,1) in the notes, but eq full of typos
	  + ( -r*g_uu(0,0)*SQR(g_uu(1,1)) + 2.0*r*g_uu(1,1)*SQR(g_uu(0,1)) )*g_dot_dd(1,1)
	  + r*SQR(g_uu(0,1))*g_uu(0,0)*g_dot_dd(0,0);
	
	coef_h0_t *= div_r;
	
	Real coef_h1_t = SQR(g_uu(0,1))*g_dot_dd(0,0)
	  + 2.0*g_uu(0,1)*g_uu(1,1)*g_dot_dd(0,1)
	  + SQR(g_uu(1,1))*g_dot_dd(1,1);
	
	coef_h1_t *= g_uu(0,1); //CHECK notes for factor *div_r

	Real coef_G_dr_t = 2.0*g_uu(0,1)*g_uu(1,1)*g_dot_dd(0,1)
	  + SQR(g_uu(0,1))*g_dot_dd(0,0)
	  + SQR(g_uu(1,1))*g_dot_dd(1,1);
	  
	coef_G_dr_t *= r*g_uu(0,1);
	
	Real coef_G_dot_t = -2.0*r*std::pow(g_uu(0,1),3)*g_dot_tdd(0,1)
	  - 2.0*g_uu(0,1)*g_dot_uu(0,1) //CHECK in the notes
	  - SQR(g_uu(0,1))*g_uu(0,0)*g_dot_dd(0,0)
	  + g_uu(1,1)*(g_uu(1,1)*g_uu(0,0)-2.0*SQR(g_uu(0,1)))*g_dot_dd(1,1);
	
	coef_G_dot_t *= -0.5*r2; //CHECK again, notes 
		  
	Psie_dyn(lm,c) = term1_K;
	Psie_dyn(lm,c) += term1_hG;
	Psie_dyn(lm,c) += fac_Psie*( term_hAB +
				     (coef_h0 + coef_h0_t) * h0(lm,c) +
				     (coef_h1 + coef_h1_t) * h1(lm,c) +
				     (coef_G_dr + coef_G_dr_t) * G_dr(lm,c) +
				     (coef_G_dot + coef_G_dot_t) * G_dot(lm,c) +
				     term2_K );
	Psie_dyn(lm,c) *= r_div_lambda;
	
	// Odd parity in general coordinates (dynamic)

	Psio_dyn(lm,c) = ( r*(H1_dot(lm,c) - H0_dr(lm,c)) + 2.0*(H0(lm,c) - H1(lm,c)*rdot) );
	Psio_dyn(lm,c) *= div_lambda_2*div_sqrtdetg;	
	
      }// real& imag
    }// for m
  }// for l

}

//----------------------------------------------------------------------------------------
// \!fn void WaveExtractRWZ::MultipolesGaugeInvariant()
// \brief compute gauge invariant multipoles from most general expression (time-dependent)
void WaveExtractRWZ::MultipolesGaugeInvariant() {
  
  kappa_dd.ZeroClear();
  kappa_d.ZeroClear();
  kappa.ZeroClear();
  Tr_kappa_dd.ZeroClear();
  
  const Real r = rsch;
  const Real r2 = SQR(rsch);
  const Real rdot = dot_rsch;
  const Real div_r = 1.0/rsch;

  //TODO This is a placeholder, these second drvt of the G multipole are not computed ATM!
  const Real G_dt2 = 0.0;
  const Real G_dtdr = 0.0;
  const Real G_dr2 = 0.0; 

  norm_Tr_kappa_dd = 0.0;
  
  for(int l=2; l<=lmax; ++l) {
    for(int m=-l; m<=l; ++m) {
      const int lm = MIndex(l,m);
      for(int c=0; c<RealImag; ++c) {

	// even-parity
	kappa_dd(0,0,lm,c) = h00(lm,c);
	kappa_dd(0,0,lm,c) += - 2.0*h0_dot(lm,c)
	  + 2.0*(Gamma_dyn(0,0,0)*h0(lm,c) + Gamma_dyn(1,0,0)*h1(lm,c));
	kappa_dd(0,0,lm,c) += - 2.0*r*rdot*G_dot(lm,c)
	  + r2*(G_dt2 - Gamma_dyn(0,0,0)*G_dot(lm,c) - Gamma_dyn(1,0,0)*G_dr(lm,c));
	
	kappa_dd(0,1,lm,c) = h01(lm,c);
	kappa_dd(0,1,lm,c) += - h1_dot(lm,c) - h0_dr(lm,c)
	  + 2.0*(Gamma_dyn(0,0,1)*h0(lm,c) + Gamma_dyn(1,0,1)*h1(lm,c));
	kappa_dd(0,1,lm,c) += r*rdot*G_dr(lm,c) + r*G_dot(lm,c)
	  + r2*(G_dtdr - Gamma_dyn(0,0,1)*G_dot(lm,c) - Gamma_dyn(1,0,1)*G_dr(lm,c));
	
	kappa_dd(1,1,lm,c) = h11(lm,c);
	kappa_dd(1,1,lm,c) += - 2.0*h1_dr(lm,c)
	  + 2.0*(Gamma_dyn(0,1,1)*h0(lm,c) + Gamma_dyn(1,1,1)*h1(lm,c));
	kappa_dd(1,1,lm,c) += 2.0*r*G_dr(lm,c)
	  + r2*(G_dr2 - Gamma_dyn(0,1,1)*G_dot(lm,c) - Gamma_dyn(1,1,1)*G_dr(lm,c));
	
	kappa(lm,c) = K(lm,c);
	kappa(lm,c) -= (g_uu(0,0) *rdot*( 2.0*h0(lm,c) - r2*G_dot(lm,c) ) +
			g_uu(0,1) *( rdot*(2.0*h1(lm,c)-r2*G_dr(lm,c)) + 2.0*h0(lm,c) -r2G_dot(lm,c) ) +
			g_uu(1,1) *(2.0*h1(lm,c)-r2*G_dr(lm,c)) )*div_r;
	  
	// odd-parity
	kappa_d(0,lm,c) = H0(lm,c) - H0_dot(lm,c) + H(lm,c)*2.0*rdot*div_r; 
	kappa_d(1,lm,c) = H1(lm,c) - H0_dr(lm,c) + H(lm,c)*2.0*div_r; 

	// Trace constraint
	Tr_kappa_dd(lm,c) = 0.0;
	for (int A=0; A<2; ++A)
	  for (int B=0; B<2; ++B) {
	    Tr_kappa_dd(lm,c) += g_uu(A,B)*kappa_dd(A,B,lm,c);
	  }

	norm_Tr_kappa_dd += SQ(Tr_kappa_dd(lm,c))
	
      }// real&imag
      
    }// for m
  }// for l

  norm_Tr_kappa_dd = std::sqrt(norm_Tr_kappa_dd);
  
}
