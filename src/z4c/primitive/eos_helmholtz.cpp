//! \file eos_helmholtz.cpp
//  \brief Implementation of EOSHelmholtz

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <iostream>

#include <hdf5.h>
#include <hdf5_hl.h>

#include "eos_helmholtz.hpp"
#include "unit_system.hpp"

using namespace Primitive;
using namespace std;

#define MYH5CHECK(ierr) \
  if(ierr < 0) { \
    stringstream ss; \
    ss << __FILE__ << ":" << __LINE__ << " error reading EOS table!"; \
    throw runtime_error(ss.str().c_str()); \
  }

EOSHelmholtz::EOSHelmholtz():
  m_id_log_ne(numeric_limits<Real>::quiet_NaN()),
  m_id_log_t(numeric_limits<Real>::quiet_NaN()),
  m_nn(0), m_nt(0) {
  n_species = 1;
  eos_units = &Nuclear;

  min_Y[SCYE] = 0.0; // will be overwritten by ReadTableFromFile
  min_Y[SCXN] = 0.0;
  min_Y[SCXP] = 0.0;
  min_Y[SCXA] = 0.0;
  min_Y[SCXH] = 0.0;
  min_Y[SCAH] = 1.0;

  max_Y[SCYE] = 1.0; // will be overwritten by ReadTableFromFile
  max_Y[SCXN] = 1.0;
  max_Y[SCXP] = 1.0;
  max_Y[SCXA] = 1.0;
  max_Y[SCXH] = 1.0;
  max_Y[SCAH] = 500.0;
}

EOSHelmholtz::~EOSHelmholtz() {
}

//Definitions for static members
Real * EOSHelmholtz::m_log_ne = nullptr;
Real * EOSHelmholtz::m_log_t = nullptr;
Real * EOSHelmholtz::m_table = nullptr;
bool EOSHelmholtz::m_initialized = false;

Real EOSHelmholtz::sm_id_log_ne = numeric_limits<Real>::quiet_NaN();
Real EOSHelmholtz::sm_id_log_t = numeric_limits<Real>::quiet_NaN();

int EOSHelmholtz::sm_nn = 0;
int EOSHelmholtz::sm_nt = 0;

Real EOSHelmholtz::s_max_n = numeric_limits<Real>::quiet_NaN();
Real EOSHelmholtz::s_min_n = numeric_limits<Real>::quiet_NaN();
Real EOSHelmholtz::s_max_T = numeric_limits<Real>::quiet_NaN();
Real EOSHelmholtz::s_min_T = numeric_limits<Real>::quiet_NaN();

Real EOSHelmholtz::TemperatureFromE(Real n, Real e, Real *Y) {
  assert (m_initialized);
  return TemperatureFromEps(n, e/(mb*n) - 1, Y);
}

Real EOSHelmholtz::TemperatureFromEps(Real n, Real eps, Real *Y) {
  Real eps_min = MinimumInternalEnergy(n, Y);
  Real eps_max = MaximumInternalEnergy(n, Y);
//  return temperature_from_var(ECE, e, n, Y);
  return (eps <= eps_min) ? min_T : (eps >= eps_max) ? max_T : temperature_from_var(ECEPS, eps, n, Y);
}

Real EOSHelmholtz::TemperatureFromP(Real n, Real p, Real *Y) {
  assert (m_initialized);
  Real p_min = MinimumPressure(n, Y);
  Real p_max = MaximumPressure(n,Y);

  return (p <= p_min) ? min_T : (p >= p_max) ? max_T :
    temperature_from_var(ECP, p, n, Y);
}

Real EOSHelmholtz::TemperatureFromEntropy(Real n, Real s, Real *Y) {
  assert (m_initialized);
  Real s_min = MinimumEntropy(n, Y);
  Real s_max = MaximumEntropy(n,Y);

  return (s <= s_min) ? min_T : (s >= s_max) ? max_T :
	       temperature_from_var(ECENT, s, n, Y);
}

Real EOSHelmholtz::SpecificInternalEnergy(Real n, Real T, Real *Y) {
  assert (m_initialized);
  return eval_at_nty(ECEPS, n, T, Y);
}

Real EOSHelmholtz::Energy(Real n, Real T, Real *Y) {
  assert (m_initialized);
  return (SpecificInternalEnergy(n, T, Y) + 1)*n*mb;
}

Real EOSHelmholtz::Pressure(Real n, Real T, Real *Y) {
  assert (m_initialized);
  return eval_at_nty(ECP, n, T, Y);
}

Real EOSHelmholtz::Abar(Real n, Real T, Real *Y) {
  assert (m_initialized);
  return 1.0/inverse_abar(Y);
}

Real EOSHelmholtz::Entropy(Real n, Real T, Real *Y) {
  assert (m_initialized);
  return eval_at_nty(ECENT, n, T, Y);
}

Real EOSHelmholtz::Enthalpy(Real n, Real T, Real *Y) {
  assert (m_initialized);
  Real const P = Pressure(n, T, Y);
  Real const e = Energy(n, T, Y);
  return (P + e)/n;
}

Real EOSHelmholtz::SoundSpeed(Real n, Real T, Real *Y) {
  assert (m_initialized);
  Real pres = Pressure(n, T, Y);
  Real ener = SpecificInternalEnergy(n, T, Y);
  Real dpresdt = eval_at_nty(ECDPDT, n, T, Y);
  Real dpresdn = eval_at_nty(ECDPDN, n, T, Y);
  Real denerdt = eval_at_nty(ECDEPSDT, n, T, Y);
  Real zzi = n / pres;
  Real zz = 1/zzi;
  Real chit = T / pres * dpresdt;
  Real chin = dpresdn * zzi;
  Real cv = denerdt;
  Real x = zz * chit / (T * cv);
  Real gam1 = chit * x + chin;
  Real z = 1.0 + (ener + mb) * zzi;
  return sqrt(gam1 / z);
}

Real EOSHelmholtz::NeutronChemicalPotential(Real n, Real T, Real *Y) {
  assert (m_initialized);
  Real Yn = Y[SCXN];
  // Non-degenerate
  return mn + T * log(n*Yn/2*pow(sac_const/(mn*T), 1.5));
  // Fermion
  // Real delta = sqrt(M_PI)*n*Yn/(2.0*2*pow(mn*T/sac_const, 1.5));
  // return inverse_fermi_one_half(delta) * T  + mn;
}

Real EOSHelmholtz::ProtonChemicalPotential(Real n, Real T, Real *Y) {
  assert (m_initialized);
  Real Yp = Y[SCXP];
  // Non-degenerate
  return mp + T * log(n*Yp/2*pow(sac_const/(mp*T), 1.5));
  // Fermion
  // Real delta = sqrt(M_PI)*n*Yp/(2.0*2*pow(mp*T/sac_const, 1.5));
  // return inverse_fermi_one_half(delta) * T  + mp;
}

Real EOSHelmholtz::ElectronChemicalPotential(Real n, Real T, Real *Y) {
  assert (m_initialized);
  Real etaele = eval_at_nty(ECETA, n, T, Y);
  return etaele * T + me;
}

Real EOSHelmholtz::BaryonChemicalPotential(Real n, Real T, Real *Y) {
  assert (m_initialized);
  return NeutronChemicalPotential(n, T, Y);
}

Real EOSHelmholtz::ChargeChemicalPotential(Real n, Real T, Real *Y) {
  assert (m_initialized);
  return ProtonChemicalPotential(n, T, Y) - NeutronChemicalPotential(n, T, Y);
}

Real EOSHelmholtz::ElectronLeptonChemicalPotential(Real n, Real T, Real *Y) {
  assert (m_initialized);
  return ElectronChemicalPotential(n, T, Y) + ChargeChemicalPotential(n, T, Y); // mu_e = mu_l - mu_q
}

Real EOSHelmholtz::MinimumEnthalpy() {
  return m_min_h;
}

Real EOSHelmholtz::MinimumPressure(Real n, Real *Y) {
  return Pressure(n, min_T, Y);
}

Real EOSHelmholtz::MaximumPressure(Real n, Real *Y) {
  return Pressure(n, max_T, Y);
}

Real EOSHelmholtz::MinimumInternalEnergy(Real n, Real *Y) {
  return SpecificInternalEnergy(n, min_T, Y);
}

Real EOSHelmholtz::MaximumInternalEnergy(Real n, Real *Y) {
  return SpecificInternalEnergy(n, max_T, Y);
}


Real EOSHelmholtz::MinimumEntropy(Real n, Real *Y) {
  return Entropy(n, min_T, Y);
}

Real EOSHelmholtz::MaximumEntropy(Real n, Real *Y) {
  return Entropy(n, max_T, Y);
}

void EOSHelmholtz::ReadTableFromFile(std::string fname, Real min_Ye, Real max_Ye) {
  #pragma omp critical (EOSHelmholtz_ReadTable)
  {
    if (m_initialized==false) {
      herr_t ierr;
      hid_t file_id;
      hsize_t sne, st, syq;

      // Open input file
      // -------------------------------------------------------------------------
      file_id = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        MYH5CHECK(file_id);

      // Get dataset sizes
      // -------------------------------------------------------------------------
      ierr = H5LTget_dataset_info(file_id, "ne", &sne, NULL, NULL);
        MYH5CHECK(ierr);
      ierr = H5LTget_dataset_info(file_id, "t", &st, NULL, NULL);
        MYH5CHECK(ierr);
      m_nn = sne;
      m_nt = st;

      // Allocate memory
      // -------------------------------------------------------------------------
      m_log_ne = new Real[m_nn];
      m_log_t = new Real[m_nt];
      m_table = new Real[ECNVARS*m_nn*m_nt];
      double * scratch = new double[m_nn*m_nt];

      // Read ne, t
      // -------------------------------------------------------------------------
      ierr = H5LTread_dataset_double(file_id, "ne", scratch);
        MYH5CHECK(ierr);
      min_n = scratch[0]/min_Ye;
      max_n = scratch[m_nn-1]/max_Ye;
      for (int in = 0; in < m_nn; ++in) {
        m_log_ne[in] = log(scratch[in]);
      }
      m_id_log_ne = 1.0/(m_log_ne[1] - m_log_ne[0]);

      ierr = H5LTread_dataset_double(file_id, "t", scratch);
        MYH5CHECK(ierr);
      min_T = scratch[1];
      max_T = scratch[m_nt-1];
      for (int it = 0; it < m_nt; ++it) {
        m_log_t[it] = log(scratch[it]);
      }
      m_id_log_t = 1.0/(m_log_t[1] - m_log_t[0]);

      // the atomic mass unit is used as the baryon mass in the Helmholtz EOS
      ierr = H5LTread_dataset_double(file_id, "mb", scratch);
        MYH5CHECK(ierr);
      mb = scratch[0] * CGS.c*CGS.c; // in erg

      // Read other thermodynamics quantities
      // -------------------------------------------------------------------------
      ierr = H5LTread_dataset_double(file_id, "p", scratch);
        MYH5CHECK(ierr);
      copy(&scratch[0], &scratch[m_nn*m_nt], &m_table[index(ECP, 0, 0)]);

      ierr = H5LTread_dataset_double(file_id, "s", scratch);
        MYH5CHECK(ierr);
      copy(&scratch[0], &scratch[m_nn*m_nt], &m_table[index(ECENT, 0, 0)]);

      ierr = H5LTread_dataset_double(file_id, "eps", scratch);
        MYH5CHECK(ierr);
      copy(&scratch[0], &scratch[m_nn*m_nt], &m_table[index(ECEPS, 0, 0)]);

      ierr = H5LTread_dataset_double(file_id, "eta", scratch);
        MYH5CHECK(ierr);
      copy(&scratch[0], &scratch[m_nn*m_nt], &m_table[index(ECETA, 0, 0)]);

      ierr = H5LTread_dataset_double(file_id, "depsdt", scratch);
        MYH5CHECK(ierr);
      copy(&scratch[0], &scratch[m_nn*m_nt], &m_table[index(ECDEPSDT, 0, 0)]);

      ierr = H5LTread_dataset_double(file_id, "dpdn", scratch);
        MYH5CHECK(ierr);
      copy(&scratch[0], &scratch[m_nn*m_nt], &m_table[index(ECDPDN, 0, 0)]);

      ierr = H5LTread_dataset_double(file_id, "dpdt", scratch);
        MYH5CHECK(ierr);
      copy(&scratch[0], &scratch[m_nn*m_nt], &m_table[index(ECDPDT, 0, 0)]);

      // Mark table as read
      m_initialized = true;

      // Cleanup
      // -------------------------------------------------------------------------
      delete[] scratch;
      H5Fclose(file_id);

      // Now that we have read everything locally, we must populate
      // the aux static variables to share this data with other threads
      sm_id_log_ne = m_id_log_ne;
      sm_id_log_t = m_id_log_t;

      sm_nn = m_nn;
      sm_nt = m_nt;

      s_max_n = max_n;
      s_min_n = min_n;
      s_max_T = max_T;
      s_min_T = min_T;

    } // if (m_initialized==false)
  } // omp critical (EOSHelmholtz_ReadTable)

  // Disseminate applicable static variables to local memory
  m_id_log_ne = sm_id_log_ne;
  m_id_log_t  = sm_id_log_t;

  m_nn = sm_nn;
  m_nt = sm_nt;

  max_n = s_max_n;
  min_n = s_min_n;
  max_T = s_max_T;
  min_T = s_min_T;
  max_Y[SCYE] = max_Ye;
  min_Y[SCYE] = min_Ye;

}

void EOSHelmholtz::SetBaryonMass(Real new_mb) {
  mb = new_mb;
}


Real EOSHelmholtz::temperature_from_var(int iv, Real var, Real n, Real *Y) const {
  int in;
  Real wn0, wn1;
  weight_idx_ln(&wn0, &wn1, &in, log(n*Y[SCYE]));

  auto f = [=](int it){
    Real var_pt = wn0 * m_table[index(iv, in+0, it)]
             + wn1 * m_table[index(iv, in+1, it)];
    var_pt = add_rad_ion(iv, var_pt, n, exp(m_log_t[it]), Y);
    return var - var_pt;
  };

  int ilo = 0;
  int ihi = m_nt-1;
  Real flo = f(ilo);
  Real fhi = f(ihi);
  while (flo*fhi>0){
    if (ilo == ihi - 1) {
      break;
    } else {
      ilo += 1;
      flo = f(ilo);
    }
  }
  if (!(flo*fhi <= 0)) {

    Real flo_ = f(0);
    Real fhi_ = f(m_nt-1);

    std::cout<<"EOSHelmholtz::temperature_from_var failed to bracket root."<<std::endl;
    std::cout<<"iv: "<<iv<<std::endl;
    std::cout<<"var: "<<var<<std::endl;
    std::cout<<"n: "<<n<<std::endl;
    std::cout<<"Yq: "<<Y[SCYE]<<std::endl;
    std::cout<<"flo: "<<flo<<std::endl;
    std::cout<<"fhi: "<<fhi<<std::endl;
    std::cout<<"varlo: "<<var - flo<<std::endl;
    std::cout<<"varhi: "<<var - fhi<<std::endl;
  }
  assert(flo*fhi <= 0);
  while (ihi - ilo > 1) {
    int ip = ilo + (ihi - ilo)/2;
    Real fp = f(ip);
    if (fp*flo <= 0) {
      ihi = ip;
      fhi = fp;
    }
    else {
      ilo = ip;
      flo = fp;
    }
  }
  assert(ihi - ilo == 1);
  Real lthi = m_log_t[ihi];
  Real ltlo = m_log_t[ilo];

  if (flo == 0) {
    return exp(ltlo);
  }
  if (fhi == 0) {
    return exp(lthi);
  }

  Real lt = m_log_t[ilo] - flo*(lthi - ltlo)/(fhi - flo);
  return exp(lt);
}

Real EOSHelmholtz::add_rad_ion(int vi, Real var, Real n, Real T, Real *Y) const {
  switch (vi) {
    case ECP:
    {
      Real prad = asol / 3.0 * T*T*T*T;
      Real pion = n * inverse_abar(Y) * T;
      return var + prad + pion;
    }
    case ECENT:
    {
      // Sackur-Tetrode equation
      Real srad = 4.0 * asol/3.0 * T*T*T / n;
      Real Yn = Y[SCXN];
      Real Yp = Y[SCXP];
      Real Ya = Y[SCXA]/4;
      Real Yh = Y[SCXH]/Y[SCAH];
      Real mbar = mb*(1+Y[SCEB]);
      Real mh = (mbar - Yn*mn - Yp*mp - Ya*ma) / Yh;
      Real sn = Yn * (2.5 - log(n*Yn/g_n*pow(sac_const/(mn*T), 1.5)));
      Real sp = Yp * (2.5 - log(n*Yp/g_n*pow(sac_const/(mp*T), 1.5)));
      Real sa = Ya * (2.5 - log(n*Ya/g_a*pow(sac_const/(ma*T), 1.5)));
      Real sh = Yh * (2.5 - log(n*Yh/g_h*pow(sac_const/(mh*T), 1.5)));
      return var + srad + sn + sp + sa + sh;
     }
    case ECEPS:
    {
      Real erad = asol * T*T*T*T / (n*mb);
      Real eion = 1.5 * T * inverse_abar(Y) / mb;
      Real ebind = Y[SCEB];
      return var + erad + eion + ebind;
    }
    case ECDEPSDT:
    {
      Real deraddt = 4.0 * asol * T*T*T / n;
      Real deiondt = 1.5 * inverse_abar(Y);
      return var + deraddt + deiondt;
    }
    case ECDPDN:
    {
      Real dpraddn = 0.0;
      Real dpiondn = T * inverse_abar(Y);
      return var + dpraddn + dpiondn;
    }
    case ECDPDT:
    {
      Real dpraddt = 4.0/3.0 * asol * T*T*T;
      Real dpiondt = n * inverse_abar(Y);
      return var + dpraddt + dpiondt;
    }
    case ECETA:
    {
      return var; // no correction electron degeneracy parameter
    }
  }
  throw std::logic_error("Invalid variable index in add_rad_ion");
}


Real EOSHelmholtz::eval_at_nty(int vi, Real n, Real T, Real *Y) const {
  Real var = eval_at_lnty(vi, log(n*Y[SCYE]), log(T));
  return add_rad_ion(vi, var, n, T, Y);
}

void EOSHelmholtz::weight_idx_ln(Real *w0, Real *w1, int *in, Real log_n) const {
  *in = (log_n - m_log_ne[0])*m_id_log_ne;
  // if outside table limits, linearly extrapolate
  if(*in > m_nn-2){
     *in = m_nn-2;
  }else if(*in < 0 ) {
      *in = 0;
  }

  *w1 = (log_n - m_log_ne[*in])*m_id_log_ne;
  *w0 = 1.0 - (*w1);
}

void EOSHelmholtz::weight_idx_lt(Real *w0, Real *w1, int *it, Real log_t) const {
  *it = (log_t - m_log_t[0])*m_id_log_t;
  // if outside table limits, linearly extrapolate
  if(*it > m_nt-2){
      *it = m_nt-2;
  } else if(*it < 0 ) {
      *it = 0;
  }
  *w1 = (log_t - m_log_t[*it])*m_id_log_t;
  *w0 = 1.0 - (*w1);
}

Real EOSHelmholtz::eval_at_lnty(int iv, Real log_n, Real log_t) const {
  // This only returns the electron part
  int in, it;
  Real wn0, wn1, wt0, wt1;

  weight_idx_ln(&wn0, &wn1, &in, log_n);
  weight_idx_lt(&wt0, &wt1, &it, log_t);

  return
    wn0 * (wt0 * m_table[index(iv, in+0, it+0)]  +
           wt1 * m_table[index(iv, in+0, it+1)]) +
    wn1 * (wt0 * m_table[index(iv, in+1, it+0)]  +
           wt1 * m_table[index(iv, in+1, it+1)]);
}
