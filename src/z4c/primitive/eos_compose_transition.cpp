//! \file eos_compose.cpp
//  \brief Implementation of EOSCompose

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <hdf5.h>
#include <hdf5_hl.h>

#include "eos_compose_transition.hpp"
#include "numtools_root.hpp"
#include "unit_system.hpp"

using namespace Primitive;
using namespace std;

// #define USE_PACZYNSKI

#ifdef USE_PACZYNSKI
#define PH_ETOT pacz_etot
#define PH_PTOT pacz_ptot
#define PH_CS pacz_cs
#else
#define PH_ETOT helm_etot
#define PH_PTOT helm_ptot
#define PH_CS helm_cs
#endif

#define MYH5CHECK(ierr) \
  if(ierr < 0) { \
    stringstream ss; \
    ss << __FILE__ << ":" << __LINE__ << " error reading EOS table!"; \
    throw runtime_error(ss.str().c_str()); \
  }

EOSCompOSETransition::EOSCompOSETransition() {
  n_species = 2; // second is abar
  eos_units = &Nuclear;
  max_iter = 50;
  T_tol = 1e-10;
  min_Y[SCABAR] = 1.0;
  max_Y[SCABAR] = 500.0;
}

EOSCompOSETransition::~EOSCompOSETransition() {
  // if (m_initialized) {
  //   delete[] m_log_nb;
  //   delete[] m_log_t;
  //   delete[] m_yq;
  //   delete[] m_table;
  // }
}

//Definitions for static members
Real * EOSCompOSETransition::m_log_nb = nullptr;
Real * EOSCompOSETransition::m_log_t = nullptr;
Real * EOSCompOSETransition::m_yq = nullptr;
Real * EOSCompOSETransition::m_table = nullptr;

Real EOSCompOSETransition::m_id_log_nb = numeric_limits<Real>::quiet_NaN();
Real EOSCompOSETransition::m_id_log_t = numeric_limits<Real>::quiet_NaN();
Real EOSCompOSETransition::m_id_yq = numeric_limits<Real>::quiet_NaN();
Real EOSCompOSETransition::m_h_id_lt = numeric_limits<Real>::quiet_NaN();
Real EOSCompOSETransition::m_h_id_y = numeric_limits<Real>::quiet_NaN();
Real EOSCompOSETransition::m_h_id_ls = numeric_limits<Real>::quiet_NaN();

int EOSCompOSETransition::m_nn = 0;
int EOSCompOSETransition::m_nt = 0;
int EOSCompOSETransition::m_ny = 0;

int EOSCompOSETransition::m_h_nt = 0;
int EOSCompOSETransition::m_h_ny = 0;
int EOSCompOSETransition::m_h_ns = 0;

Real EOSCompOSETransition::mb = numeric_limits<Real>::quiet_NaN();
Real EOSCompOSETransition::max_n = numeric_limits<Real>::quiet_NaN();
Real EOSCompOSETransition::min_n = numeric_limits<Real>::quiet_NaN();
Real EOSCompOSETransition::max_T = numeric_limits<Real>::quiet_NaN();
Real EOSCompOSETransition::min_T = numeric_limits<Real>::quiet_NaN();
Real EOSCompOSETransition::max_Y[MAX_SPECIES] = {0};
Real EOSCompOSETransition::min_Y[MAX_SPECIES] = {0};

Real EOSCompOSETransition::h_max_s = numeric_limits<Real>::quiet_NaN();
Real EOSCompOSETransition::h_min_s = numeric_limits<Real>::quiet_NaN();
Real EOSCompOSETransition::h_max_y = numeric_limits<Real>::quiet_NaN();
Real EOSCompOSETransition::h_min_y = numeric_limits<Real>::quiet_NaN();
Real EOSCompOSETransition::h_max_t = numeric_limits<Real>::quiet_NaN();
Real EOSCompOSETransition::h_min_t = numeric_limits<Real>::quiet_NaN();

Real EOSCompOSETransition::m_min_h = numeric_limits<Real>::max();
Real EOSCompOSETransition::m_trans_T_width = numeric_limits<Real>::quiet_NaN();
Real EOSCompOSETransition::m_trans_ln_width = numeric_limits<Real>::quiet_NaN();
Real EOSCompOSETransition::trans_T_start = numeric_limits<Real>::quiet_NaN();
Real EOSCompOSETransition::trans_T_end = numeric_limits<Real>::quiet_NaN();
Real EOSCompOSETransition::trans_ln_start = numeric_limits<Real>::quiet_NaN();
Real EOSCompOSETransition::trans_ln_end = numeric_limits<Real>::quiet_NaN();
Real EOSCompOSETransition::helm_ln_max = numeric_limits<Real>::quiet_NaN();
Real EOSCompOSETransition::helm_lt_max = numeric_limits<Real>::quiet_NaN();
Real EOSCompOSETransition::comp_ln_min = numeric_limits<Real>::quiet_NaN();
Real EOSCompOSETransition::comp_lt_min = numeric_limits<Real>::quiet_NaN();
bool EOSCompOSETransition::m_initialized = false;

Real EOSCompOSETransition::TemperatureFromEps(Real n, Real eps, Real *Y) {
  assert (m_initialized);
  Real eps_min = MinimumSpecificInternalEnergy(n, Y);
  Real eps_max = MaximumSpecificInternalEnergy(n, Y);
  return (eps <= eps_min) ? min_T : (eps >= eps_max) ? max_T :
    temperature_from_var(ECLOGE, log(eps), n, Y[SCYE], Y[SCABAR], Y[SCEB]);
}

Real EOSCompOSETransition::TemperatureFromEps(Real n, Real eps, Real *Y, Real Tguess) {
  assert (m_initialized);
  Real eps_min = MinimumSpecificInternalEnergy(n, Y);
  Real eps_max = MaximumSpecificInternalEnergy(n, Y);
  return (eps <= eps_min) ? min_T : (eps >= eps_max) ? max_T :
    temperature_from_var_with_guess(ECLOGE, log(eps), n, Y[SCYE], Y[SCABAR], Y[SCEB], Tguess);
}

Real EOSCompOSETransition::TemperatureFromEntropy(Real n, Real s, Real *Y) {
  throw std::logic_error("EOSCompOSETransition::TemperatureFromEntropy not currently implemented.");
}

Real EOSCompOSETransition::TemperatureFromE(Real n, Real e, Real *Y) {
  assert (m_initialized);
  return TemperatureFromEps(n, e/(mb*n) - 1.0, Y);
}

Real EOSCompOSETransition::TemperatureFromE(Real n, Real e, Real *Y, Real Tguess) {
  assert (m_initialized);
  return TemperatureFromEps(n, e/(mb*n) - 1.0, Y, Tguess);
}

Real EOSCompOSETransition::TemperatureFromP(Real n, Real p, Real *Y) {
  assert (m_initialized);
  Real p_min = MinimumPressure(n, Y);
  Real p_max = MaximumPressure(n,Y);

  return (p <= p_min) ? min_T : (p >= p_max) ? max_T :
    temperature_from_var(ECLOGP, log(p), n, Y[SCYE], Y[SCABAR], Y[SCEB]);
}

Real EOSCompOSETransition::TemperatureFromP(Real n, Real p, Real *Y, Real Tguess) {
  assert (m_initialized);
  Real p_min = MinimumPressure(n, Y);
  Real p_max = MaximumPressure(n,Y);

  return (p <= p_min) ? min_T : (p >= p_max) ? max_T :
    temperature_from_var_with_guess(ECLOGP, log(p), n, Y[SCYE], Y[SCABAR], Y[SCEB], Tguess);
}

Real EOSCompOSETransition::Energy(Real n, Real T, Real *Y) {
  return (SpecificInternalEnergy(n, T, Y) + 1.0) * mb * n;
}

Real EOSCompOSETransition::Pressure(Real n, Real T, Real *Y) {
  assert (m_initialized);
  return exp(eval_at_nty(ECLOGP, n, T, Y[SCYE], Y[SCABAR], Y[SCEB]));
}

Real EOSCompOSETransition::Entropy(Real n, Real T, Real *Y) {
  assert (m_initialized);
  return eval_at_nty(ECENT, n, T, Y[SCYE], Y[SCABAR], Y[SCEB]);
}

Real EOSCompOSETransition::Abar(Real n, Real T, Real *Y) {
  assert (m_initialized);
  return eval_at_nty(ECABAR, n, T, Y[SCYE], Y[SCABAR], Y[SCEB]);
}

Real EOSCompOSETransition::Enthalpy(Real n, Real T, Real *Y) {
  assert (m_initialized);
  Real const P = Pressure(n, T, Y);
  Real const e = Energy(n, T, Y);
  return (P + e)/n;
}

Real EOSCompOSETransition::SoundSpeed(Real n, Real T, Real *Y) {
  assert (m_initialized);
  return eval_at_nty(ECCS, n, T, Y[SCYE], Y[SCABAR], Y[SCEB]);
}

Real EOSCompOSETransition::SpecificInternalEnergy(Real n, Real T, Real *Y) {
  assert (m_initialized);
  return exp(eval_at_nty(ECLOGE, n, T, Y[SCYE], Y[SCABAR], Y[SCEB]));
}

Real EOSCompOSETransition::BaryonChemicalPotential(Real n, Real T, Real *Y) {
  assert (m_initialized);
  return eval_at_nty(ECMUB, n, T, Y[SCYE], Y[SCABAR], Y[SCEB]);
}

Real EOSCompOSETransition::ChargeChemicalPotential(Real n, Real T, Real *Y) {
  assert (m_initialized);
  return eval_at_nty(ECMUQ, n, T, Y[SCYE], Y[SCABAR], Y[SCEB]);
}

Real EOSCompOSETransition::ElectronLeptonChemicalPotential(Real n, Real T, Real *Y) {
  assert (m_initialized);
  return eval_at_nty(ECMUL, n, T, Y[SCYE], Y[SCABAR], Y[SCEB]);
}

Real EOSCompOSETransition::FrYn(Real n, Real T, Real *Y) {
  throw std::logic_error("EOSCompOSETransition::FrYn not currently implemented.");
}

Real EOSCompOSETransition::FrYp(Real n, Real T, Real *Y) {
  throw std::logic_error("EOSCompOSETransition::FrYp not currently implemented.");
}

Real EOSCompOSETransition::FrYh(Real n, Real T, Real *Y) {
  throw std::logic_error("EOSCompOSETransition::FrYh not currently implemented.");
}

Real EOSCompOSETransition::AN(Real n, Real T, Real *Y) {
  throw std::logic_error("EOSCompOSETransition::AN not currently implemented.");
}

Real EOSCompOSETransition::ZN(Real n, Real T, Real *Y) {
  throw std::logic_error("EOSCompOSETransition::ZN not currently implemented.");
}

Real EOSCompOSETransition::MinimumEnthalpy() {
  return m_min_h;
}

Real EOSCompOSETransition::MinimumPressure(Real n, Real *Y) {
  return Pressure(n, min_T, Y);
}

Real EOSCompOSETransition::MaximumPressure(Real n, Real *Y) {
  return Pressure(n, max_T, Y);
}

Real EOSCompOSETransition::MinimumSpecificInternalEnergy(Real n, Real *Y) {
  return SpecificInternalEnergy(n, min_T, Y);
}

Real EOSCompOSETransition::MaximumSpecificInternalEnergy(Real n, Real *Y) {
  return SpecificInternalEnergy(n, max_T, Y);
}

Real EOSCompOSETransition::MinimumEnergy(Real n, Real *Y) {
  return Energy(n, min_T, Y);
}

Real EOSCompOSETransition::MaximumEnergy(Real n, Real *Y) {
  return Energy(n, max_T, Y);
}

void EOSCompOSETransition::SetTransition(Real n_start, Real n_end, Real T_start, Real T_end) {
  if (m_initialized) {
    std::stringstream msg;
    msg << "### EOSCompOSETransition: Transition must be set before initialization." << std::endl;
    throw std::runtime_error(msg.str());
  }

  if (n_start <= n_end) {
    std::stringstream msg;
    msg << "### EOSCompOSETransition: density transition start: " << n_start <<
        " is not larger than end: " << n_end << std::endl;
    // ATHENA_ERROR(msg);
    throw std::runtime_error(msg.str());
  }

  if (T_start <= T_end) {
    std::stringstream msg;
    msg << "### EOSCompOSETransition: temperature transition start: " << T_start <<
        " is not larger than end: " << T_end << std::endl;
    // ATHENA_ERROR(msg);
    throw std::runtime_error(msg.str());
  }

  // if (n_end < exp(m_log_nb[0])) {
  //   std::stringstream msg;
  //   msg << "### EOSCompOSETransition: density transition end: " << n_end <<
  //       " is less than CompOSE minimum density: " << exp(m_log_nb[0]) << std::endl;
  //   // ATHENA_ERROR(msg);
  //   throw std::runtime_error(msg.str());
  // }


  // if (T_end < exp(m_log_t[0])) {
  //   std::stringstream msg;
  //   msg << "### EOSCompOSETransition: temperature transition end: " << T_end <<
  //       " is less than CompOSE minimum temperature: " << exp(m_log_t[0]) << std::endl;
  //   // ATHENA_ERROR(msg);
  //   throw std::runtime_error(msg.str());
  // }

  trans_ln_start = log(n_start);
  trans_ln_end = log(n_end);
  m_trans_ln_width = log(n_start/n_end);

  trans_T_start = T_start;
  trans_T_end = T_end;
  m_trans_T_width = T_start - T_end;


  // if (m_initialized) {
  //   update_baryon_mass();
  //   update_bounds();
  // }
}

void EOSCompOSETransition::SetMaxIteration(int iter_max) {
  if (iter_max < 1) {
    std::stringstream msg;
    msg << "### EOSCompOSETransition: MaxIteration must be greater than 0." << std::endl;
    // ATHENA_ERROR(msg);
    throw std::runtime_error(msg.str());
  }
  max_iter = iter_max;
}

void EOSCompOSETransition::SetTemperatureTolerance(Real tol) {
  if (tol <= 0.0) {
    std::stringstream msg;
    msg << "### EOSCompOSETransition: TemperatureTolerance must be greater than 0." << std::endl;
    // ATHENA_ERROR(msg);
    throw std::runtime_error(msg.str());
  }
  T_tol = tol;
}

void EOSCompOSETransition::PrintParameters() {
  printf("EOSCompOSETransition:\n");
  printf("  min_n = %e\n", min_n);
  printf("  max_n = %e\n", max_n);
  printf("  min_T = %e\n", min_T);
  printf("  max_T = %e\n", max_T);
  printf("  min_Y = %e\n", min_Y[SCYE]);
  printf("  max_Y = %e\n", max_Y[SCYE]);
  printf("  helm_ln_max = %e\n", helm_ln_max);
  printf("  helm_lt_max = %e\n", helm_lt_max);
  printf("  max_Y = %e\n", max_Y[SCYE]);
  printf("  min_Abar = %e\n", min_Y[SCABAR]);
  printf("  max_Abar = %e\n", max_Y[SCABAR]);
  printf("  min_h = %.15e MeV\n", m_min_h);
  printf("  mb = %.15e MeV\n", mb);
  printf("  T transition start = %e\n", trans_T_start);
  printf("  T transition end = %e\n", trans_T_end);
  printf("  n transition start = %e\n", exp(trans_ln_start));
  printf("  n transition end = %e\n", exp(trans_ln_end));
  printf("  dens conversion = %.15e\n", eos_units->DensityConversion(CGS)*mb*eos_units->MassConversion(CGS));
  printf("  temp conversion = %.15e\n", eos_units->TemperatureConversion(CGS));
  printf("  pres conversion = %.15e\n", CGS.PressureConversion(*eos_units));
  printf("  inte conversion = %.15e\n", CGS.SpecificInternalEnergyConversion(*eos_units));
  printf("  entr conversion = %.15e\n", CGS.EntropyConversion(*eos_units)*mb*eos_units->MassConversion(CGS));
}

void EOSCompOSETransition::read_compose_table(std::string fname) {
  herr_t ierr;
  hid_t file_id;
  hsize_t snb, st, syq;

  // Open input file
  // -------------------------------------------------------------------------
  file_id = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    MYH5CHECK(file_id);

  // Get dataset sizes
  // -------------------------------------------------------------------------
  ierr = H5LTget_dataset_info(file_id, "nb", &snb, NULL, NULL);
    MYH5CHECK(ierr);
  ierr = H5LTget_dataset_info(file_id, "t", &st, NULL, NULL);
    MYH5CHECK(ierr);
  ierr = H5LTget_dataset_info(file_id, "yq", &syq, NULL, NULL);
    MYH5CHECK(ierr);
  m_nn = snb;
  m_nt = st;
  m_ny = syq;

  // Allocate memory
  // -------------------------------------------------------------------------
  m_log_nb = new Real[m_nn];
  m_log_t = new Real[m_nt];
  m_yq = new Real[m_ny];
  m_table = new Real[ECNVARS*m_nn*m_ny*m_nt];
  double * scratch = new double[m_nn*m_ny*m_nt];

  // Read nb, t, yq
  // -------------------------------------------------------------------------
  ierr = H5LTread_dataset_double(file_id, "nb", scratch);
    MYH5CHECK(ierr);
  min_n = scratch[0];
  max_n = scratch[m_nn-1];
  comp_ln_min = log(min_n);
  for (int in = 0; in < m_nn; ++in) {
    m_log_nb[in] = log(scratch[in]);
  }
  m_id_log_nb = 1.0/(m_log_nb[1] - m_log_nb[0]);

  ierr = H5LTread_dataset_double(file_id, "t", scratch);
    MYH5CHECK(ierr);
  min_T = scratch[1];
  max_T = scratch[m_nt-1];
  comp_lt_min = log(min_T);
  for (int it = 0; it < m_nt; ++it) {
    m_log_t[it] = log(scratch[it]);
  }
  m_id_log_t = 1.0/(m_log_t[1] - m_log_t[0]);

  ierr = H5LTread_dataset_double(file_id, "yq", scratch);
    MYH5CHECK(ierr);
  min_Y[SCYE] = scratch[0];
  max_Y[SCYE] = scratch[m_ny-1];
  for (int iy = 0; iy < m_ny; ++iy) {
    m_yq[iy] = scratch[iy];
  }
  m_id_yq = 1.0/(m_yq[1] - m_yq[0]);

  // the neutron mass is used as the baryon mass in CompOSE
  ierr = H5LTread_dataset_double(file_id, "mn", scratch);
    MYH5CHECK(ierr);
  mb = scratch[0];

  // Read other thermodynamics quantities
  // -------------------------------------------------------------------------
  ierr = H5LTread_dataset_double(file_id, "Q1", scratch);
    MYH5CHECK(ierr);
  for (int inb = 0; inb < m_nn; ++inb) {
  for (int iyq = 0; iyq < m_ny; ++iyq) {
  for (int it = 0; it < m_nt; ++it) {
    m_table[index(ECLOGP, inb, iyq, it)] =
        log(scratch[index(0, inb, iyq, it)]) + m_log_nb[inb];
  }}}

  ierr = H5LTread_dataset_double(file_id, "Q2", scratch);
    MYH5CHECK(ierr);
  copy(&scratch[0], &scratch[m_nn*m_ny*m_nt], &m_table[index(ECENT, 0, 0, 0)]);

  ierr = H5LTread_dataset_double(file_id, "Q3", scratch);
    MYH5CHECK(ierr);
  for (int in = 0; in < m_nn; ++in) {
  for (int iy = 0; iy < m_ny; ++iy) {
  for (int it = 0; it < m_nt; ++it) {
    m_table[index(ECMUB, in, iy, it)] =
      mb*(scratch[index(0, in, iy, it)] + 1);
  }}}

  ierr = H5LTread_dataset_double(file_id, "Q4", scratch);
    MYH5CHECK(ierr);
  for (int in = 0; in < m_nn; ++in) {
  for (int iy = 0; iy < m_ny; ++iy) {
  for (int it = 0; it < m_nt; ++it) {
    m_table[index(ECMUQ, in, iy, it)] = mb*scratch[index(0, in, iy, it)];
  }}}

  ierr = H5LTread_dataset_double(file_id, "Q5", scratch);
    MYH5CHECK(ierr);
  for (int in = 0; in < m_nn; ++in) {
  for (int iy = 0; iy < m_ny; ++iy) {
  for (int it = 0; it < m_nt; ++it) {
    m_table[index(ECMUL, in, iy, it)] = mb*scratch[index(0, in, iy, it)];
  }}}

  ierr = H5LTread_dataset_double(file_id, "Q7", scratch);
    MYH5CHECK(ierr);
  for (int in = 0; in < m_nn; ++in) {
  for (int iy = 0; iy < m_ny; ++iy) {
  for (int it = 0; it < m_nt; ++it) {
    m_table[index(ECLOGE, in, iy, it)] =
      scratch[index(0, in, iy, it)]; // Will be converted to log(eps) later
  }}}

  ierr = H5LTread_dataset_double(file_id, "cs2", scratch);
    MYH5CHECK(ierr);
  for (int in = 0; in < m_nn; ++in) {
  for (int iy = 0; iy < m_ny; ++iy) {
  for (int it = 0; it < m_nt; ++it) {
    m_table[index(ECCS, in, iy, it)] = sqrt(scratch[index(0, in, iy, it)]);
  }}}

  ierr = H5LTread_dataset_double(file_id, "Abar", scratch);
    MYH5CHECK(ierr);
  copy(&scratch[0], &scratch[m_nn*m_ny*m_nt], &m_table[index(ECABAR, 0, 0, 0)]);

  // Cleanup
  // -------------------------------------------------------------------------
  delete[] scratch;
  H5Fclose(file_id);
}

void EOSCompOSETransition::read_helmholtz_table(std::string fname) {
  int str_len = fname.length();
  read_helm_table(fname.c_str(), &str_len);
}

void EOSCompOSETransition::read_heating_table(std::string fname) {
  herr_t ierr;
  hid_t file_id;
  hsize_t size[3];

  // Open input file
  // -------------------------------------------------------------------------
  file_id = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    MYH5CHECK(file_id);

    static Real * m_heating_s;
    static Real * m_heating_ye;
    static Real * m_heating_tau;
    static Real * m_heating_table;

  // Get dataset sizes
  // -------------------------------------------------------------------------
  ierr = H5LTget_dataset_info(file_id, "ent", size, NULL, NULL);
    MYH5CHECK(ierr);
  m_h_nt = size[0];
  m_h_ny = size[1];
  m_h_ns = size[2];

  // Allocate memory
  // -------------------------------------------------------------------------
  m_heating_tau = new Real[m_h_nt];
  m_heating_ye = new Real[m_h_ny];
  m_heating_s = new Real[m_h_ns];
  m_heating_table = new Real[HNVARS*m_h_ns*m_h_ny*m_h_nt];
  double * scratch = new double[m_h_ns*m_h_ny*m_h_nt];

  // Read tau, ye, s
  ierr = H5LTread_dataset_double(file_id, "tau", scratch);
    MYH5CHECK(ierr);
  h_min_t = scratch[h_index(0,0,0,0)];
  h_max_t = scratch[h_index(0,m_h_nt-1,m_h_ny-1,m_h_ns-1)];
  for (int it = 0; it < m_h_nt; ++it) {
    m_heating_tau[it] = scratch[h_index(0,it,0,0)];
  }
  m_h_id_lt = 1.0/(m_heating_tau[1] - m_heating_tau[0]);

  ierr = H5LTread_dataset_double(file_id, "ye", scratch);
    MYH5CHECK(ierr);
  h_min_y = scratch[h_index(0,0,0,0)];
  h_max_y = scratch[h_index(0,m_h_nt-1,m_h_ny-1,m_h_ns-1)];
  for (int iy = 0; iy < m_h_ny; ++iy) {
    m_heating_ye[iy] = scratch[h_index(0,0,iy,0)];
  }
  m_h_id_y = 1.0/(m_heating_ye[1] - m_heating_ye[0]);

  ierr = H5LTread_dataset_double(file_id, "s", scratch);
    MYH5CHECK(ierr);
  h_min_s = scratch[h_index(0, 0, 0, 0)];
  h_max_s = scratch[h_index(0, m_h_nt-1, m_h_ny-1, m_h_ns-1)];
  for (int is = 0; is < m_h_ns; ++is) {
    m_heating_s[is] = scratch[h_index(0,0,0,is)];
  }
  m_h_id_ls = 1.0/(m_heating_s[1] - m_heating_s[0]);

  ierr = H5LTread_dataset_double(file_id, "A", scratch);
    MYH5CHECK(ierr);
  for (int it = 0; it < m_nn; ++it) {
  for (int iy = 0; iy < m_ny; ++iy) {
  for (int is = 0; is < m_nt; ++is) {
    m_table[h_index(HA, it, iy, is)] = scratch[h_index(0, is, iy, is)];
  }}}

  ierr = H5LTread_dataset_double(file_id, "alpha", scratch);
    MYH5CHECK(ierr);
  for (int it = 0; it < m_nn; ++it) {
  for (int iy = 0; iy < m_ny; ++iy) {
  for (int is = 0; is < m_nt; ++is) {
    m_table[h_index(HALPHA, it, iy, is)] = scratch[h_index(0, is, iy, is)];
  }}}

  ierr = H5LTread_dataset_double(file_id, "sigma", scratch);
    MYH5CHECK(ierr);
  for (int it = 0; it < m_nn; ++it) {
  for (int iy = 0; iy < m_ny; ++iy) {
  for (int is = 0; is < m_nt; ++is) {
    m_table[h_index(HSIGMA, it, iy, is)] = scratch[h_index(0, is, iy, is)];
  }}}

  ierr = H5LTread_dataset_double(file_id, "t_0", scratch);
    MYH5CHECK(ierr);
  for (int it = 0; it < m_nn; ++it) {
  for (int iy = 0; iy < m_ny; ++iy) {
  for (int is = 0; is < m_nt; ++is) {
    m_table[h_index(HT0, it, iy, is)] = scratch[h_index(0, is, iy, is)];
  }}}
}

void EOSCompOSETransition::update_baryon_mass(Real new_mb) {
  Real Abar = 1.0;

  // Real new_mb = mb;
  // for (int in = 0; in < m_nn; ++in) {
  // Real ln = m_log_nb[in];
  // if (ln > helm_ln_max) continue;
  // for (int it = 0; it < m_nt; ++it) {
  // Real lT = m_log_t[it];
  // if (lT > helm_lt_max) continue;
  // for (int iy = 0; iy < m_ny; ++iy) {
  // Real ye = m_yq[iy];
  // Real eps = m_table[index(ECLOGE, in, iy, it)];
  // Real eps_helm = exp(eval_helm_at_lnty(ECLOGE, ln, lT, ye, Abar));
  // new_mb = min(mb*(1 + eps - eps_helm), new_mb);
  // }
  // }
  // }

  // Update the baryon mass
  Real mb_ratio = mb/new_mb;
  mb = new_mb;
  Real mb_cgs = mb*eos_units->MassConversion(CGS);
  set_mb(&mb_cgs);

  // Update the internal energy
  for (int in = 0; in < m_nn; ++in) {
  for (int it = 0; it < m_nt; ++it) {
  for (int iy = 0; iy < m_ny; ++iy) {
    Real eps = m_table[index(ECLOGE, in, iy, it)];
    Real new_eps = mb_ratio*eps + mb_ratio - 1.0;
    m_table[index(ECLOGE, in, iy, it)] = log(new_eps);

    // if (ln <= trans_ln_start and exp(lT) <= trans_T_start) assert(new_eps >= eps_helm);
    Real lT = m_log_t[it];
    Real ln = m_log_nb[in];
    if ((lT < helm_lt_max and ln < trans_ln_start) or (ln < helm_ln_max and  exp(lT) < trans_T_start)) {
      Real y = m_yq[iy];
      Abar = m_table[index(ECABAR, in, iy, it)];
      Real eps_helm =  exp(eval_helm_at_lnty(ECLOGE, ln, lT, y, Abar, 0));
      if (new_eps < eps_helm) {
        printf("ln=%.10e lt=%.10e y=%.10e, new_eps=%.10e, eps_helm=%.10e \n", ln, lT, y, new_eps, eps_helm);
      }
    }
  }}}
}

void EOSCompOSETransition::update_bounds() {
  // Check and update bounds
  // -------------------------------------------------------------------------
  const Real rho_trans = exp(trans_ln_start)
    * eos_units->DensityConversion(CGS)*mb*eos_units->MassConversion(CGS);
  const Real temp_trans = trans_T_start
    * eos_units->TemperatureConversion(CGS);

#ifdef USE_PACZYNSKI
  // there are no constraints on the lower bounds
  min_n = 1e-35;
  min_T = 1e-10;
  helm_ln_max = log(max_n);
  helm_lt_max = log(max_T);
#else
  bool success;
  check_bounds(&rho_trans, &temp_trans, &max_Y[SCYE], &success);
  assert(success);

  Real rho_min, rho_max, temp_min, temp_max;
  get_bounds(&min_Y[SCYE], &rho_min, &rho_max, &temp_min, &temp_max);

  assert(rho_max >= rho_trans);
  assert(temp_max >= temp_trans);

  helm_ln_max = log(rho_max
                    *CGS.DensityConversion(*eos_units)
                    *CGS.MassConversion(*eos_units)/mb);

  helm_lt_max = log(temp_max
                    *CGS.TemperatureConversion(*eos_units));

  min_n = rho_min / (eos_units->DensityConversion(CGS)
                     * mb * eos_units->MassConversion(CGS));
  min_T = temp_min / (eos_units->TemperatureConversion(CGS)) * (1 + 1e-5);
#endif

  Real min_ln = log(min_n)+1e-9;
  Real min_lT = log(min_T)+1e-9;

  for (int iy = 0; iy < m_ny; ++iy) {
    // Eval helmholtz with maximum Abar because it corresponds to the minimum of eps
    Real eps = exp(eval_helm_at_lnty(ECLOGE, min_ln, min_lT, m_yq[iy], max_Y[SCABAR], 0));
    Real p = exp(eval_helm_at_lnty(ECLOGP, min_ln, min_lT, m_yq[iy], max_Y[SCABAR], 0));
    m_min_h = min(m_min_h, (1 + eps)*mb + p/min_n);
  }
}

Real EOSCompOSETransition::GetBindingEnergy(Real n, Real T, Real *Y) {
  Real ln = log(n);
  Real lt = log(T);
  if (ln > helm_ln_max or lt > helm_lt_max) {
    return 0.0;
  } else if (ln < comp_ln_min or lt < comp_lt_min) {
    return Y[SCEB];
  } else {
    Real Abar = Y[SCABAR];
    Real eps_helm = exp(eval_helm_at_lnty(ECLOGE, ln, lt, Y[SCYE], Abar, 0));
    Real eps_comp = exp(eval_compose_at_lnty(ECLOGE, ln, lt, Y[SCYE]));
    return eps_comp - eps_helm;
  }
}

Real EOSCompOSETransition::HeatingRate(Real tau, Real ye, Real s, Real t) {
  // Fit for the heating rate at early times
  //  A * (0.5-arctan((t-t_0)/sigma)/pi)^alpha
  // https://arxiv.org/pdf/2111.06870.pdf (eq 2)

  tau *= code_units->TimeConversion(CGS);;
  t *= code_units->TimeConversion(CGS);;
  const Real ls = log10(s);
  const Real ltau = log10(tau);
  // interpolate the parameters of the fit from the table
  Real A = eval_heating_at_ltys(HA, ltau, ye, ls);
  Real alpha = eval_heating_at_ltys(HALPHA, ltau, ye, ls);
  Real sigma = eval_heating_at_ltys(HSIGMA, ltau, ye, ls);
  Real t_0 = eval_heating_at_ltys(HT0, ltau, ye, ls);
  Real rate = A * pow(0.5 - atan((t-t_0)/sigma)/M_PI, alpha);
  return rate*CGS.SpecificInternalEnergyConversion(*code_units)/CGS.TimeConversion(*code_units);
}

void EOSCompOSETransition::InitializeTables(std::string fname, std::string helm_fname, std::string heating_fname, Real baryon_mass) {
  #pragma omp critical
  {
    if (not m_initialized) {

      read_compose_table(fname);
#ifndef USE_PACZYNSKI
      read_helmholtz_table(helm_fname);
#endif
      read_heating_table(heating_fname);

      // Set the baryon mass in the Helmholtz EOS to the current mb
      Real mb_cgs = mb*eos_units->MassConversion(CGS);
      set_mb(&mb_cgs);

      // Initialize the transitions if they have not been initialized
      // -------------------------------------------------------------------------
      if (std::isnan(m_trans_T_width)) {
        m_trans_T_width = 1e9 * CGS.TemperatureConversion(*eos_units); // 1 GK
        trans_T_end = min_T;
        trans_T_start = min_T + m_trans_T_width;
      }

      if (std::isnan(m_trans_ln_width)) {
        m_trans_ln_width = log(5); // start = 5 times end of transition
        trans_ln_end = log(min_n);
        trans_ln_start = trans_ln_end + m_trans_ln_width;
      }

      update_bounds();
      update_baryon_mass(baryon_mass);
      update_bounds();

      m_initialized = true;
      PrintParameters();
    }
  }
}

Real EOSCompOSETransition::temperature_from_var(int iv, Real var, Real n, Real Yq, Real Abar, Real Eb) const {
  Real ln = log(n);

  if (std::isnan(var)) {
    printf("EOSCompOSETransition::temperature_from_var: var is NaN\n");
    printf("  iv = %d\n", iv);
  }

  auto func = [=](Real lT){
    Real res =  eval_at_lnty(iv, ln, lT, Yq, Abar, Eb);
    // check nans
    if (std::isnan(res)) {
      std::stringstream msg;
      msg << "EOSCompOSETransition: temperature_from_var: res is nan\n"
          << "  iv = " << iv << "\n"
          << "  var = " << var << "\n"
          << "  n = " << n << "\n"
          << "  Yq = " << Yq << "\n"
          << "  Abar = " << Abar << "\n"
          << "  Eb = " << Eb << "\n"
          << "  lT = " << lT << "\n"
          << "  ln = " << ln << "\n"
          << "  res = " << res;
        throw std::runtime_error(msg.str());
    }
    return res - var;
  };

  auto tol = [=] (Real lTa, Real lTb) {
    return lTb - lTa < T_tol;
  };

  Real lower = log(min_T);
  Real upper = log(max_T);

  if (func(lower)*func(upper) > 0) {
    // printf("%4d ", max_iter);
    if (func(lower) > 0) {
      return min_T;
    } else {
      return max_T;
  }
    }

  boost::uintmax_t n_iter = max_iter;  // Maximum iterations

  try  {
    std::pair<Real, Real> res = boost::math::tools::toms748_solve(func, lower, upper, tol, n_iter);
    return exp((res.first + res.second)/2);
  }
  catch (boost::wrapexcept<std::domain_error>) {
    printf("Caught domain error\n");
    printf("lower = %e\n", lower);
    printf("upper = %e\n", upper);
    printf("func(lower) = %e\n", func(lower));
    printf("func(uppper) = %e\n", func(upper));
    throw;
  }
  // printf("%4d ", n_iter);
}

Real EOSCompOSETransition::temperature_from_var_with_guess(int iv, Real var, Real n, Real Yq, Real Abar, Real Eb, Real Tguess) const {
  auto func = [=](Real T){
    return eval_at_nty(iv, n, T, Yq, Abar, Eb) - var;
  };

  auto tol = [=] (Real Ta, Real Tb) {
    return (Tb - Ta)/(Ta + Tb) < T_tol;
  };

  boost::uintmax_t n_iter = max_iter;  // Maximum iterations

  std::pair<Real, Real> res;
  try {
    res = boost::math::tools::bracket_and_solve_root(func, Tguess, 1.2, true, tol, n_iter);
  }
  catch (boost::math::evaluation_error) {
    return temperature_from_var(iv, var, n, Yq, Abar, Eb);
  }
  // printf("%4d ", n_iter);
  return (res.first + res.second)/2;
}

Real EOSCompOSETransition::eval_at_nty(int iv, Real n, Real T, Real Yq, Real Abar, Real Eb) const {
  return eval_at_lnty(iv, log(n), log(T), Yq, Abar, Eb);
}

Real EOSCompOSETransition::TransitionFactor(Real n, Real T) const {
  Real ln = log(n);
  if (((ln > trans_ln_start) and (T > trans_T_start))
      or ln > helm_ln_max){
    return 0.0;
  }
  if ((ln < trans_ln_end) or (T < trans_T_end)) {
    return 1.0;
}

  Real w = 1;
  if ((T < trans_T_start) and (T > trans_T_end)) {
    w *= (T - trans_T_end)/m_trans_T_width;
  }
  if ((ln > trans_ln_end) and (ln < trans_ln_start)) {
    w *= (ln - trans_ln_end)/m_trans_ln_width;
  }
  return w;
}

Real EOSCompOSETransition::eval_at_lnty(int iv, Real ln, Real lT, Real Yq, Real Abar, Real Eb) const {
  Real T = exp(lT);
  Real n = exp(ln);

  Real w = TransitionFactor(n, T);

  if (w == 0.0) {
    return eval_compose_at_lnty(iv, ln, lT, Yq);
  }
  if (w == 1.0) {
    return eval_helm_at_lnty(iv, ln, lT, Yq, Abar, Eb);
}
  Real q_helmholtz = eval_helm_at_lnty(iv, ln, lT, Yq, Abar, Eb);
  Real q_compose = eval_compose_at_lnty(iv, ln, lT, Yq);
  return q_helmholtz*(1-w) + q_compose*w;
}

Real EOSCompOSETransition::eval_compose_at_lnty(int iv, Real ln, Real lT, Real Yq) const {
  int in, iy, it;
  Real wn0, wn1, wy0, wy1, wt0, wt1;

  weight_idx_ln(&wn0, &wn1, &in, ln);
  weight_idx_yq(&wy0, &wy1, &iy, Yq);
  weight_idx_lt(&wt0, &wt1, &it, lT);

  return
    wn0 * (wy0 * (wt0 * m_table[index(iv, in+0, iy+0, it+0)]   +
                  wt1 * m_table[index(iv, in+0, iy+0, it+1)])  +
           wy1 * (wt0 * m_table[index(iv, in+0, iy+1, it+0)]   +
                  wt1 * m_table[index(iv, in+0, iy+1, it+1)])) +
    wn1 * (wy0 * (wt0 * m_table[index(iv, in+1, iy+0, it+0)]   +
                  wt1 * m_table[index(iv, in+1, iy+0, it+1)])  +
           wy1 * (wt0 * m_table[index(iv, in+1, iy+1, it+0)]   +
                  wt1 * m_table[index(iv, in+1, iy+1, it+1)]));
}

Real EOSCompOSETransition::eval_helm_at_lnty(int iv, Real ln, Real lT, Real Yq, Real Abar, Real Eb) const {
  Real rho = exp(ln) * eos_units->DensityConversion(CGS)*mb*eos_units->MassConversion(CGS);
  Real temp = exp(lT) * eos_units->TemperatureConversion(CGS);
  Real Zbar = Abar*Yq;

  Real etot;
  Real ptot;
  Real cs;
  Real stot = 0;
  Real etaele = 0;
  Real etaion = 0;
  Real res;

  bool success_flag = false;

  switch (iv) {
    case ECLOGE:
      PH_ETOT(&rho, &temp, &Abar, &Zbar, &etot, &success_flag);
      res = log(etot*CGS.SpecificInternalEnergyConversion(*eos_units) + Eb);
      break;
    case ECLOGP:
      PH_PTOT(&rho, &temp, &Abar, &Zbar, &ptot, &success_flag);
      res = log(ptot*CGS.PressureConversion(*eos_units));
      break;
    case ECCS:
      PH_CS(&rho, &temp, &Abar, &Zbar, &cs, &success_flag);
      res = cs/CGS.c*eos_units->c;
      break;
    case ECABAR:
      res = Abar;
      break;
    // TODO add these but are they even needed?
    case ECENT:
      res = 0;
      break;
    case ECMUB:
      res = 0;
      break;
    case ECMUQ:
      res = 0;
      break;
    case ECMUL:
      res = 0;
      break;
  }
  return res;

  if (not success_flag) {
    std::ostringstream oss;
    oss << "Failed to evaluate helmholtz EOS\n";
    oss << "ln = " << ln << ", lT = " << lT << ", Yq = " << Yq << "\n";
    oss << "rho = " << rho << ", temp = " << temp << ", Abar = " << Abar << ", Zbar = " << Zbar << "\n";
    oss << "etot = " << etot << ", ptot = " << ptot << ", stot = " << stot;

    throw std::runtime_error(oss.str());
  } else {
    return res;
  }
}


void EOSCompOSETransition::weight_idx_ln(Real *w0, Real *w1, int *in, Real log_n) const {
  *in = (log_n - m_log_nb[0])*m_id_log_nb;
  *w1 = (log_n - m_log_nb[*in])*m_id_log_nb;
  *w0 = 1.0 - (*w1);
}

void EOSCompOSETransition::weight_idx_yq(Real *w0, Real *w1, int *iy, Real yq) const {
  *iy = (yq - m_yq[0])*m_id_yq;
  *w1 = (yq - m_yq[*iy])*m_id_yq;
  *w0 = 1.0 - (*w1);
}

void EOSCompOSETransition::weight_idx_lt(Real *w0, Real *w1, int *it, Real log_t) const {
  *it = (log_t - m_log_t[0])*m_id_log_t;
  *w1 = (log_t - m_log_t[*it])*m_id_log_t;
  *w0 = 1.0 - (*w1);
}

Real EOSCompOSETransition::eval_heating_at_ltys(int iv, Real lt, Real ye, Real ls) const {
  int it, iy, is;
  Real wt0, wt1, wy0, wy1, ws0, ws1;

  h_weight_idx_lt(&wt0, &wt1, &it, lt);
  h_weight_idx_ye(&wy0, &wy1, &iy, ye);
  h_weight_idx_ls(&ws0, &ws1, &is, ls);

  return
    wt0 * (wy0 * (ws0 * m_table[index(iv, it+0, iy+0, is+0)]   +
                  ws1 * m_table[index(iv, it+0, iy+0, is+1)])  +
           wy1 * (ws0 * m_table[index(iv, it+0, iy+1, is+0)]   +
                  ws1 * m_table[index(iv, it+0, iy+1, is+1)])) +
    wt1 * (wy0 * (ws0 * m_table[index(iv, it+1, iy+0, is+0)]   +
                  ws1 * m_table[index(iv, it+1, iy+0, is+1)])  +
           wy1 * (ws0 * m_table[index(iv, it+1, iy+1, is+0)]   +
                  ws1 * m_table[index(iv, it+1, iy+1, is+1)]));
}

void EOSCompOSETransition::h_weight_idx_lt(Real *w0, Real *w1, int *it, Real log_tau) const {
  *it = (log_tau - m_heating_ltau[0])*m_h_id_lt;
  *w1 = (log_tau - m_heating_ltau[*it])*m_h_id_lt;
  *w0 = 1.0 - (*w1);
}

void EOSCompOSETransition::h_weight_idx_ye(Real *w0, Real *w1, int *iy, Real ye) const {
  *iy = (ye - m_heating_ye[0])*m_h_id_y;
  *w1 = (ye - m_heating_ye[*iy])*m_h_id_y;
  *w0 = 1.0 - (*w1);
}

void EOSCompOSETransition::h_weight_idx_ls(Real *w0, Real *w1, int *is, Real log_s) const {
  *is = (log_s - m_heating_ls[0])*m_h_id_ls;
  *w1 = (log_s - m_heating_ls[*is])*m_h_id_ls;
  *w0 = 1.0 - (*w1);
}
