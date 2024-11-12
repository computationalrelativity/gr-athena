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


#define MYH5CHECK(ierr) \
  if(ierr < 0) { \
    stringstream ss; \
    ss << __FILE__ << ":" << __LINE__ << " error reading EOS table!"; \
    throw runtime_error(ss.str().c_str()); \
  }

EOSCompOSETransition::EOSCompOSETransition():
  m_id_log_nb(numeric_limits<Real>::quiet_NaN()),
  m_id_log_t(numeric_limits<Real>::quiet_NaN()),
  m_id_yq(numeric_limits<Real>::quiet_NaN()),
  m_nn(0), m_nt(0), m_ny(0),
  m_min_h(numeric_limits<Real>::max()),
  m_log_nb(nullptr),
  m_log_t(nullptr),
  m_yq(nullptr),
  m_table(nullptr),
  m_initialized(false),
  m_trans_T_width(numeric_limits<Real>::quiet_NaN()),
  m_trans_ln_width(numeric_limits<Real>::quiet_NaN()) {
  n_species = 2; // second is abar
  eos_units = &Nuclear;
  max_iter = 50;
  T_tol = 1e-10;
  // set manually or after table read
  trans_T_start = numeric_limits<Real>::quiet_NaN();
  trans_T_end = numeric_limits<Real>::quiet_NaN();
  trans_ln_start = numeric_limits<Real>::quiet_NaN();
  trans_ln_end = numeric_limits<Real>::quiet_NaN();
}

EOSCompOSETransition::~EOSCompOSETransition() {
  if (m_initialized) {
    delete[] m_log_nb;
    delete[] m_log_t;
    delete[] m_yq;
    delete[] m_table;
  }
}

Real EOSCompOSETransition::TemperatureFromEps(Real n, Real eps, Real *Y) {
  assert (m_initialized);
  return temperature_from_var(ECLOGE, log(eps), n, Y[0], Y[1]);
}

Real EOSCompOSETransition::TemperatureFromEps(Real n, Real eps, Real *Y, Real Tguess) {
  assert (m_initialized);
  return temperature_from_var_with_guess(ECLOGE, log(eps), n, Y[0], Y[1], Tguess);
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
  if (p <= 0.0) {
    printf("Pressure is %e\n", p);
  }
  assert (p > 0.);
  return temperature_from_var(ECLOGP, log(p), n, Y[0], Y[1]);
}

Real EOSCompOSETransition::TemperatureFromP(Real n, Real p, Real *Y, Real Tguess) {
  assert (m_initialized);
  return temperature_from_var_with_guess(ECLOGP, log(p), n, Y[0], Y[1], Tguess);
}

Real EOSCompOSETransition::Energy(Real n, Real T, Real *Y) {
  return (SpecificInternalEnergy(n, T, Y) + 1.0) * mb * n;
}

Real EOSCompOSETransition::Pressure(Real n, Real T, Real *Y) {
  assert (m_initialized);
  return exp(eval_at_nty(ECLOGP, n, T, Y[0], Y[1]));
}

Real EOSCompOSETransition::Entropy(Real n, Real T, Real *Y) {
  assert (m_initialized);
  return eval_at_nty(ECENT, n, T, Y[0], Y[1]);
}

Real EOSCompOSETransition::Abar(Real n, Real T, Real *Y) {
  assert (m_initialized);
  return eval_at_nty(ECABAR, n, T, Y[0], Y[1]);
}

Real EOSCompOSETransition::Enthalpy(Real n, Real T, Real *Y) {
  assert (m_initialized);
  Real const P = Pressure(n, T, Y);
  Real const e = Energy(n, T, Y);
  return (P + e)/n;
}

Real EOSCompOSETransition::SoundSpeed(Real n, Real T, Real *Y) {
  assert (m_initialized);
  return eval_at_nty(ECCS, n, T, Y[0], Y[1]);
}

Real EOSCompOSETransition::SpecificInternalEnergy(Real n, Real T, Real *Y) {
  assert (m_initialized);
  return exp(eval_at_nty(ECLOGE, n, T, Y[0], Y[1]));
}

Real EOSCompOSETransition::BaryonChemicalPotential(Real n, Real T, Real *Y) {
  assert (m_initialized);
  return eval_at_nty(ECMUB, n, T, Y[0], Y[1]);
}

Real EOSCompOSETransition::ChargeChemicalPotential(Real n, Real T, Real *Y) {
  assert (m_initialized);
  return eval_at_nty(ECMUQ, n, T, Y[0], Y[1]);
}

Real EOSCompOSETransition::ElectronLeptonChemicalPotential(Real n, Real T, Real *Y) {
  assert (m_initialized);
  return eval_at_nty(ECMUL, n, T, Y[0], Y[1]);
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

Real EOSCompOSETransition::MinimumEnergy(Real n, Real *Y) {
  return Energy(n, min_T, Y);
}

Real EOSCompOSETransition::MaximumEnergy(Real n, Real *Y) {
  return Energy(n, max_T, Y);
}

void EOSCompOSETransition::SetTemperatureTransition(Real T_start, Real T_end) {
  if (m_initialized) {
    std::stringstream msg;
    msg << "### EOSCompOSETransition: SetTemperatureTransition mus be called before InitializeTables.";
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

  if (T_end < exp(m_log_t[0])) {
    std::stringstream msg;
    msg << "### EOSCompOSETransition: temperature transition end: " << T_end <<
        " is less than CompOSE minimum temperature: " << exp(m_log_t[0]) << std::endl;
    // ATHENA_ERROR(msg);
    throw std::runtime_error(msg.str());
  }
  trans_T_start = T_start;
  trans_T_end = T_end;
  m_trans_T_width = T_start - T_end;
}

void EOSCompOSETransition::SetDensityTransition(Real n_start, Real n_end) {
  if (m_initialized) {
    std::stringstream msg;
    msg << "### EOSCompOSETransition: SetDensityTransition mus be called before InitializeTables.";
    // ATHENA_ERROR(msg);
    throw std::runtime_error(msg.str());
  }
  if (n_start <= n_end) {
    std::stringstream msg;
    msg << "### EOSCompOSETransition: density transition start: " << n_start <<
        " is not larger than end: " << n_end << std::endl;
    // ATHENA_ERROR(msg);
    throw std::runtime_error(msg.str());
  }

  if (n_end < exp(m_log_nb[0])) {
    std::stringstream msg;
    msg << "### EOSCompOSETransition: density transition end: " << n_end <<
        " is less than CompOSE minimum density: " << exp(m_log_nb[0]) << std::endl;
    // ATHENA_ERROR(msg);
    throw std::runtime_error(msg.str());
  }
  trans_ln_start = log(n_start);
  trans_ln_end = log(n_end);
  m_trans_ln_width = log(n_start/n_end);
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
  for (int in = 0; in < m_nn; ++in) {
    m_log_nb[in] = log(scratch[in]);
  }
  m_id_log_nb = 1.0/(m_log_nb[1] - m_log_nb[0]);

  ierr = H5LTread_dataset_double(file_id, "t", scratch);
    MYH5CHECK(ierr);
  min_T = scratch[1];
  max_T = scratch[m_nt-1];
  for (int it = 0; it < m_nt; ++it) {
    m_log_t[it] = log(scratch[it]);
  }
  m_id_log_t = 1.0/(m_log_t[1] - m_log_t[0]);

  ierr = H5LTread_dataset_double(file_id, "yq", scratch);
    MYH5CHECK(ierr);
  min_Y[0] = scratch[0];
  max_Y[0] = scratch[m_ny-1];
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
      scratch[index(0, in, iy, it)]; // Will be converted to log(E) later
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

void EOSCompOSETransition::update_baryon_mass() {
  Real Abar = 1.0;

  Real new_mb = mb;
  for (int in = 0; in < m_nn; ++in) {
    Real ln = m_log_nb[in];
    if (ln >= -8) break;
    for (int it = 0; it < m_nt; ++it) {
      Real lT = m_log_t[it];
      if (lT >= -1) break;
      if (ln > trans_ln_start and exp(lT) > trans_T_start) break;
      for (int iy = 0; iy < m_ny; ++iy) {
        Real ye = m_yq[iy];
        Real eps = m_table[index(ECLOGE, in, iy, it)]; // table is not yet log(eps)
        Real eps_helm =  exp(eval_helm_at_lnty(ECLOGE, ln, lT, ye, Abar));
        new_mb = min(mb*(1 + eps - eps_helm), new_mb);
      }
    }
  }

  // printf("mb,  new_mb = %f, %f\n", mb, new_mb);

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
    if ((lT < -1 and ln < trans_ln_start) or (ln < -8 and  exp(lT) < trans_T_start)) {
      Real y = m_yq[iy];
      Abar = m_table[index(ECABAR, in, iy, it)];
      Real eps_helm =  exp(eval_helm_at_lnty(ECLOGE, ln, lT, y, Abar));
      assert(new_eps >= eps_helm);
    }
  }}}
}

void EOSCompOSETransition::InitializeTables(std::string fname, std::string helm_fname) {
  #pragma omp critical (ReadCompOSETable)
  {
    read_compose_table(fname);
    // // Now that we have read everything locally, we must populate
    // // the aux static variables to share this data with other threads
    // sm_id_log_nb = m_id_log_nb;
    // sm_id_log_t = m_id_log_t;
    // sm_id_yq = m_id_yq;

    // sm_nn = m_nn;
    // sm_nt = m_nt;
    // sm_ny = m_ny;

    // sm_min_h = m_min_h;

    // s_mb = mb;
    // s_max_n = max_n;
    // s_min_n = min_n;
    // s_max_T = max_T;
    // s_min_T = min_T;
    // s_max_Y[0] = max_Y[0];
    // s_min_Y[0] = min_Y[0];
  } // omp critical (EOSCompOSE_ReadTable)

  // // Disseminate applicable static variables to local memory
  // m_id_log_nb = sm_id_log_nb;
  // m_id_log_t  = sm_id_log_t;
  // m_id_yq     = sm_id_yq;

  // m_nn = sm_nn;
  // m_nt = sm_nt;
  // m_ny = sm_ny;

  // m_min_h = sm_min_h;

  // mb       = s_mb;
  // max_n    = s_max_n;
  // min_n    = s_min_n;
  // max_T    = s_max_T;
  // min_T    = s_min_T;
  // max_Y[0] = s_max_Y[0];
  // min_Y[0] = s_min_Y[0];

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

  #pragma omp critical (ReadHelmholtzTable)
  {
    read_helmholtz_table(helm_fname);
  } // end omp critical ReadHelmholtzTable

  // Set the baryon mass in the Helmholtz EOS to the current mb
  Real mb_cgs = mb*eos_units->MassConversion(CGS);
  set_mb(&mb_cgs);

  update_baryon_mass();

  // Set limits for Abar
  min_Y[1] = 1.0;
  max_Y[1] = 500.0;

  // Check and update bounds
  // -------------------------------------------------------------------------
  const Real rho_trans = exp(trans_ln_start) * eos_units->DensityConversion(CGS)*mb*eos_units->MassConversion(CGS);
  const Real temp_trans = trans_T_start * eos_units->TemperatureConversion(CGS);
  bool success;
  check_bounds(&rho_trans, &temp_trans, &min_Y[0], &max_Y[0], &success);
  assert(success);

  Real rho_min, rho_max, temp_min, temp_max;
  get_bounds(&min_Y[0], &rho_min, &rho_max, &temp_min, &temp_max);

  assert(rho_max >= rho_trans);
  assert(temp_max >= temp_trans);

  min_n = rho_min / (eos_units->DensityConversion(CGS)*mb*eos_units->MassConversion(CGS));
  min_T = temp_min / (eos_units->TemperatureConversion(CGS)) * (1 + 1e-5);
  // printf("min_n, min_T = %e, %e\n", min_n, min_T);
  Real min_ln = log(min_n)+1e-9;
  Real min_lT = log(min_T)+1e-9;

  for (int iy = 0; iy < m_ny; ++iy) {
    // Eval helmholtz with maximum Abar because it corresponds to the minimum of eps
    Real eps = exp(eval_helm_at_lnty(ECLOGE, min_ln, min_lT, m_yq[iy], max_Y[1]));
    Real p = exp(eval_helm_at_lnty(ECLOGP, min_ln, min_lT, m_yq[iy], max_Y[1]));
    m_min_h = min(m_min_h, (1 + eps)*mb + p/min_n);
  }

  m_initialized = true;

  // printf("EOSCompOSETransition: Initialized\n");
  // printf("  min_n = %e\n", min_n);
  // printf("  max_n = %e\n", max_n);
  // printf("  min_T = %e\n", min_T);
  // printf("  max_T = %e\n", max_T);
  // printf("  min_Y = %e\n", min_Y[0]);
  // printf("  max_Y = %e\n", max_Y[0]);
  // printf("  min_Abar = %e\n", min_Y[1]);
  // printf("  max_Abar = %e\n", max_Y[1]);
  // printf("  min_h = %.15e MeV\n", m_min_h);
  // printf("  mb = %.15e MeV\n", mb);
  // printf("  T transition start = %e\n", trans_T_start);
  // printf("  T transition end = %e\n", trans_T_end);
  // printf("  n transition start = %e\n", exp(trans_ln_start));
  // printf("  n transition end = %e\n", exp(trans_ln_end));
  // printf("  dens conversion = %.15e\n", eos_units->DensityConversion(CGS)*mb*eos_units->MassConversion(CGS));
  // printf("  temp conversion = %.15e\n", eos_units->TemperatureConversion(CGS));
  // printf("  pres conversion = %.15e\n", CGS.PressureConversion(*eos_units));
  // printf("  inte conversion = %.15e\n", CGS.SpecificInternalEnergyConversion(*eos_units));
  // printf("  entr conversion = %.15e\n", CGS.EntropyConversion(*eos_units)*mb*eos_units->MassConversion(CGS));
}

Real EOSCompOSETransition::temperature_from_var(int iv, Real var, Real n, Real Yq, Real Abar) const {
  Real ln = log(n);

  if (std::isnan(var)) {
    printf("EOSCompOSETransition::temperature_from_var: var is NaN\n");
    printf("  iv = %d\n", iv);
  }

  auto func = [=](Real lT){
    Real res =  eval_at_lnty(iv, ln, lT, Yq, Abar);
    // check nans
    if (std::isnan(res)) {
      printf("EOSCompOSETransition: temperature_from_var: res is nan\n");
      printf("  iv = %d\n", iv);
      printf("  var = %e\n", var);
      printf("  n = %e\n", n);
      printf("  Yq = %e\n", Yq);
      printf("  Abar = %e\n", Abar);
      printf("  lT = %e\n", lT);
      printf("  ln = %e\n", ln);
      printf("  res = %e\n", res);
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

Real EOSCompOSETransition::temperature_from_var_with_guess(int iv, Real var, Real n, Real Yq, Real Abar, Real Tguess) const {
  auto func = [=](Real T){
    return eval_at_nty(iv, n, T, Yq, Abar) - var;
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
    return temperature_from_var(iv, var, n, Yq, Abar);
  }
  // printf("%4d ", n_iter);
  return (res.first + res.second)/2;
}

Real EOSCompOSETransition::eval_at_nty(int iv, Real n, Real T, Real Yq, Real Abar) const {
  return eval_at_lnty(iv, log(n), log(T), Yq, Abar);
}

Real EOSCompOSETransition::eval_at_lnty(int iv, Real ln, Real lT, Real Yq, Real Abar) const {
  Real T = exp(lT);

  if ((ln > trans_ln_start) and (T > trans_T_start)) {
    return eval_compose_at_lnty(iv, ln, lT, Yq);
  }
  if ((ln < trans_ln_end) or (T < trans_T_end)) {
    return eval_helm_at_lnty(iv, ln, lT, Yq, Abar);
}

  Real q_helmholtz = eval_helm_at_lnty(iv, ln, lT, Yq, Abar);
  Real q_compose = eval_compose_at_lnty(iv, ln, lT, Yq);
  Real w = 1;

  if ((T < trans_T_start) and (T > trans_T_end)) {
    w *= (T - trans_T_end)/m_trans_T_width;
  }
  if ((ln > trans_ln_end) and (ln < trans_ln_start)) {
    w *= (ln - trans_ln_end)/m_trans_ln_width;
  }

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

Real EOSCompOSETransition::eval_helm_at_lnty(int iv, Real ln, Real lT, Real Yq, Real Abar) const {
  Real rho = exp(ln) * eos_units->DensityConversion(CGS)*mb*eos_units->MassConversion(CGS);
  Real temp = exp(lT) * eos_units->TemperatureConversion(CGS);
  Real Zbar = Abar*Yq;

  Real etot = 0;
  Real ptot = 0;
  Real stot = 0;
  Real etaele = 0;
  Real etaion = 0;
  Real cs = 0;

  bool success_flag;

  // printf("n = %e, T = %e, Ye = %.5f, Abar = %.2f\n", exp(ln), exp(lT), Yq, Abar);
  // printf("rho = %.15e, temp = %.15e, Yq = %.5f, Abar = %.5f\n", rho, temp, Yq, Abar);

  helm_eos_wrap(
    &rho, &temp, &Abar, &Zbar,
    &etot, &ptot, &stot,
    &etaele, &etaion, &cs,
    &success_flag
  );


  if (!success_flag) {
    printf("Failed to evaluate helmholtz EOS\n");
    printf("ln = %e, lT = %e, Yq = %e\n", ln, lT, Yq);
    printf("rho = %e, temp = %e, Abar = %e, Zbar = %e\n", rho, temp, Abar, Zbar);
    printf("etot = %e, ptot = %e, stot = %e\n", etot, ptot, stot);
    return NAN;
  }

  switch (iv) {
    case ECLOGE:
      return log(etot*CGS.SpecificInternalEnergyConversion(*eos_units));
    case ECLOGP:
      return log(ptot*CGS.PressureConversion(*eos_units));
    case ECENT:
      return stot*CGS.EntropyConversion(*eos_units)*mb*eos_units->MassConversion(CGS);
    case ECCS:
      return cs*CGS.c/eos_units->c;
    case ECABAR:
      return Abar;
    // TODO the chemical pots are prob not correct
    // but are they even needed?
    case ECMUB:
      return etaion*exp(lT);
    case ECMUQ:
      return (etaele + etaion*Yq)*exp(lT);
    case ECMUL:
      return etaele*exp(lT);
  }

  printf("Invalid variable index %d\n", iv);
  return NAN;
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
