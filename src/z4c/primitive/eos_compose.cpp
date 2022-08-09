//! \file eos_compose.cpp
//  \brief Implementation of EOSCompose

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <sstream>
#include <stdexcept>

#ifdef HDF5OUTPUT
#include <hdf5.h>
#include <hdf5_hl.h>

#include "eos_compose.hpp"
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

EOSCompOSE::EOSCompOSE():
  m_id_log_nb(numeric_limits<Real>::quiet_NaN()),
  m_id_log_t(numeric_limits<Real>::quiet_NaN()),
  m_id_yq(numeric_limits<Real>::quiet_NaN()),
  m_nn(0), m_nt(0), m_ny(0),
  m_min_h(numeric_limits<Real>::max()),
  m_log_nb(nullptr),
  m_log_t(nullptr),
  m_yq(nullptr),
  m_table(nullptr),
  m_initialized(false) {
  n_species = 1;
  eos_units = &Nuclear;
}

EOSCompOSE::~EOSCompOSE() {
  if (m_initialized) {
    delete[] m_log_nb;
    delete[] m_log_t;
    delete[] m_yq;
    delete[] m_table;
  }
}

Real EOSCompOSE::TemperatureFromE(Real n, Real e, Real *Y) {
  assert (m_initialized);
  return temperature_from_var(ECLOGE, log(e), n, Y[0]);
}

Real EOSCompOSE::TemperatureFromP(Real n, Real p, Real *Y) {
  assert (m_initialized);
  return temperature_from_var(ECLOGP, log(p), n, Y[0]);
}

Real EOSCompOSE::Energy(Real n, Real T, Real *Y) {
  assert (m_initialized);
  return exp(eval_at_nty(ECLOGE, n, T, Y[0]));
}

Real EOSCompOSE::Pressure(Real n, Real T, Real *Y) {
  assert (m_initialized);
  return exp(eval_at_nty(ECLOGP, n, T, Y[0]));
}

Real EOSCompOSE::Entropy(Real n, Real T, Real *Y) {
  assert (m_initialized);
  return eval_at_nty(ECENT, n, T, Y[0]);
}

Real EOSCompOSE::Enthalpy(Real n, Real T, Real *Y) {
  assert (m_initialized);
  Real const P = Pressure(n, T, Y);
  Real const e = Energy(n, T, Y);
  return (P + e)/n;
}

Real EOSCompOSE::SoundSpeed(Real n, Real T, Real *Y) {
  assert (m_initialized);
  return eval_at_nty(ECCS, n, T, Y[0]);
}

Real EOSCompOSE::SpecificInternalEnergy(Real n, Real T, Real *Y) {
  return Energy(n, T, Y)/(mb*n) - 1;
}

Real EOSCompOSE::MinimumEnthalpy() {
  return m_min_h;
}

Real EOSCompOSE::MinimumPressure(Real n, Real *Y) {
  return Pressure(n, min_T, Y);
}

Real EOSCompOSE::MaximumPressure(Real n, Real *Y) {
  return Pressure(n, max_T, Y);
}

Real EOSCompOSE::MinimumEnergy(Real n, Real *Y) {
  return Energy(n, min_T, Y);
}

Real EOSCompOSE::MaximumEnergy(Real n, Real *Y) {
  return Energy(n, max_T, Y);
}

void EOSCompOSE::ReadTableFromFile(std::string fname) {
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
  min_T = scratch[0];
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
      log(mb*(scratch[index(0, in, iy, it)] + 1)) + m_log_nb[in];
  }}}

  ierr = H5LTread_dataset_double(file_id, "cs2", scratch);
    MYH5CHECK(ierr);
  for (int in = 0; in < m_nn; ++in) {
  for (int iy = 0; iy < m_ny; ++iy) {
  for (int it = 0; it < m_nt; ++it) {
    m_table[index(ECCS, in, iy, it)] = sqrt(scratch[index(0, in, iy, it)]);
  }}}

  // Cleanup
  // -------------------------------------------------------------------------
  delete[] scratch;
  H5Fclose(file_id);

  m_initialized = true;

  // Compute minimum enthalpy
  // -------------------------------------------------------------------------
  for (int in = 0; in < m_nn; ++in) {
    Real const nb = exp(m_log_nb[in]);
    for (int it = 0; it < m_nt; ++it) {
      Real const t = exp(m_log_t[it]);
      for (int iy = 0; iy < m_ny; ++iy) {
        m_min_h = min(m_min_h, Enthalpy(nb, t, &m_yq[iy]));
      }
    }
  }
}

Real EOSCompOSE::temperature_from_var(int iv, Real var, Real n, Real Yq) const {
  int in, iy;
  Real wn0, wn1, wy0, wy1;
  weight_idx_ln(&wn0, &wn1, &in, log(n));
  weight_idx_yq(&wy0, &wy1, &iy, Yq);

  auto f = [=](int it){
    Real var_pt =
      wn0 * (wy0 * m_table[index(iv, in+0, iy+0, it)]  +
             wy1 * m_table[index(iv, in+0, iy+1, it)]) +
      wn1 * (wy0 * m_table[index(iv, in+1, iy+0, it)]  +
             wy1 * m_table[index(iv, in+1, iy+1, it)]);

    return var - var_pt;
  };

  int ilo = 0;
  int ihi = m_nt-1;
  Real flo = f(ilo);
  Real fhi = f(ihi);
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

Real EOSCompOSE::eval_at_nty(int vi, Real n, Real T, Real Yq) const {
  return eval_at_lnty(vi, log(n), log(T), Yq);
}

void EOSCompOSE::weight_idx_ln(Real *w0, Real *w1, int *in, Real log_n) const {
  *in = (log_n - m_log_nb[0])*m_id_log_nb;
  *w1 = (log_n - m_log_nb[*in])*m_id_log_nb;
  *w0 = 1.0 - (*w1);
}

void EOSCompOSE::weight_idx_yq(Real *w0, Real *w1, int *iy, Real yq) const {
  *iy = (yq - m_yq[0])*m_id_yq;
  *w1 = (yq - m_yq[*iy])*m_id_yq;
  *w0 = 1.0 - (*w1);
}

void EOSCompOSE::weight_idx_lt(Real *w0, Real *w1, int *it, Real log_t) const {
  *it = (log_t - m_log_t[0])*m_id_log_t;
  *w1 = (log_t - m_log_t[*it])*m_id_log_t;
  *w0 = 1.0 - (*w1);
}

Real EOSCompOSE::eval_at_lnty(int iv, Real log_n, Real log_t, Real yq) const {
  int in, iy, it;
  Real wn0, wn1, wy0, wy1, wt0, wt1;

  weight_idx_ln(&wn0, &wn1, &in, log_n);
  weight_idx_yq(&wy0, &wy1, &iy, yq);
  weight_idx_lt(&wt0, &wt1, &it, log_t);

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

#else //HDF5OUTPUT
// Consider adding no-ops here?
#endif
