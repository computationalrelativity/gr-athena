//! \file hybrid_table.cpp
//  \brief Implementation of HybridTable

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <iostream>

//#ifdef HDF5OUTPUT
#include <hdf5.h>
#include <hdf5_hl.h>

#include "hybrid_table.hpp"
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

HybridTable::HybridTable():
  m_id_log_nb(numeric_limits<Real>::quiet_NaN()),
  m_nn(0),
  m_min_h(numeric_limits<Real>::max()) {

  min_T = 0.0;
  max_T = std::numeric_limits<Real>::max();
  n_species = 0;
  for (int i = 0; i < MAX_SPECIES; i++) {
    min_Y[i] = 0.0;
    max_Y[i] = 1.0;
  }

  eos_units = &Nuclear;
}

HybridTable::~HybridTable() {
    // These are static variables, so no need to delete
  /*
  if (m_initialized) {
    delete[] m_log_nb;
    delete[] m_table;
  }
  */
}

//Definitions for static members
Real * HybridTable::m_log_nb = nullptr;
Real * HybridTable::m_table = nullptr;
bool HybridTable::m_initialized = false;

Real HybridTable::sm_id_log_nb = numeric_limits<Real>::quiet_NaN();

int HybridTable::sm_nn = 0;

Real HybridTable::sm_min_h = numeric_limits<Real>::max();

Real HybridTable::s_mb = numeric_limits<Real>::quiet_NaN();
Real HybridTable::s_max_n = numeric_limits<Real>::quiet_NaN();
Real HybridTable::s_min_n = numeric_limits<Real>::quiet_NaN();

Real HybridTable::TemperatureFromE(Real n, Real e, Real *Y) {
  assert (m_initialized);
  Real e_cold = ColdEnergy(n);
  Real T = gamma_th_m1*(e-e_cold)/n;
  return std::fmax(T,0.0);
}

Real HybridTable::TemperatureFromP(Real n, Real p, Real *Y) {
  assert (m_initialized);
  Real p_cold = ColdPressure(n);
  Real T = (p-p_cold)/n;
  return std::fmax(T,0.0);
}

Real HybridTable::Energy(Real n, Real T, Real *Y) {
  assert (m_initialized);
  Real e_cold = ColdEnergy(n);
  Real e_th   = n*T/gamma_th_m1;
  return e_cold + e_th;
}

Real HybridTable::Pressure(Real n, Real T, Real *Y) {
  assert (m_initialized);
  Real p_cold = ColdPressure(n);
  Real p_th   = n*T;
  return p_cold + p_th;
}

Real HybridTable::Entropy(Real n, Real T, Real *Y) {
  throw std::logic_error("HybridTable::Entropy not currently implemented.");
}

Real HybridTable::Enthalpy(Real n, Real T, Real *Y) {
  assert (m_initialized);
  Real const P = Pressure(n, T, Y);
  Real const e = Energy(n, T, Y);
  return (P + e)/n;
}

Real HybridTable::SoundSpeed(Real n, Real T, Real *Y) {
  assert (m_initialized);
  Real H_cold = ColdEnthalpy(n);
  Real H_th   = (gamma_th*T)/(gamma_th_m1);

  Real Hcs2_cold = std::pow(ColdSoundSpeed(n),2.0)*H_cold;
  Real Hcs2_th   = gamma_th*T;

  return std::sqrt((Hcs2_cold + Hcs2_th)/(H_cold + H_th));
}

Real HybridTable::SpecificInternalEnergy(Real n, Real T, Real *Y) {
  assert (m_initialized);
  return Energy(n, T, Y)/(mb*n) - 1;
}

Real HybridTable::BaryonChemicalPotential(Real n, Real T, Real *Y) {
  throw std::logic_error("HybridTable::BaryonChemicalPotential not currently implemented.");
}

Real HybridTable::ChargeChemicalPotential(Real n, Real T, Real *Y) {
  throw std::logic_error("HybridTable::ChargeChemicalPotential not currently implemented.");
}

Real HybridTable::ElectronLeptonChemicalPotential(Real n, Real T, Real *Y) {
  throw std::logic_error("HybridTable::ElectronLeptonChemicalPotential not currently implemented.");
}

Real HybridTable::Abar(Real n, Real T, Real *Y) {
  throw std::logic_error("HybridTable::Abar not currently implemented.");
}

Real HybridTable::ColdEnergy(Real n) {
  assert (m_initialized);
  return exp(eval_at_n(ECLOGE, n));
}

Real HybridTable::ColdPressure(Real n) {
  assert (m_initialized);
  return exp(eval_at_n(ECLOGP, n));

}

Real HybridTable::ColdEnthalpy(Real n) {
  assert (m_initialized);
  Real const p_cold = ColdPressure(n);
  Real const e_cold = ColdEnergy(n);
  return (p_cold + e_cold)/n;
}

Real HybridTable::ColdSoundSpeed(Real n) {
  assert (m_initialized);
  return eval_at_n(ECCS, n);
}

Real HybridTable::MinimumEnthalpy() {
  return m_min_h;
}

Real HybridTable::MinimumPressure(Real n, Real *Y) {
  assert (m_initialized);
  return ColdPressure(n);
}

Real HybridTable::MaximumPressure(Real n, Real *Y) {
  return std::numeric_limits<Real>::max();
}

Real HybridTable::MinimumEnergy(Real n, Real *Y) {
  assert (m_initialized);
  return ColdEnergy(n);
}

Real HybridTable::MaximumEnergy(Real n, Real *Y) {
  return std::numeric_limits<Real>::max();
}

void HybridTable::ReadTableFromFile(std::string fname) {
  #pragma omp critical (HybridTable_ReadTable)
  {
    if (m_initialized==false) {
      herr_t ierr;
      hid_t file_id;
      hsize_t snb;

      // Open input file
      // -------------------------------------------------------------------------
      file_id = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        MYH5CHECK(file_id);

      // Get dataset sizes
      // -------------------------------------------------------------------------
      ierr = H5LTget_dataset_info(file_id, "nb", &snb, NULL, NULL);
        MYH5CHECK(ierr);
      m_nn = snb;

      // Allocate memory
      // -------------------------------------------------------------------------
      m_log_nb = new Real[m_nn];
      m_table = new Real[ECNVARS*m_nn];
      double * scratch = new double[m_nn];

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

      // the neutron mass is used as the baryon mass in CompOSE
      ierr = H5LTread_dataset_double(file_id, "mn", scratch);
      MYH5CHECK(ierr);
      mb = scratch[0];

      // Read other thermodynamics quantities
      // -------------------------------------------------------------------------
      ierr = H5LTread_dataset_double(file_id, "Q1", scratch);
        MYH5CHECK(ierr);
      for (int inb = 0; inb < m_nn; ++inb) {
        m_table[index(ECLOGP, inb)] =
          log(scratch[index(0, inb)]) + m_log_nb[inb];
      }

      ierr = H5LTread_dataset_double(file_id, "Q2", scratch);
        MYH5CHECK(ierr);
      copy(&scratch[0], &scratch[m_nn], &m_table[index(ECENT, 0)]);

      ierr = H5LTread_dataset_double(file_id, "Q3", scratch);
        MYH5CHECK(ierr);
      for (int in = 0; in < m_nn; ++in) {
        m_table[index(ECMUB, in)] =
          mb*(scratch[index(0, in)] + 1);
      }

      ierr = H5LTread_dataset_double(file_id, "Q4", scratch);
        MYH5CHECK(ierr);
      for (int in = 0; in < m_nn; ++in) {
        m_table[index(ECMUQ, in)] = mb*scratch[index(0, in)];
      }

      ierr = H5LTread_dataset_double(file_id, "Q5", scratch);
        MYH5CHECK(ierr);
      for (int in = 0; in < m_nn; ++in) {
        m_table[index(ECMUL, in)] = mb*scratch[index(0, in)];
      }

      ierr = H5LTread_dataset_double(file_id, "Q7", scratch);
        MYH5CHECK(ierr);
      for (int in = 0; in < m_nn; ++in) {
        m_table[index(ECLOGE, in)] =
          log(mb*(scratch[index(0, in)] + 1)) + m_log_nb[in];
      }

      ierr = H5LTread_dataset_double(file_id, "cs2", scratch);
        MYH5CHECK(ierr);
      for (int in = 0; in < m_nn; ++in) {
        m_table[index(ECCS, in)] = sqrt(scratch[index(0, in)]);
      }
      
      
      // Mark table as read
      m_initialized = true;

      // Cleanup
      // -------------------------------------------------------------------------
      delete[] scratch;
      H5Fclose(file_id);

      // Compute minimum enthalpy
      // -------------------------------------------------------------------------
      for (int in = 0; in < m_nn; ++in) {
        Real const nb = exp(m_log_nb[in]);
        m_min_h = min(m_min_h, ColdEnthalpy(nb));
      }

      // Now that we have read everything locally, we must populate
      // the aux static variables to share this data with other threads
      sm_id_log_nb = m_id_log_nb;
      sm_nn = m_nn;
      sm_min_h = m_min_h;

      s_mb = mb;
      s_max_n = max_n;
      s_min_n = min_n;

    } // if (sm_initialized==false)
  } // omp critical (EOSCompOSE_ReadTable)

  // Disseminate applicable static variables to local memory
  m_id_log_nb = sm_id_log_nb;
  m_nn = sm_nn;
  m_min_h = sm_min_h;

  mb       = s_mb;
  max_n    = s_max_n;
  min_n    = s_min_n;
  
  }


Real HybridTable::eval_at_n(int vi, Real n) const {
  return eval_at_ln(vi, log(n));
}

void HybridTable::weight_idx_ln(Real *w0, Real *w1, int *in, Real log_n) const {
  *in = (log_n - m_log_nb[0])*m_id_log_nb;
  *w1 = (log_n - m_log_nb[*in])*m_id_log_nb;
  *w0 = 1.0 - (*w1);
}

Real HybridTable::eval_at_ln(int iv, Real log_n) const {
  int in;
  Real wn0, wn1;

  weight_idx_ln(&wn0, &wn1, &in, log_n);

  return
    wn0 * m_table[index(iv, in+0)] +
    wn1 * m_table[index(iv, in+1)];
}
