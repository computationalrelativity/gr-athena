//! \file eos_compose.cpp
//  \brief Implementation of EOSCompose

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

// #ifdef HDF5OUTPUT
#include <hdf5.h>
#include <hdf5_hl.h>

#include "eos_compose.hpp"
#include "numtools_root.hpp"
#include "unit_system.hpp"

using namespace Primitive;
using namespace std;

#define MYH5CHECK(ierr)                                               \
  if (ierr < 0)                                                       \
  {                                                                   \
    stringstream ss;                                                  \
    ss << __FILE__ << ":" << __LINE__ << " error reading EOS table!"; \
    throw runtime_error(ss.str().c_str());                            \
  }

EOSCompOSE::EOSCompOSE()
    : m_id_log_nb(numeric_limits<Real>::quiet_NaN()),
      m_id_log_t(numeric_limits<Real>::quiet_NaN()),
      m_id_yq(numeric_limits<Real>::quiet_NaN()),
      m_nn(0),
      m_nt(0),
      m_ny(0),
      m_min_h(numeric_limits<Real>::max())
{
  n_species = 1;
  eos_units = &Nuclear;
}
// These are static now, so are defined separately below
/*
m_log_nb(nullptr),
m_log_t(nullptr),
m_yq(nullptr),
m_table(nullptr),
m_initialized(false)
*/
EOSCompOSE::~EOSCompOSE()
{
  // These are static variables now, so no need to delete
  /*
  if (m_initialized) {
    delete[] m_log_nb;
    delete[] m_log_t;
    delete[] m_yq;
    delete[] m_table;
  }
  */
}

// Definitions for static members
Real* EOSCompOSE::m_log_nb     = nullptr;
Real* EOSCompOSE::m_log_t      = nullptr;
Real* EOSCompOSE::m_yq         = nullptr;
Real* EOSCompOSE::m_table      = nullptr;
bool EOSCompOSE::m_initialized = false;

Real EOSCompOSE::sm_id_log_nb = numeric_limits<Real>::quiet_NaN();
Real EOSCompOSE::sm_id_log_t  = numeric_limits<Real>::quiet_NaN();
Real EOSCompOSE::sm_id_yq     = numeric_limits<Real>::quiet_NaN();

int EOSCompOSE::sm_nn = 0;
int EOSCompOSE::sm_nt = 0;
int EOSCompOSE::sm_ny = 0;

Real EOSCompOSE::sm_min_h = numeric_limits<Real>::max();

Real EOSCompOSE::s_mb                 = numeric_limits<Real>::quiet_NaN();
Real EOSCompOSE::s_max_n              = numeric_limits<Real>::quiet_NaN();
Real EOSCompOSE::s_min_n              = numeric_limits<Real>::quiet_NaN();
Real EOSCompOSE::s_max_T              = numeric_limits<Real>::quiet_NaN();
Real EOSCompOSE::s_min_T              = numeric_limits<Real>::quiet_NaN();
Real EOSCompOSE::s_max_Y[MAX_SPECIES] = { 0 };
Real EOSCompOSE::s_min_Y[MAX_SPECIES] = { 0 };

Real EOSCompOSE::s_mn = numeric_limits<Real>::quiet_NaN();
Real EOSCompOSE::s_mp = numeric_limits<Real>::quiet_NaN();

Real EOSCompOSE::TemperatureFromE(Real n, Real e, Real* Y)
{
  assert(m_initialized);
  // Hoist density and composition weights: computed once, reused for
  // bounds checks and the inner root-find.
  int in, iy;
  Real wn0, wn1, wy0, wy1;
  weight_idx_ln(&wn0, &wn1, &in, log(n));
  weight_idx_yq(&wy0, &wy1, &iy, Y[0]);

  // Evaluate log(e) at the table boundaries using 4 lookups each
  // (the temperature weight is trivially 1 at grid endpoints).
  Real loge_min = eval_at_it(ECLOGE, wn0, wn1, in, wy0, wy1, iy, 0);
  Real loge_max = eval_at_it(ECLOGE, wn0, wn1, in, wy0, wy1, iy, m_nt - 1);
  Real e_min    = exp(loge_min);
  Real e_max    = exp(loge_max);

  if (e <= e_min)
    return min_T;
  if (e >= e_max)
    return max_T;
  return temperature_from_var_precomp(
    loge_min, loge_max, ECLOGE, log(e), wn0, wn1, in, wy0, wy1, iy);
}

Real EOSCompOSE::TemperatureFromP(Real n, Real p, Real* Y)
{
  assert(m_initialized);
  int in, iy;
  Real wn0, wn1, wy0, wy1;
  weight_idx_ln(&wn0, &wn1, &in, log(n));
  weight_idx_yq(&wy0, &wy1, &iy, Y[0]);

  Real logp_min = eval_at_it(ECLOGP, wn0, wn1, in, wy0, wy1, iy, 0);
  Real logp_max = eval_at_it(ECLOGP, wn0, wn1, in, wy0, wy1, iy, m_nt - 1);
  Real p_min    = exp(logp_min);
  Real p_max    = exp(logp_max);

  if (p <= p_min)
    return min_T;
  if (p >= p_max)
    return max_T;
  return temperature_from_var_precomp(
    logp_min, logp_max, ECLOGP, log(p), wn0, wn1, in, wy0, wy1, iy);
}

Real EOSCompOSE::TemperatureFromEntropy(Real n, Real s, Real* Y)
{
  assert(m_initialized);
  int in, iy;
  Real wn0, wn1, wy0, wy1;
  weight_idx_ln(&wn0, &wn1, &in, log(n));
  weight_idx_yq(&wy0, &wy1, &iy, Y[0]);

  // Entropy is stored directly (not in log space).
  Real s_min = eval_at_it(ECENT, wn0, wn1, in, wy0, wy1, iy, 0);
  Real s_max = eval_at_it(ECENT, wn0, wn1, in, wy0, wy1, iy, m_nt - 1);

  if (s <= s_min)
    return min_T;
  if (s >= s_max)
    return max_T;
  return temperature_from_var_precomp(s_min, s_max, ECENT, s, wn0, wn1, in, wy0, wy1, iy);
}

Real EOSCompOSE::Energy(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  return exp(eval_at_nty(ECLOGE, n, T, Y[0]));
}

Real EOSCompOSE::Pressure(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  return exp(eval_at_nty(ECLOGP, n, T, Y[0]));
}

Real EOSCompOSE::Entropy(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  return eval_at_nty(ECENT, n, T, Y[0]);
}

Real EOSCompOSE::Enthalpy(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  Real const P = Pressure(n, T, Y);
  Real const e = Energy(n, T, Y);
  return (P + e) / n;
}

void EOSCompOSE::PressureAndEnthalpy(Real n, Real T, Real* Y, Real* P, Real* h)
{
  assert(m_initialized);
  Real log_n = log(n);
  Real log_t = log(T);
  Real Yq    = Y[0];

  int in, iy, it;
  Real wn0, wn1, wy0, wy1, wt0, wt1;
  weight_idx_ln(&wn0, &wn1, &in, log_n);
  weight_idx_yq(&wy0, &wy1, &iy, Yq);
  weight_idx_lt(&wt0, &wt1, &it, log_t);

  // Interpolate log(P) and log(e) with shared weights.
  ptrdiff_t bp00 = index(ECLOGP, in, iy, it);
  ptrdiff_t bp01 = index(ECLOGP, in, iy + 1, it);
  ptrdiff_t bp10 = index(ECLOGP, in + 1, iy, it);
  ptrdiff_t bp11 = index(ECLOGP, in + 1, iy + 1, it);

  Real logP = wn0 * (wy0 * (wt0 * m_table[bp00] + wt1 * m_table[bp00 + 1]) +
                     wy1 * (wt0 * m_table[bp01] + wt1 * m_table[bp01 + 1])) +
              wn1 * (wy0 * (wt0 * m_table[bp10] + wt1 * m_table[bp10 + 1]) +
                     wy1 * (wt0 * m_table[bp11] + wt1 * m_table[bp11 + 1]));

  // ECLOGE base offsets: same (in, iy, it) cell, different variable slice.
  ptrdiff_t be00 = index(ECLOGE, in, iy, it);
  ptrdiff_t be01 = index(ECLOGE, in, iy + 1, it);
  ptrdiff_t be10 = index(ECLOGE, in + 1, iy, it);
  ptrdiff_t be11 = index(ECLOGE, in + 1, iy + 1, it);

  Real logE = wn0 * (wy0 * (wt0 * m_table[be00] + wt1 * m_table[be00 + 1]) +
                     wy1 * (wt0 * m_table[be01] + wt1 * m_table[be01 + 1])) +
              wn1 * (wy0 * (wt0 * m_table[be10] + wt1 * m_table[be10 + 1]) +
                     wy1 * (wt0 * m_table[be11] + wt1 * m_table[be11 + 1]));

  Real Pval = exp(logP);
  Real eval = exp(logE);
  *P        = Pval;
  *h        = (Pval + eval) / n;
}

void EOSCompOSE::TemperaturePressureAndEnthalpyFromE(Real n,
                                                     Real e,
                                                     Real* Y,
                                                     Real* T,
                                                     Real* P,
                                                     Real* h,
                                                     int* guess_it)
{
  assert(m_initialized);
  int in, iy;
  Real wn0, wn1, wy0, wy1;
  Real log_n = log(n);
  weight_idx_ln(&wn0, &wn1, &in, log_n);
  weight_idx_yq(&wy0, &wy1, &iy, Y[0]);

  ptrdiff_t const be00 = index(ECLOGE, in, iy, 0);
  ptrdiff_t const be01 = index(ECLOGE, in, iy + 1, 0);
  ptrdiff_t const be10 = index(ECLOGE, in + 1, iy, 0);
  ptrdiff_t const be11 = index(ECLOGE, in + 1, iy + 1, 0);

  Real var = log(e);

  auto f = [=](int it) -> Real
  {
    Real var_pt = wn0 * (wy0 * m_table[be00 + it] + wy1 * m_table[be01 + it]) +
                  wn1 * (wy0 * m_table[be10 + it] + wy1 * m_table[be11 + it]);
    return var - var_pt;
  };

  int ilo        = 0;
  int ihi        = m_nt - 1;
  Real flo       = 0.0;
  Real fhi       = 0.0;
  bool bracketed = false;

  // Hunt locally first
  if (guess_it && *guess_it >= 0 && *guess_it < m_nt - 1)
  {
    int it  = *guess_it;
    Real fl = f(it);
    Real fh = f(it + 1);
    if (fl * fh <= 0)
    {
      ilo       = it;
      ihi       = it + 1;
      flo       = fl;
      fhi       = fh;
      bracketed = true;
    }
    else if (fl < 0 && it > 0) // Try shifting left
    {
      Real fl_minus = f(it - 1);
      if (fl_minus * fl <= 0)
      {
        ilo       = it - 1;
        ihi       = it;
        flo       = fl_minus;
        fhi       = fl;
        bracketed = true;
      }
    }
    else if (fh > 0 && it + 2 < m_nt) // Try shifting right
    {
      Real fh_plus = f(it + 2);
      if (fh * fh_plus <= 0)
      {
        ilo       = it + 1;
        ihi       = it + 2;
        flo       = fh;
        fhi       = fh_plus;
        bracketed = true;
      }
    }
  }

  if (!bracketed)
  {
    // Evaluate log(e) at the table boundaries using 4 lookups each
    Real loge_min = eval_at_it(ECLOGE, wn0, wn1, in, wy0, wy1, iy, 0);
    Real loge_max = eval_at_it(ECLOGE, wn0, wn1, in, wy0, wy1, iy, m_nt - 1);
    Real e_min    = exp(loge_min);
    Real e_max    = exp(loge_max);

    if (e <= e_min)
    {
      *T = min_T;
      PressureAndEnthalpy(n, *T, Y, P, h);
      return;
    }
    if (e >= e_max)
    {
      *T = max_T;
      PressureAndEnthalpy(n, *T, Y, P, h);
      return;
    }

    flo = log(e) - loge_min;
    fhi = log(e) - loge_max;

    if (flo * fhi > 0)
    {
      // Bracket already at adjacent points is the best we can do
    }
    else
    {
      int it_guess =
        static_cast<int>(static_cast<Real>(m_nt - 1) * flo / (flo - fhi));
      it_guess = std::max(0, std::min(m_nt - 2, it_guess));

      Real fg = f(it_guess);
      if (fg * flo <= 0)
      {
        ihi = it_guess;
        fhi = fg;
      }
      else
      {
        ilo = it_guess;
        flo = fg;
      }

      while (ihi - ilo > 1)
      {
        int ip  = ilo + (ihi - ilo) / 2;
        Real fp = f(ip);
        if (fp * flo <= 0)
        {
          ihi = ip;
          fhi = fp;
        }
        else
        {
          ilo = ip;
          flo = fp;
        }
      }
    }
  }

  assert(ihi - ilo == 1 || flo * fhi <= 0);
  if (guess_it)
  {
    *guess_it = ilo;
  }
  Real ltlo = m_log_t[ilo];
  Real lthi = m_log_t[ihi];

  Real lt;
  Real wt0, wt1;

  if (flo == 0)
  {
    lt  = ltlo;
    wt0 = 1.0;
    wt1 = 0.0;
  }
  else if (fhi == 0)
  {
    lt  = lthi;
    wt0 = 0.0;
    wt1 = 1.0;
  }
  else
  {
    lt  = ltlo - flo * (lthi - ltlo) / (fhi - flo);
    wt1 = (lt - ltlo) * m_id_log_t;
    wt0 = 1.0 - wt1;
  }

  *T = exp(lt);

  int it = ilo;

  ptrdiff_t bp00 = index(ECLOGP, in, iy, it);
  ptrdiff_t bp01 = index(ECLOGP, in, iy + 1, it);
  ptrdiff_t bp10 = index(ECLOGP, in + 1, iy, it);
  ptrdiff_t bp11 = index(ECLOGP, in + 1, iy + 1, it);

  Real logP = wn0 * (wy0 * (wt0 * m_table[bp00] + wt1 * m_table[bp00 + 1]) +
                     wy1 * (wt0 * m_table[bp01] + wt1 * m_table[bp01 + 1])) +
              wn1 * (wy0 * (wt0 * m_table[bp10] + wt1 * m_table[bp10 + 1]) +
                     wy1 * (wt0 * m_table[bp11] + wt1 * m_table[bp11 + 1]));

  *P = exp(logP);
  *h = (*P + e) / n;
}

void EOSCompOSE::PressureAndEnthalpyFromE(Real n,
                                                     Real e,
                                                     Real* Y,
                                                     Real* P,
                                                     Real* h,
                                                     int* guess_it)
{
  assert(m_initialized);
  int in, iy;
  Real wn0, wn1, wy0, wy1;
  Real log_n = log(n);
  weight_idx_ln(&wn0, &wn1, &in, log_n);
  weight_idx_yq(&wy0, &wy1, &iy, Y[0]);

  ptrdiff_t const be00 = index(ECLOGE, in, iy, 0);
  ptrdiff_t const be01 = index(ECLOGE, in, iy + 1, 0);
  ptrdiff_t const be10 = index(ECLOGE, in + 1, iy, 0);
  ptrdiff_t const be11 = index(ECLOGE, in + 1, iy + 1, 0);

  Real var = log(e);

  auto f = [=](int it) -> Real
  {
    Real var_pt = wn0 * (wy0 * m_table[be00 + it] + wy1 * m_table[be01 + it]) +
                  wn1 * (wy0 * m_table[be10 + it] + wy1 * m_table[be11 + it]);
    return var - var_pt;
  };

  int ilo        = 0;
  int ihi        = m_nt - 1;
  Real flo       = 0.0;
  Real fhi       = 0.0;
  bool bracketed = false;

  // Hunt locally first
  if (guess_it && *guess_it >= 0 && *guess_it < m_nt - 1)
  {
    int it  = *guess_it;
    Real fl = f(it);
    Real fh = f(it + 1);
    if (fl * fh <= 0)
    {
      ilo       = it;
      ihi       = it + 1;
      flo       = fl;
      fhi       = fh;
      bracketed = true;
    }
    else if (fl < 0 && it > 0) // Try shifting left
    {
      Real fl_minus = f(it - 1);
      if (fl_minus * fl <= 0)
      {
        ilo       = it - 1;
        ihi       = it;
        flo       = fl_minus;
        fhi       = fl;
        bracketed = true;
      }
    }
    else if (fh > 0 && it + 2 < m_nt) // Try shifting right
    {
      Real fh_plus = f(it + 2);
      if (fh * fh_plus <= 0)
      {
        ilo       = it + 1;
        ihi       = it + 2;
        flo       = fh;
        fhi       = fh_plus;
        bracketed = true;
      }
    }
  }

  if (!bracketed)
  {
    // Evaluate log(e) at the table boundaries using 4 lookups each
    Real loge_min = eval_at_it(ECLOGE, wn0, wn1, in, wy0, wy1, iy, 0);
    Real loge_max = eval_at_it(ECLOGE, wn0, wn1, in, wy0, wy1, iy, m_nt - 1);
    Real e_min    = exp(loge_min);
    Real e_max    = exp(loge_max);

    if (e <= e_min)
    {
      PressureAndEnthalpy(n, min_T, Y, P, h);
      return;
    }
    if (e >= e_max)
    {
      PressureAndEnthalpy(n, max_T, Y, P, h);
      return;
    }

    flo = log(e) - loge_min;
    fhi = log(e) - loge_max;

    if (flo * fhi > 0)
    {
      // Bracket already at adjacent points is the best we can do
    }
    else
    {
      int it_guess =
        static_cast<int>(static_cast<Real>(m_nt - 1) * flo / (flo - fhi));
      it_guess = std::max(0, std::min(m_nt - 2, it_guess));

      Real fg = f(it_guess);
      if (fg * flo <= 0)
      {
        ihi = it_guess;
        fhi = fg;
      }
      else
      {
        ilo = it_guess;
        flo = fg;
      }

      while (ihi - ilo > 1)
      {
        int ip  = ilo + (ihi - ilo) / 2;
        Real fp = f(ip);
        if (fp * flo <= 0)
        {
          ihi = ip;
          fhi = fp;
        }
        else
        {
          ilo = ip;
          flo = fp;
        }
      }
    }
  }

  assert(ihi - ilo == 1 || flo * fhi <= 0);
  if (guess_it)
  {
    *guess_it = ilo;
  }
  Real ltlo = m_log_t[ilo];
  Real lthi = m_log_t[ihi];

  Real lt;
  Real wt0, wt1;

  if (flo == 0)
  {
    lt  = ltlo;
    wt0 = 1.0;
    wt1 = 0.0;
  }
  else if (fhi == 0)
  {
    lt  = lthi;
    wt0 = 0.0;
    wt1 = 1.0;
  }
  else
  {
    lt  = ltlo - flo * (lthi - ltlo) / (fhi - flo);
    wt1 = (lt - ltlo) * m_id_log_t;
    wt0 = 1.0 - wt1;
  }


  int it = ilo;

  ptrdiff_t bp00 = index(ECLOGP, in, iy, it);
  ptrdiff_t bp01 = index(ECLOGP, in, iy + 1, it);
  ptrdiff_t bp10 = index(ECLOGP, in + 1, iy, it);
  ptrdiff_t bp11 = index(ECLOGP, in + 1, iy + 1, it);

  Real logP = wn0 * (wy0 * (wt0 * m_table[bp00] + wt1 * m_table[bp00 + 1]) +
                     wy1 * (wt0 * m_table[bp01] + wt1 * m_table[bp01 + 1])) +
              wn1 * (wy0 * (wt0 * m_table[bp10] + wt1 * m_table[bp10 + 1]) +
                     wy1 * (wt0 * m_table[bp11] + wt1 * m_table[bp11 + 1]));

  *P = exp(logP);
  *h = (*P + e) / n;
}

Real EOSCompOSE::SoundSpeed(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  return eval_at_nty(ECCS, n, T, Y[0]);
}

Real EOSCompOSE::FrYn(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  return eval_at_nty(ECYN, n, T, Y[0]);
}

Real EOSCompOSE::FrYp(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  return eval_at_nty(ECYP, n, T, Y[0]);
}

Real EOSCompOSE::FrYh(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  return eval_at_nty(ECYH, n, T, Y[0]);
}

Real EOSCompOSE::AN(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  return eval_at_nty(ECAN, n, T, Y[0]);
}

Real EOSCompOSE::ZN(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  return eval_at_nty(ECZN, n, T, Y[0]);
}

Real EOSCompOSE::SpecificInternalEnergy(Real n, Real T, Real* Y)
{
  return Energy(n, T, Y) / (mb * n) - 1;
}

Real EOSCompOSE::BaryonChemicalPotential(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  return eval_at_nty(ECMUB, n, T, Y[0]);
}

Real EOSCompOSE::ChargeChemicalPotential(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  return eval_at_nty(ECMUQ, n, T, Y[0]);
}

Real EOSCompOSE::ElectronLeptonChemicalPotential(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  return eval_at_nty(ECMUL, n, T, Y[0]);
}

Real EOSCompOSE::InteractionPotentialDifference(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  return eval_at_nty(ECDU, n, T, Y[0]);
}

Real EOSCompOSE::MinimumEnthalpy()
{
  return m_min_h;
}

Real EOSCompOSE::MinimumPressure(Real n, Real* Y)
{
  return Pressure(n, min_T, Y);
}

Real EOSCompOSE::MaximumPressure(Real n, Real* Y)
{
  return Pressure(n, max_T, Y);
}

Real EOSCompOSE::MinimumEnergy(Real n, Real* Y)
{
  return Energy(n, min_T, Y);
}

Real EOSCompOSE::MaximumEnergy(Real n, Real* Y)
{
  return Energy(n, max_T, Y);
}

Real EOSCompOSE::MinimumEntropy(Real n, Real* Y)
{
  return Entropy(n, min_T, Y);
}

Real EOSCompOSE::MaximumEntropy(Real n, Real* Y)
{
  return Entropy(n, max_T, Y);
}

void EOSCompOSE::ReadTableFromFile(std::string fname)
{
#pragma omp critical(EOSCompose_ReadTable)
  {
    if (m_initialized == false)
    {
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
      m_log_nb        = new Real[m_nn];
      m_log_t         = new Real[m_nt];
      m_yq            = new Real[m_ny];
      m_table         = new Real[ECNVARS * m_nn * m_ny * m_nt];
      double* scratch = new double[m_nn * m_ny * m_nt];

      // Read nb, t, yq
      // -------------------------------------------------------------------------
      ierr = H5LTread_dataset_double(file_id, "nb", scratch);
      MYH5CHECK(ierr);
      min_n = scratch[0];
      max_n = scratch[m_nn - 1];
      for (int in = 0; in < m_nn; ++in)
      {
        m_log_nb[in] = log(scratch[in]);
      }
      m_id_log_nb = 1.0 / (m_log_nb[1] - m_log_nb[0]);

      ierr = H5LTread_dataset_double(file_id, "t", scratch);
      MYH5CHECK(ierr);
      min_T = scratch[0];
      max_T = scratch[m_nt - 1];
      for (int it = 0; it < m_nt; ++it)
      {
        m_log_t[it] = log(scratch[it]);
      }
      m_id_log_t = 1.0 / (m_log_t[1] - m_log_t[0]);

      ierr = H5LTread_dataset_double(file_id, "yq", scratch);
      MYH5CHECK(ierr);
      min_Y[0] = scratch[0];
      max_Y[0] = scratch[m_ny - 1];
      for (int iy = 0; iy < m_ny; ++iy)
      {
        m_yq[iy] = scratch[iy];
      }
      m_id_yq = 1.0 / (m_yq[1] - m_yq[0]);

      // the neutron mass is used as the baryon mass in CompOSE
      ierr = H5LTread_dataset_double(file_id, "mn", scratch);
      MYH5CHECK(ierr);
      mb = scratch[0];

      // Read other thermodynamics quantities
      // -------------------------------------------------------------------------
      ierr = H5LTread_dataset_double(file_id, "Q1", scratch);
      MYH5CHECK(ierr);
      for (int inb = 0; inb < m_nn; ++inb)
      {
        for (int iyq = 0; iyq < m_ny; ++iyq)
        {
          for (int it = 0; it < m_nt; ++it)
          {
            m_table[index(ECLOGP, inb, iyq, it)] =
              log(scratch[index(0, inb, iyq, it)]) + m_log_nb[inb];
          }
        }
      }

      ierr = H5LTread_dataset_double(file_id, "Q2", scratch);
      MYH5CHECK(ierr);
      copy(&scratch[0],
           &scratch[m_nn * m_ny * m_nt],
           &m_table[index(ECENT, 0, 0, 0)]);

      ierr = H5LTread_dataset_double(file_id, "Q3", scratch);
      MYH5CHECK(ierr);
      for (int in = 0; in < m_nn; ++in)
      {
        for (int iy = 0; iy < m_ny; ++iy)
        {
          for (int it = 0; it < m_nt; ++it)
          {
            m_table[index(ECMUB, in, iy, it)] =
              mb * (scratch[index(0, in, iy, it)] + 1);
          }
        }
      }

      ierr = H5LTread_dataset_double(file_id, "Q4", scratch);
      MYH5CHECK(ierr);
      for (int in = 0; in < m_nn; ++in)
      {
        for (int iy = 0; iy < m_ny; ++iy)
        {
          for (int it = 0; it < m_nt; ++it)
          {
            m_table[index(ECMUQ, in, iy, it)] =
              mb * scratch[index(0, in, iy, it)];
          }
        }
      }

      ierr = H5LTread_dataset_double(file_id, "Q5", scratch);
      MYH5CHECK(ierr);
      for (int in = 0; in < m_nn; ++in)
      {
        for (int iy = 0; iy < m_ny; ++iy)
        {
          for (int it = 0; it < m_nt; ++it)
          {
            m_table[index(ECMUL, in, iy, it)] =
              mb * scratch[index(0, in, iy, it)];
          }
        }
      }

      ierr = H5LTread_dataset_double(file_id, "Q7", scratch);
      MYH5CHECK(ierr);
      for (int in = 0; in < m_nn; ++in)
      {
        for (int iy = 0; iy < m_ny; ++iy)
        {
          for (int it = 0; it < m_nt; ++it)
          {
            m_table[index(ECLOGE, in, iy, it)] =
              log(mb * (scratch[index(0, in, iy, it)] + 1)) + m_log_nb[in];
          }
        }
      }

      ierr = H5LTread_dataset_double(file_id, "cs2", scratch);
      MYH5CHECK(ierr);
      for (int in = 0; in < m_nn; ++in)
      {
        for (int iy = 0; iy < m_ny; ++iy)
        {
          for (int it = 0; it < m_nt; ++it)
          {
            m_table[index(ECCS, in, iy, it)] =
              sqrt(scratch[index(0, in, iy, it)]);
          }
        }
      }

      ierr = H5LTread_dataset_double(file_id, "Y[n]", scratch);
      MYH5CHECK(ierr);
      for (int in = 0; in < m_nn; ++in)
      {
        for (int iy = 0; iy < m_ny; ++iy)
        {
          for (int it = 0; it < m_nt; ++it)
          {
            m_table[index(ECYN, in, iy, it)] = scratch[index(0, in, iy, it)];
          }
        }
      }

      ierr = H5LTread_dataset_double(file_id, "Y[p]", scratch);
      MYH5CHECK(ierr);
      for (int in = 0; in < m_nn; ++in)
      {
        for (int iy = 0; iy < m_ny; ++iy)
        {
          for (int it = 0; it < m_nt; ++it)
          {
            m_table[index(ECYP, in, iy, it)] = scratch[index(0, in, iy, it)];
          }
        }
      }

      ierr = H5LTread_dataset_double(file_id, "A[N]", scratch);
      MYH5CHECK(ierr);
      for (int in = 0; in < m_nn; ++in)
      {
        for (int iy = 0; iy < m_ny; ++iy)
        {
          for (int it = 0; it < m_nt; ++it)
          {
            m_table[index(ECAN, in, iy, it)] = scratch[index(0, in, iy, it)];
          }
        }
      }

      ierr = H5LTread_dataset_double(file_id, "Z[N]", scratch);
      MYH5CHECK(ierr);
      for (int in = 0; in < m_nn; ++in)
      {
        for (int iy = 0; iy < m_ny; ++iy)
        {
          for (int it = 0; it < m_nt; ++it)
          {
            m_table[index(ECZN, in, iy, it)] = scratch[index(0, in, iy, it)];
          }
        }
      }

      ierr = H5LTread_dataset_double(file_id, "dU", scratch);
      if (ierr < 0) {
        // Dataset "dU" not found: fill table with zeros
        fill(&m_table[index(ECDU, 0, 0, 0)],
            &m_table[index(ECDU, 0, 0, 0)] + m_nn*m_ny*m_nt,
            0.0);
      } else {
        // Dataset read successfully, copy into table
        copy(&scratch[0], &scratch[m_nn*m_ny*m_nt], &m_table[index(ECDU, 0, 0, 0)]);
      }

      // The following requires Abar?:
      ierr = H5LTread_dataset_double(file_id, "Y[N]", scratch);
      MYH5CHECK(ierr);
      for (int in = 0; in < m_nn; ++in)
      {
        for (int iy = 0; iy < m_ny; ++iy)
        {
          for (int it = 0; it < m_nt; ++it)
          {
            const Real YN = scratch[index(0, in, iy, it)];
            const Real AN = m_table[index(ECAN, in, iy, it)];

            m_table[index(ECYH, in, iy, it)] = AN * YN;
          }
        }
      }

      // couple of final constants
      Real mn;  // neutron mass (note also used as baryonic)
      ierr = H5LTread_dataset_double(file_id, "mn", scratch);
      MYH5CHECK(ierr);
      mn = scratch[0];

      Real mp;
      ierr = H5LTread_dataset_double(file_id, "mp", scratch);
      MYH5CHECK(ierr);
      mp = scratch[0];

      // Mark table as read
      m_initialized = true;

      // Cleanup
      // -------------------------------------------------------------------------
      delete[] scratch;
      H5Fclose(file_id);

      // Compute minimum enthalpy
      // -------------------------------------------------------------------------
      for (int in = 0; in < m_nn; ++in)
      {
        Real const nb = exp(m_log_nb[in]);
        for (int it = 0; it < m_nt; ++it)
        {
          Real const t = exp(m_log_t[it]);
          for (int iy = 0; iy < m_ny; ++iy)
          {
            m_min_h = min(m_min_h, Enthalpy(nb, t, &m_yq[iy]));
          }
        }
      }

      // Now that we have read everything locally, we must populate
      // the aux static variables to share this data with other threads
      sm_id_log_nb = m_id_log_nb;
      sm_id_log_t  = m_id_log_t;
      sm_id_yq     = m_id_yq;

      sm_nn = m_nn;
      sm_nt = m_nt;
      sm_ny = m_ny;

      sm_min_h = m_min_h;

      s_mb       = mb;
      s_max_n    = max_n;
      s_min_n    = min_n;
      s_max_T    = max_T;
      s_min_T    = min_T;
      s_max_Y[0] = max_Y[0];
      s_min_Y[0] = min_Y[0];

      s_mn = mn;
      s_mp = mp;
    }  // if (sm_initialized==false)
  }  // omp critical (EOSCompOSE_ReadTable)

  // Disseminate applicable static variables to local memory
  m_id_log_nb = sm_id_log_nb;
  m_id_log_t  = sm_id_log_t;
  m_id_yq     = sm_id_yq;

  m_nn = sm_nn;
  m_nt = sm_nt;
  m_ny = sm_ny;

  m_min_h = sm_min_h;

  mb       = s_mb;
  max_n    = s_max_n;
  min_n    = s_min_n;
  max_T    = s_max_T;
  min_T    = s_min_T;
  max_Y[0] = s_max_Y[0];
  min_Y[0] = s_min_Y[0];
}

Real EOSCompOSE::temperature_from_var(int iv, Real var, Real n, Real Yq) const
{
  int in, iy;
  Real wn0, wn1, wy0, wy1;
  weight_idx_ln(&wn0, &wn1, &in, log(n));
  weight_idx_yq(&wy0, &wy1, &iy, Yq);
  Real var_min = eval_at_it(iv, wn0, wn1, in, wy0, wy1, iy, 0);
  Real var_max = eval_at_it(iv, wn0, wn1, in, wy0, wy1, iy, m_nt - 1);
  return temperature_from_var_precomp(var_min, var_max, iv, var, wn0, wn1, in, wy0, wy1, iy);
}

Real EOSCompOSE::temperature_from_var_precomp(Real var_min,
                                              Real var_max,
                                              int iv,
                                              Real var,
                                              Real wn0,
                                              Real wn1,
                                              int in,
                                              Real wy0,
                                              Real wy1,
                                              int iy) const
{
  // Pre-compute the four base offsets for the (iv, in, iy) cell.
  // Temperature indices are contiguous, so f(it) = m_table[base + it].
  ptrdiff_t const b00 = index(iv, in, iy, 0);
  ptrdiff_t const b01 = index(iv, in, iy + 1, 0);
  ptrdiff_t const b10 = index(iv, in + 1, iy, 0);
  ptrdiff_t const b11 = index(iv, in + 1, iy + 1, 0);

  // Lambda: evaluate the bilinear interpolant at temperature index it,
  // return residual (var - interpolated_value).
  auto f = [=](int it) -> Real
  {
    Real var_pt = wn0 * (wy0 * m_table[b00 + it] + wy1 * m_table[b01 + it]) +
                  wn1 * (wy0 * m_table[b10 + it] + wy1 * m_table[b11 + it]);
    return var - var_pt;
  };

  int ilo  = 0;
  int ihi  = m_nt - 1;
  Real flo = var - var_min;
  Real fhi = var - var_max;

  // Binary search for the sign change.
  // The table variable is monotone in T at fixed (n, Yq), so there is
  // at most one sign change.  Binary search finds it in O(log m_nt).
  if (flo * fhi > 0)
  {
    // Should not happen after the caller's bounds check, but handle
    // gracefully: bracket already at adjacent points is the best we can do.
  }
  else
  {
    // Use the boundary residuals to estimate the root location via
    // false-position on the full index range.  For log-stored variables
    // (ECLOGP, ECLOGE) the interpolant is roughly linear in the uniform
    // log-T index, so this guess typically lands within a few cells of
    // the true root, shrinking the bracket from m_nt to O(1) and saving
    // most of the subsequent binary-search iterations.
    int it_guess =
      static_cast<int>(static_cast<Real>(m_nt - 1) * flo / (flo - fhi));
    it_guess = std::max(0, std::min(m_nt - 2, it_guess));

    Real fg = f(it_guess);
    if (fg * flo <= 0)
    {
      ihi = it_guess;
      fhi = fg;
    }
    else
    {
      ilo = it_guess;
      flo = fg;
    }

    // Refine the bracket with standard binary search.
    while (ihi - ilo > 1)
    {
      int ip  = ilo + (ihi - ilo) / 2;
      Real fp = f(ip);
      if (fp * flo <= 0)
      {
        ihi = ip;
        fhi = fp;
      }
      else
      {
        ilo = ip;
        flo = fp;
      }
    }
  }

  assert(ihi - ilo == 1 || flo * fhi <= 0);
  Real ltlo = m_log_t[ilo];
  Real lthi = m_log_t[ihi];

  if (flo == 0)
  {
    return exp(ltlo);
  }
  if (fhi == 0)
  {
    return exp(lthi);
  }

  // False-position interpolation in log-T for sub-cell accuracy.
  Real lt = ltlo - flo * (lthi - ltlo) / (fhi - flo);
  return exp(lt);
}

Real EOSCompOSE::eval_at_nty(int vi, Real n, Real T, Real Yq) const
{
  return eval_at_lnty(vi, log(n), log(T), Yq);
}

void EOSCompOSE::weight_idx_ln(Real* w0, Real* w1, int* in, Real log_n) const
{
  *in = (log_n - m_log_nb[0]) * m_id_log_nb;
  // if outside table limits, linearly extrapolate
  if (*in > m_nn - 2)
  {
    *in = m_nn - 2;
  }
  else if (*in < 0)
  {
    *in = 0;
  }

  *w1 = (log_n - m_log_nb[*in]) * m_id_log_nb;
  *w0 = 1.0 - (*w1);
}

void EOSCompOSE::weight_idx_yq(Real* w0, Real* w1, int* iy, Real yq) const
{
  *iy = (yq - m_yq[0]) * m_id_yq;
  // if outside table limits, linearly extrapolate
  if (*iy > m_ny - 2)
  {
    *iy = m_ny - 2;
  }
  else if (*iy < 0)
  {
    *iy = 0;
  }

  *w1 = (yq - m_yq[*iy]) * m_id_yq;
  *w0 = 1.0 - (*w1);
}

void EOSCompOSE::weight_idx_lt(Real* w0, Real* w1, int* it, Real log_t) const
{
  *it = (log_t - m_log_t[0]) * m_id_log_t;
  // if outside table limits, linearly extrapolate
  if (*it > m_nt - 2)
  {
    *it = m_nt - 2;
  }
  else if (*it < 0)
  {
    *it = 0;
  }
  *w1 = (log_t - m_log_t[*it]) * m_id_log_t;
  *w0 = 1.0 - (*w1);
}

Real EOSCompOSE::eval_at_lnty(int iv, Real log_n, Real log_t, Real yq) const
{
  int in, iy, it;
  Real wn0, wn1, wy0, wy1, wt0, wt1;

  weight_idx_ln(&wn0, &wn1, &in, log_n);
  weight_idx_yq(&wy0, &wy1, &iy, yq);
  weight_idx_lt(&wt0, &wt1, &it, log_t);

  // Pre-compute the four base offsets; it and it+1 are contiguous.
  ptrdiff_t const b00 = index(iv, in, iy, it);
  ptrdiff_t const b01 = index(iv, in, iy + 1, it);
  ptrdiff_t const b10 = index(iv, in + 1, iy, it);
  ptrdiff_t const b11 = index(iv, in + 1, iy + 1, it);

  return wn0 * (wy0 * (wt0 * m_table[b00] + wt1 * m_table[b00 + 1]) +
                wy1 * (wt0 * m_table[b01] + wt1 * m_table[b01 + 1])) +
         wn1 * (wy0 * (wt0 * m_table[b10] + wt1 * m_table[b10 + 1]) +
                wy1 * (wt0 * m_table[b11] + wt1 * m_table[b11 + 1]));
}

// #else //HDF5OUTPUT
//  Consider adding no-ops here?
// #endif
