//! \file eos_helmholtz.cpp
//  \brief Implementation of EOSHelmholtz

#include "eos_helmholtz.hpp"

#include <hdf5.h>
#include <hdf5_hl.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

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

EOSHelmholtz::EOSHelmholtz()
    : m_id_log_ne(numeric_limits<Real>::quiet_NaN()),
      m_id_log_t(numeric_limits<Real>::quiet_NaN()),
      m_nn(0),
      m_nt(0)
{
  n_species = 1;
  eos_units = &Nuclear;

  min_Y[SCYE] = 0.0;  // will be overwritten by ReadTableFromFile
  min_Y[SCXN] = 0.0;
  min_Y[SCXP] = 0.0;
  min_Y[SCXA] = 0.0;
  min_Y[SCXH] = 0.0;
  min_Y[SCAH] = 1.0;

  max_Y[SCYE] = 1.0;  // will be overwritten by ReadTableFromFile
  max_Y[SCXN] = 1.0;
  max_Y[SCXP] = 1.0;
  max_Y[SCXA] = 1.0;
  max_Y[SCXH] = 1.0;
  max_Y[SCAH] = 500.0;
}

EOSHelmholtz::~EOSHelmholtz()
{
}

// Definitions for static members
Real* EOSHelmholtz::m_log_ne     = nullptr;
Real* EOSHelmholtz::m_log_t      = nullptr;
Real* EOSHelmholtz::m_table      = nullptr;
bool EOSHelmholtz::m_initialized = false;

// Physical nucleon masses, CODATA defaults; EOSTransition overrides these
// with the compose table values (SetNucleonMasses).
Real EOSHelmholtz::mn = EOSHelmholtz::mn_codata;
Real EOSHelmholtz::mp = EOSHelmholtz::mp_codata;

Real EOSHelmholtz::sm_id_log_ne = numeric_limits<Real>::quiet_NaN();
Real EOSHelmholtz::sm_id_log_t  = numeric_limits<Real>::quiet_NaN();

int EOSHelmholtz::sm_nn = 0;
int EOSHelmholtz::sm_nt = 0;

Real EOSHelmholtz::s_mb = numeric_limits<Real>::quiet_NaN();
Real EOSHelmholtz::s_max_n = numeric_limits<Real>::quiet_NaN();
Real EOSHelmholtz::s_min_n = numeric_limits<Real>::quiet_NaN();
Real EOSHelmholtz::s_max_T = numeric_limits<Real>::quiet_NaN();
Real EOSHelmholtz::s_min_T = numeric_limits<Real>::quiet_NaN();

Real EOSHelmholtz::TemperatureFromE(Real n, Real e, Real* Y)
{
  assert(m_initialized);
  return TemperatureFromEps(n, e / (mb * n) - 1, Y);
}

Real EOSHelmholtz::TemperatureFromEps(Real n, Real eps, Real* Y)
{
  Real eps_min = MinimumInternalEnergy(n, Y);
  Real eps_max = MaximumInternalEnergy(n, Y);
  return (eps <= eps_min) ? min_T
       : (eps >= eps_max) ? max_T
                          : temperature_from_var(ECLOGEPS, log(eps), n, Y);
}

Real EOSHelmholtz::TemperatureFromP(Real n, Real p, Real* Y)
{
  assert(m_initialized);
  Real p_min = MinimumPressure(n, Y);
  Real p_max = MaximumPressure(n, Y);

  return (p <= p_min) ? min_T
       : (p >= p_max) ? max_T
                      : temperature_from_var(ECLOGP, log(p), n, Y);
}

Real EOSHelmholtz::TemperatureFromEntropy(Real n, Real s, Real* Y)
{
  assert(m_initialized);
  Real s_min = MinimumEntropy(n, Y);
  Real s_max = MaximumEntropy(n, Y);

  return (s <= s_min) ? min_T
       : (s >= s_max) ? max_T
                      : temperature_from_var(ECENT, s, n, Y);
}

Real EOSHelmholtz::SpecificInternalEnergy(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  return exp(eval_at_nty(ECLOGEPS, n, T, Y));
}

Real EOSHelmholtz::Energy(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  return (SpecificInternalEnergy(n, T, Y) + 1) * n * mb;
}

Real EOSHelmholtz::Pressure(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  return exp(eval_at_nty(ECLOGP, n, T, Y));
}

Real EOSHelmholtz::Abar(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  return 1.0 / inverse_abar(Y);
}

Real EOSHelmholtz::Entropy(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  return eval_at_nty(ECENT, n, T, Y);
}

Real EOSHelmholtz::Enthalpy(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  Real const P = Pressure(n, T, Y);
  Real const e = Energy(n, T, Y);
  return (P + e) / n;
}

Real EOSHelmholtz::SoundSpeed(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  // Timmes & Arnett (1999) Gamma_1 from the (n, T) derivatives, translated
  // to nuclear units. eps and cv = deps/dT are per unit mass (eps
  // dimensionless, cv in 1/MeV), so the mass density is rho = n * mb.
  Real pres    = Pressure(n, T, Y);
  Real eps     = SpecificInternalEnergy(n, T, Y);
  Real dpresdt = eval_at_nty(ECDPDT, n, T, Y);   // dP/dT  [fm^-3]
  // The dpdn table channel is dP_ele/dn_e; converting to a baryon-density
  // derivative requires a factor Ye (n_e = Ye * n_b) before the ion term
  // is added.
  Real dpele_dne = eval_at_lnty(ECDPDN, log(n * Y[SCYE]), log(T));
  Real dpresdn   = Y[SCYE] * dpele_dne + T * inverse_abar(Y); // dP/dn [MeV]
  Real cv      = eval_at_nty(ECDEPSDT, n, T, Y); // deps/dT [1/MeV]
  Real chit    = T / pres * dpresdt;
  Real chin    = n / pres * dpresdn;
  Real x       = pres * chit / (n * mb * T * cv);
  Real gam1    = chit * x + chin;
  // Relativistic sound speed: cs^2 = Gamma_1 * P / (e + P), with the total
  // energy density e = n * mb * (1 + eps).
  Real z = 1.0 + (1.0 + eps) * n * mb / pres;
  return sqrt(gam1 / z);
}

Real EOSHelmholtz::NeutronChemicalPotential(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  // NSE can drive the free-nucleon fraction to exactly zero (or a tiny
  // negative rounding residue), where log(n*Yn*...) is -inf/NaN. Floor at
  // the abundance regularizer RHINE already uses for the same log
  // (log10(y + 1e-25), rhine_optim.hpp:330); the ideal-gas mu is only a
  // large negative number there, as it should be for an absent species.
  Real Yn = fmax(Y[SCXN], 1e-25);
  // Non-degenerate
  return mn + T * log(n * Yn / 2 * pow(sac_const / (mn * T), 1.5));
}

Real EOSHelmholtz::ProtonChemicalPotential(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  // Same floor as NeutronChemicalPotential.
  Real Yp = fmax(Y[SCXP], 1e-25);
  // Non-degenerate
  return mp + T * log(n * Yp / 2 * pow(sac_const / (mp * T), 1.5));
}

Real EOSHelmholtz::ElectronChemicalPotential(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  Real etaele = eval_at_nty(ECETA, n, T, Y);
  return etaele * T + me;
}

Real EOSHelmholtz::BaryonChemicalPotential(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  return NeutronChemicalPotential(n, T, Y);
}

Real EOSHelmholtz::ChargeChemicalPotential(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  return ProtonChemicalPotential(n, T, Y) - NeutronChemicalPotential(n, T, Y);
}

Real EOSHelmholtz::ElectronLeptonChemicalPotential(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  return ElectronChemicalPotential(n, T, Y) +
         ChargeChemicalPotential(n, T, Y);  // mu_e = mu_l - mu_q
}

Real EOSHelmholtz::MinimumEnthalpy()
{
  return m_min_h;
}

Real EOSHelmholtz::MinimumPressure(Real n, Real* Y)
{
  return Pressure(n, min_T, Y);
}

Real EOSHelmholtz::MaximumPressure(Real n, Real* Y)
{
  return Pressure(n, max_T, Y);
}

Real EOSHelmholtz::MinimumInternalEnergy(Real n, Real* Y)
{
  return SpecificInternalEnergy(n, min_T, Y);
}

Real EOSHelmholtz::MaximumInternalEnergy(Real n, Real* Y)
{
  return SpecificInternalEnergy(n, max_T, Y);
}

Real EOSHelmholtz::MinimumEntropy(Real n, Real* Y)
{
  return Entropy(n, min_T, Y);
}

Real EOSHelmholtz::MaximumEntropy(Real n, Real* Y)
{
  return Entropy(n, max_T, Y);
}

void EOSHelmholtz::ReadTableFromFile(std::string fname,
                                     Real min_Ye,
                                     Real max_Ye)
{
#pragma omp critical(EOSHelmholtz_ReadTable)
  {
    if (m_initialized == false)
    {
      herr_t ierr;
      hid_t file_id;
      hsize_t sne, st;

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
      m_log_ne        = new Real[m_nn];
      m_log_t         = new Real[m_nt];
      m_table         = new Real[ECNVARS * m_nn * m_nt];
      double* scratch = new double[m_nn * m_nt];

      // Read ne, t
      // -------------------------------------------------------------------------
      ierr = H5LTread_dataset_double(file_id, "ne", scratch);
      MYH5CHECK(ierr);
      min_n = scratch[0] / min_Ye;
      max_n = scratch[m_nn - 1] / max_Ye;
      for (int in = 0; in < m_nn; ++in)
      {
        m_log_ne[in] = log(scratch[in]);
      }
      m_id_log_ne = 1.0 / (m_log_ne[1] - m_log_ne[0]);

      ierr = H5LTread_dataset_double(file_id, "t", scratch);
      MYH5CHECK(ierr);
      min_T = scratch[0];
      max_T = scratch[m_nt - 1];
      for (int it = 0; it < m_nt; ++it)
      {
        m_log_t[it] = log(scratch[it]);
      }
      m_id_log_t = 1.0 / (m_log_t[1] - m_log_t[0]);

      // the atomic mass unit is used as the baryon mass in the Helmholtz
      // EOS; the table stores it in MeV (eos units)
      ierr = H5LTread_dataset_double(file_id, "mb", scratch);
      MYH5CHECK(ierr);
      mb = scratch[0];
      // The generator's reference mass (= amu): the tabulated eps/depsdt
      // are per unit mass of abar = zbar = 1 matter, i.e. per (mb_file of
      // mass per electron). Fold it into the buffers below so they store
      // the physical, mass-convention-independent energy per ELECTRON in
      // MeV; lookups then divide by the runtime baryon mass (which
      // SetBaryonMass may change after reading).
      const Real mb_file = scratch[0];

      // Read other thermodynamics quantities
      // -------------------------------------------------------------------------
      ierr = H5LTread_dataset_double(file_id, "p", scratch);
      MYH5CHECK(ierr);
      for (int i = 0; i < m_nn * m_nt; ++i)
      {
        m_table[index(ECLOGP, 0, 0) + i] = log(scratch[i]);
      }

      ierr = H5LTread_dataset_double(file_id, "s", scratch);
      MYH5CHECK(ierr);
      copy(&scratch[0], &scratch[m_nn * m_nt], &m_table[index(ECENT, 0, 0)]);

      ierr = H5LTread_dataset_double(file_id, "eps", scratch);
      MYH5CHECK(ierr);
      // stored as log of the energy per electron in MeV (see mb_file above)
      for (int i = 0; i < m_nn * m_nt; ++i)
      {
        if (scratch[i] <= 0.0)
        {
          throw runtime_error(
            "Error reading EOS table: eps dataset contains non-positive "
            "value, cannot take log");
        }
        m_table[index(ECLOGEPS, 0, 0) + i] = log(scratch[i] * mb_file);
      }

      ierr = H5LTread_dataset_double(file_id, "eta", scratch);
      MYH5CHECK(ierr);
      copy(&scratch[0], &scratch[m_nn * m_nt], &m_table[index(ECETA, 0, 0)]);

      ierr = H5LTread_dataset_double(file_id, "depsdt", scratch);
      MYH5CHECK(ierr);
      // per electron, consistent with the eps channel (linear storage)
      for (int i = 0; i < m_nn * m_nt; ++i)
      {
        m_table[index(ECDEPSDT, 0, 0) + i] = scratch[i] * mb_file;
      }

      ierr = H5LTread_dataset_double(file_id, "dpdn", scratch);
      MYH5CHECK(ierr);
      copy(&scratch[0], &scratch[m_nn * m_nt], &m_table[index(ECDPDN, 0, 0)]);

      ierr = H5LTread_dataset_double(file_id, "dpdt", scratch);
      MYH5CHECK(ierr);
      copy(&scratch[0], &scratch[m_nn * m_nt], &m_table[index(ECDPDT, 0, 0)]);

      // Mark table as read
      m_initialized = true;

      // Cleanup
      // -------------------------------------------------------------------------
      delete[] scratch;
      H5Fclose(file_id);

      // Now that we have read everything locally, we must populate
      // the aux static variables to share this data with other threads
      sm_id_log_ne = m_id_log_ne;
      sm_id_log_t  = m_id_log_t;

      sm_nn = m_nn;
      sm_nt = m_nt;

      s_mb    = mb;
      s_max_n = max_n;
      s_min_n = min_n;
      s_max_T = max_T;
      s_min_T = min_T;

    }  // if (m_initialized==false)
  }  // omp critical (EOSHelmholtz_ReadTable)

  // Disseminate applicable static variables to local memory
  m_id_log_ne = sm_id_log_ne;
  m_id_log_t  = sm_id_log_t;

  m_nn = sm_nn;
  m_nt = sm_nt;

  mb          = s_mb;
  max_n       = s_max_n;
  min_n       = s_min_n;
  max_T       = s_max_T;
  min_T       = s_min_T;
  max_Y[SCYE] = max_Ye;
  min_Y[SCYE] = min_Ye;
}

void EOSHelmholtz::SetBaryonMass(Real new_mb)
{
  mb = new_mb;
}

void EOSHelmholtz::SetNucleonMasses(Real new_mn, Real new_mp)
{
  mn = new_mn;
  mp = new_mp;
}

void EOSHelmholtz::SetNSpecies(int n)
{
  if (n > MAX_SPECIES || n < 0)
  {
    throw std::out_of_range(
      "EOSHelmholtz::SetNSpecies - n cannot exceed MAX_SPECIES.");
  }
  n_species = n;
}

Real EOSHelmholtz::temperature_from_var(int iv,
                                        Real var,
                                        Real n,
                                        Real* Y) const
{
  int in;
  Real wn0, wn1;
  weight_idx_ln(&wn0, &wn1, &in, log(n * Y[SCYE]));

  auto f = [=](int it)
  {
    Real var_pt = wn0 * m_table[index(iv, in + 0, it)] +
                  wn1 * m_table[index(iv, in + 1, it)];
    var_pt = add_rad_ion(iv, var_pt, n, exp(m_log_t[it]), Y);
    return var - var_pt;
  };

  int ilo  = 0;
  int ihi  = m_nt - 1;
  Real flo = f(ilo);
  Real fhi = f(ihi);
  while (flo * fhi > 0)
  {
    if (ilo == ihi - 1)
    {
      break;
    }
    else
    {
      ilo += 1;
      flo = f(ilo);
    }
  }
  if (!(flo * fhi <= 0))
  {
    std::cout << "EOSHelmholtz::temperature_from_var failed to bracket root."
              << std::endl;
    std::cout << "iv: " << iv << std::endl;
    std::cout << "var: " << var << std::endl;
    std::cout << "n: " << n << std::endl;
    std::cout << "Yq: " << Y[SCYE] << std::endl;
    std::cout << "flo: " << flo << std::endl;
    std::cout << "fhi: " << fhi << std::endl;
    std::cout << "varlo: " << var - flo << std::endl;
    std::cout << "varhi: " << var - fhi << std::endl;
  }
  assert(flo * fhi <= 0);
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
  assert(ihi - ilo == 1);
  Real lthi = m_log_t[ihi];
  Real ltlo = m_log_t[ilo];

  if (flo == 0)
  {
    return exp(ltlo);
  }
  if (fhi == 0)
  {
    return exp(lthi);
  }

  // Refine within the bracketing cell. The analytic radiation/ion terms
  // make f nonlinear in log T inside the cell, so a single secant step
  // leaves O(1e-3) errors on a coarse temperature grid; iterate a
  // false-position (Anderson-Bjorck) rule to convergence instead.
  Real const v_lo = wn0 * m_table[index(iv, in + 0, ilo)] +
                    wn1 * m_table[index(iv, in + 1, ilo)];
  Real const v_hi = wn0 * m_table[index(iv, in + 0, ihi)] +
                    wn1 * m_table[index(iv, in + 1, ihi)];
  auto g = [=](Real lt)
  {
    Real wt     = (lt - ltlo) / (lthi - ltlo);
    Real var_pt = (1.0 - wt) * v_lo + wt * v_hi;
    return var - add_rad_ion(iv, var_pt, n, exp(lt), Y);
  };

  Real la = ltlo, lb_ = lthi;
  Real fa = flo, fb = fhi;
  Real lt   = la - fa * (lb_ - la) / (fb - fa);
  int side  = 0;
  for (int i = 0; i < 50; ++i)
  {
    Real ft = g(lt);
    if (ft == 0.0)
    {
      break;
    }
    if (ft * fa > 0)
    {
      if (side == 1)
      {
        Real m = 1.0 - ft / fa;
        fb     = (m > 0) ? fb * m : 0.5 * fb;
      }
      la   = lt;
      fa   = ft;
      side = 1;
    }
    else
    {
      if (side == -1)
      {
        Real m = 1.0 - ft / fb;
        fa     = (m > 0) ? fa * m : 0.5 * fa;
      }
      lb_  = lt;
      fb   = ft;
      side = -1;
    }
    Real lt_new = la - fa * (lb_ - la) / (fb - fa);
    if (std::fabs(lt_new - lt) <= 1e-13 * (std::fabs(lt_new) + 1e-13))
    {
      lt = lt_new;
      break;
    }
    lt = lt_new;
  }
  return exp(lt);
}

Real EOSHelmholtz::add_rad_ion(int vi, Real var, Real n, Real T, Real* Y) const
{
  // The buffers store the electron gas PER ELECTRON (eps/depsdt in MeV,
  // converted at read time; s in kB). Converting to the per-baryon(-mass)
  // quantities the analytic radiation/ion terms use requires a factor Ye
  // (n_e = Ye * n_b), and for the per-mass channels additionally 1/mb
  // (the runtime baryon mass, set by SetBaryonMass). Pressure-like
  // channels are intensive in n_e and need no mass factor.
  const Real Ye      = Y[SCYE];
  const Real eps_fac = Ye / mb;
  switch (vi)
  {
    case ECLOGP:
    {
      Real prad = asol / 3.0 * T * T * T * T;
      Real pion = n * inverse_abar(Y) * T;
      return log(exp(var) + prad + pion);
    }
    case ECENT:
    {
      // Sackur-Tetrode equation. Each species term Y*(2.5 - log(...Y...))
      // tends to 0 as Y -> 0 but evaluates to 0 * inf = NaN, so vanishing
      // species must be skipped explicitly.
      Real srad = 4.0 * asol / 3.0 * T * T * T / n;
      Real Yn   = Y[SCXN];
      Real Yp   = Y[SCXP];
      Real Ya   = Y[SCXA] / 4;
      Real Yh   = Y[SCXH] / Y[SCAH];
      Real sn =
        (Yn > 0.0)
          ? Yn * (2.5 - log(n * Yn / g_n * pow(sac_const / (mn * T), 1.5)))
          : 0.0;
      Real sp =
        (Yp > 0.0)
          ? Yp * (2.5 - log(n * Yp / g_p * pow(sac_const / (mp * T), 1.5)))
          : 0.0;
      Real sa =
        (Ya > 0.0)
          ? Ya * (2.5 - log(n * Ya / g_a * pow(sac_const / (ma * T), 1.5)))
          : 0.0;
      Real sh = 0.0;
      if (Yh > 0.0)
      {
        // representative heavy-nucleus mass from the mean baryon mass
        Real mbar = mb * (1 + Y[SCEB]);
        Real mh   = (mbar - Yn * mn - Yp * mp - Ya * ma) / Yh;
        if (mh > 0.0)
        {
          sh = Yh * (2.5 - log(n * Yh / g_h * pow(sac_const / (mh * T), 1.5)));
        }
      }
      return Ye * var + srad + sn + sp + sa + sh;
    }
    case ECLOGEPS:
    {
      Real erad  = asol * T * T * T * T / (n * mb);
      Real eion  = 1.5 * T * inverse_abar(Y) / mb;
      Real ebind = Y[SCEB];
      return log(eps_fac * exp(var) + erad + eion + ebind);
    }
    case ECDEPSDT:
    {
      // per unit mass, consistent with the eps channel above
      Real deraddt = 4.0 * asol * T * T * T / (n * mb);
      Real deiondt = 1.5 * inverse_abar(Y) / mb;
      return eps_fac * var + deraddt + deiondt;
    }
    case ECDPDN:
    {
      // table channel is dP_ele/dn_e; chain rule to d/dn_b needs Ye
      Real dpraddn = 0.0;
      Real dpiondn = T * inverse_abar(Y);
      return Ye * var + dpraddn + dpiondn;
    }
    case ECDPDT:
    {
      Real dpraddt = 4.0 / 3.0 * asol * T * T * T;
      Real dpiondt = n * inverse_abar(Y);
      return var + dpraddt + dpiondt;
    }
    case ECETA:
    {
      return var;  // no correction electron degeneracy parameter
    }
  }
  throw std::logic_error("Invalid variable index in add_rad_ion");
}

Real EOSHelmholtz::eval_at_nty(int vi, Real n, Real T, Real* Y) const
{
  Real var = eval_at_lnty(vi, log(n * Y[SCYE]), log(T));
  return add_rad_ion(vi, var, n, T, Y);
}

void EOSHelmholtz::weight_idx_ln(Real* w0, Real* w1, int* in, Real log_n) const
{
  *in = (log_n - m_log_ne[0]) * m_id_log_ne;
  // if outside table limits, linearly extrapolate
  if (*in > m_nn - 2)
  {
    *in = m_nn - 2;
  }
  else if (*in < 0)
  {
    *in = 0;
  }

  *w1 = (log_n - m_log_ne[*in]) * m_id_log_ne;
  *w0 = 1.0 - (*w1);
}

void EOSHelmholtz::weight_idx_lt(Real* w0, Real* w1, int* it, Real log_t) const
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

Real EOSHelmholtz::eval_at_lnty(int iv, Real log_n, Real log_t) const
{
  // This only returns the electron part
  int in, it;
  Real wn0, wn1, wt0, wt1;

  weight_idx_ln(&wn0, &wn1, &in, log_n);
  weight_idx_lt(&wt0, &wt1, &it, log_t);

  return wn0 * (wt0 * m_table[index(iv, in + 0, it + 0)] +
                wt1 * m_table[index(iv, in + 0, it + 1)]) +
         wn1 * (wt0 * m_table[index(iv, in + 1, it + 0)] +
                wt1 * m_table[index(iv, in + 1, it + 1)]);
}
