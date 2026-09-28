// Standalone driver for the transition-EOS entropy inversion
// (EOSTransition::TemperatureFromEntropy). Compiles the gr-athena
// primitive sources directly against a stubbed Globals.
//
//   ./build.sh && ./test_eos_entropy_inversion

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "z4c/primitive/eos.hpp"
#include "z4c/primitive/eos_transition.hpp"
#include "z4c/primitive/reset_floor_transition.hpp"

using namespace Primitive;

namespace Globals { int my_rank = 0, nranks = 1, mpi_tag_ub = 0; }

static EOS<EOSTransition, ResetFloorTransition>* peos = nullptr;
static int n_fail = 0, n_pass = 0;
static Real COMPOSE_MIN_T = 0.1;

static Real RelErr(Real a, Real b)
{
  Real s = std::max(std::abs(a), std::abs(b));
  return (s > 0.0) ? std::abs(a - b) / s : 0.0;
}

// A composition consistent with NSE at these densities: mostly heavy
// nuclei and alphas, few free nucleons. The blended s(T) is monotone for
// this mixture (see TestStripMonotonicity).
static void NSELikeComposition(Real* Y)
{
  for (int i = 0; i < MAX_SPECIES; ++i) Y[i] = 0.0;
  Y[SCYE] = 0.30; Y[SCXN] = 0.020; Y[SCXP] = 0.001; Y[SCXA] = 0.100;
  Y[SCXH] = 0.879; Y[SCAH] = 56.0; Y[SCEB] = 0.0085;
}

// A deliberately out-of-equilibrium mixture: 55% free neutrons at 1e7
// g/cc. Its EIR mixing entropy exceeds the NSE table value, so the blend
// is non-monotone across the strip. Kept to document that limit.
static void GenericComposition(Real* Y)
{
  for (int i = 0; i < MAX_SPECIES; ++i) Y[i] = 0.0;
  Y[SCYE] = 0.30; Y[SCXN] = 0.55; Y[SCXP] = 0.05; Y[SCXA] = 0.25;
  Y[SCXH] = 0.15; Y[SCAH] = 56.0; Y[SCEB] = 0.002;
}

static void Check(bool ok, const char* what, const char* detail = "")
{
  if (ok) { ++n_pass; std::printf("  [ ok ] %-22s %s\n", what, detail); }
  else    { ++n_fail; std::printf("  [FAIL] %-22s %s\n", what, detail); }
}

struct Regime { const char* name; Real n; Real T; };
static const Regime regimes[] = {
  { "EIR regime",         1e-9,  0.30 },
  { "EIR, cold",          1e-8,  0.10 },
  { "compose regime",     1e-3,  1.00 },
  { "T strip",            1e-8,  0.55 },
  { "density strip",      3e-12, 1.00 },
  { "validity ramp",      3e-7,  0.30 },
  { "validity ramp cold", 3e-7,  0.12 },
};

// 1. Round trip T -> s -> T in every regime, against the energy inversion
//    already in production as the reference.
static void TestRoundTrip()
{
  std::printf("Round trip in each regime (NSE-like composition)\n");
  for (auto& r : regimes)
  {
    Real Y[MAX_SPECIES]; NSELikeComposition(Y);
    Real e    = peos->GetEnergy(r.n, r.T, Y);
    Real T_e  = peos->GetTemperatureFromE(r.n, e, Y);
    Real s    = peos->GetEntropyPerBaryon(r.n, r.T, Y);
    Real T_s  = peos->GetTemperatureFromEntropy(r.n, s, Y);
    char buf[256];
    std::snprintf(buf, sizeof buf,
                  "s=%9.4f  T(s)=%.6e (err %.1e)  T(e) err %.1e",
                  s, T_s, RelErr(r.T, T_s), RelErr(r.T, T_e));
    Check(RelErr(r.T, T_s) < 1e-6, r.name, buf);
  }
}

// 2. Dense sweep over the joint domain. T below the compose table minimum
//    is excluded: both inversions clamp there by design.
static void TestSweep(void (*comp)(Real*), const char* label, bool strict)
{
  const int NN = 28, NT = 28;
  const Real ln_lo = std::log(1e-12), ln_hi = std::log(1e-2);
  const Real lT_lo = std::log(COMPOSE_MIN_T), lT_hi = std::log(20.0);
  int bad_s = 0, bad_e = 0, tried = 0;
  Real worst = 0.0, worst_n = 0, worst_T = 0;
  for (int a = 0; a < NN; ++a)
  for (int b = 0; b < NT; ++b)
  {
    Real n = std::exp(ln_lo + (ln_hi - ln_lo) * a / (NN - 1.0));
    Real T = std::exp(lT_lo + (lT_hi - lT_lo) * b / (NT - 1.0));
    Real Y[MAX_SPECIES]; comp(Y);
    Real T_s, T_e;
    try
    {
      T_s = peos->GetTemperatureFromEntropy(
              n, peos->GetEntropyPerBaryon(n, T, Y), Y);
      T_e = peos->GetTemperatureFromE(n, peos->GetEnergy(n, T, Y), Y);
    }
    catch (std::exception&) { continue; }  // out-of-table corners
    ++tried;
    if (RelErr(T, T_e) > 1e-6) ++bad_e;
    Real err = RelErr(T, T_s);
    if (err > worst) { worst = err; worst_n = n; worst_T = T; }
    if (err > 1e-6) ++bad_s;
  }
  char buf[256];
  std::snprintf(buf, sizeof buf,
                "%d/%d off (energy %d); worst %.1e at n=%.2e T=%.2e",
                bad_s, tried, bad_e, worst, worst_n, worst_T);
  Check(strict ? (bad_s == 0 && tried > 400) : (tried > 400), label, buf);
}

// 3. The blend must not fold s(T) over for a composition near NSE. This
//    is what makes the inversion single-valued in production.
static void TestStripMonotonicity()
{
  std::printf("Blended s(T) monotone across the strip\n");
  const Real ns[] = { 1e-9, 1e-8, 1e-7, 1e-6 };
  for (Real n : ns)
  {
    Real Y[MAX_SPECIES]; NSELikeComposition(Y);
    Real prev = -1e300;
    Real worst_drop = 0.0, at_T = 0.0;
    for (int i = 0; i < 200; ++i)
    {
      Real T = 0.30 + (0.90 - 0.30) * i / 199.0;
      Real s = peos->GetEntropyPerBaryon(n, T, Y);
      if (s < prev && prev - s > worst_drop) { worst_drop = prev - s; at_T = T; }
      prev = s;
    }
    char buf[128];
    std::snprintf(buf, sizeof buf, "n=%.0e  largest drop %.2e at T=%.3f",
                  n, worst_drop, at_T);
    Check(worst_drop < 1e-10, "monotone", buf);
  }
}

// 4. The Liebendoerfer prescription treats GetElectronLeptonChemicalPotential
//    as the conjugate of Ye. Measure de/dYe at fixed (n, s) and see where
//    that actually holds.
static void TestConjugate()
{
  std::printf("de/dYe at fixed (n,s) vs GetElectronLeptonChemicalPotential\n");
  for (auto& r : regimes)
  {
    Real Y[MAX_SPECIES]; NSELikeComposition(Y);
    const Real n = r.n, T = r.T;
    const Real mu = peos->GetElectronLeptonChemicalPotential(n, T, Y);
    const Real s0 = peos->GetEntropyPerBaryon(n, T, Y);
    const Real h  = 1e-4;

    Real Yp[MAX_SPECIES], Ym[MAX_SPECIES];
    for (int i = 0; i < MAX_SPECIES; ++i) { Yp[i] = Y[i]; Ym[i] = Y[i]; }
    Yp[SCYE] += h; Ym[SCYE] -= h;

    Real Tp, Tm;
    try
    {
      Tp = peos->GetTemperatureFromEntropy(n, s0, Yp);
      Tm = peos->GetTemperatureFromEntropy(n, s0, Ym);
    }
    catch (std::exception& ex) { Check(false, r.name, ex.what()); continue; }

    Real up   = (peos->GetEnergy(n, Tp, Yp) - peos->GetEnergy(n, T, Y))
                / (n * h);
    Real down = (peos->GetEnergy(n, T, Y) - peos->GetEnergy(n, Tm, Ym))
                / (n * h);
    Real ctr  = 0.5 * (up + down);
    char buf[256];
    std::snprintf(buf, sizeof buf,
                  "mu=%8.3f  de/dYe|_s = %8.3f (one-sided %7.3f / %7.3f)",
                  mu, ctr, up, down);
    // Only the NSE branch treats Ye as its sole composition variable; on
    // the EIR branch the mass fractions are advected and frozen, so
    // moving Ye alone is not a capture and mu is not its conjugate.
    bool nse = (peos->TransitionFactor(n, T) == 1.0);
    Check(!nse || RelErr(mu, ctr) < 2e-2, r.name, buf);
  }
}

// 5. A finite capture step taken in s and in e. The two forms are the same
//    first law in different variables, so what separates them is the
//    table's own interpolation consistency, measured here as the energy
//    the entropy step actually deposits versus the one it prescribes.
static void TestCaptureStep()
{
  std::printf("Capture step: entropy form vs energy form\n");
  const Real E_nu = 10.0;  // MeV
  const Real dYe  = -1e-4;
  for (auto& r : regimes)
  {
    Real Y[MAX_SPECIES]; NSELikeComposition(Y);
    const Real n = r.n, T = r.T;
    const Real mu    = peos->GetElectronLeptonChemicalPotential(n, T, Y);
    const Real E_esc = std::min(E_nu, mu);
    const Real E0    = peos->GetEnergy(n, T, Y);
    const Real s0    = peos->GetEntropyPerBaryon(n, T, Y);

    Real Yn[MAX_SPECIES];
    for (int i = 0; i < MAX_SPECIES; ++i) Yn[i] = Y[i];
    Yn[SCYE] += dYe;

    const Real dE = n * dYe * E_esc;         // prescribed energy loss
    Real T_E, T_s;
    try
    {
      T_E = peos->GetTemperatureFromE(n, E0 + dE, Yn);
      T_s = peos->GetTemperatureFromEntropy(n, s0 - dYe * (mu - E_esc) / T,
                                            Yn);
    }
    catch (std::exception& ex) { Check(false, r.name, ex.what()); continue; }

    // How much energy the entropy step actually moved, against the budget
    // the same step prescribes.
    Real dE_s   = peos->GetEnergy(n, T_s, Yn) - E0;
    Real budget = (dE != 0.0) ? std::abs(dE_s - dE) / std::abs(dE) : 0.0;
    // Only meaningful where mu is the conjugate of Ye, i.e. on the NSE
    // branch; elsewhere the prescription itself does not apply and the
    // number is reported, not asserted.
    bool nse = (peos->TransitionFactor(n, T) == 1.0);
    char buf[256];
    std::snprintf(buf, sizeof buf,
                  "%s dT_e=%+.3e dT_s=%+.3e  budget off by %6.2f%%",
                  nse ? "NSE " : "----", T_E - T, T_s - T, 100.0 * budget);
    Check(!nse || budget < 0.15, r.name, buf);
  }
}

// 6. Above the trapping density the capture is isentropic. The entropy
//    form enforces that exactly; the energy form only to the table's
//    interpolation consistency. Measure the latter.
static void TestTrappedIsentropy()
{
  std::printf("Trapped capture (E_esc = mu): entropy drift\n");
  const Real dYe = -1e-4;
  for (auto& r : regimes)
  {
    Real Y[MAX_SPECIES]; NSELikeComposition(Y);
    const Real n = r.n, T = r.T;
    const Real mu = peos->GetElectronLeptonChemicalPotential(n, T, Y);
    const Real s0 = peos->GetEntropyPerBaryon(n, T, Y);

    Real Yn[MAX_SPECIES];
    for (int i = 0; i < MAX_SPECIES; ++i) Yn[i] = Y[i];
    Yn[SCYE] += dYe;

    Real T_E;
    try
    {
      T_E = peos->GetTemperatureFromE(
              n, peos->GetEnergy(n, T, Y) + n * dYe * mu, Yn);
    }
    catch (std::exception& ex) { Check(false, r.name, ex.what()); continue; }

    Real ds  = peos->GetEntropyPerBaryon(n, T_E, Yn) - s0;
    Real rel = std::abs(ds) / std::max(std::abs(s0), 1e-30);
    bool nse = (peos->TransitionFactor(n, T) == 1.0);
    char buf[256];
    std::snprintf(buf, sizeof buf,
                  "%s s0=%9.4f  energy form drifts ds=%+.3e (%.3f%% of s)",
                  nse ? "NSE " : "----", s0, ds, 100.0 * rel);
    Check(!nse || rel < 1e-3, r.name, buf);
  }
}

// 7. An ash-marked cell is held on the NSE branch by TransitionWeight even
//    inside the strip, which is where the deleptonization runs. The
//    inversion must follow it there.
static void TestAshBranch()
{
  std::printf("Ash-marked cells invert on the NSE branch\n");
  const Regime ash[] = {
    { "strip, ash",      1e-8,  0.55 },
    { "below strip, ash", 1e-8, 0.45 },
    { "EIR side, ash",   1e-9,  0.30 },
  };
  for (auto& r : ash)
  {
    Real Y[MAX_SPECIES]; NSELikeComposition(Y);
    Y[SCASH] = 1.0;
    Real T_s;
    try
    {
      T_s = peos->GetTemperatureFromEntropy(
              r.n, peos->GetEntropyPerBaryon(r.n, r.T, Y), Y);
    }
    catch (std::exception& ex) { Check(false, r.name, ex.what()); continue; }
    char buf[192];
    std::snprintf(buf, sizeof buf, "w=%.3f  T=%.6e (err %.1e)",
                  ((EOSTransition*)peos)->TransitionWeight(r.n, r.T, Y),
                  T_s, RelErr(r.T, T_s));
    Check(RelErr(r.T, T_s) < 1e-6, r.name, buf);
  }
}

// 8. With ash_forces_nse = false (post-bounce) the marker is passive: the
//    weight is the thermodynamic one, so marked matter in or below the strip
//    freezes out like fuel.
static void TestAshPassive()
{
  std::printf("Passive ash marker follows the thermodynamic weight\n");
  EOSTransition* pt = (EOSTransition*)peos;
  pt->SetAshForcesNSE(false);
  const Regime ash[] = {
    { "strip, passive ash",       1e-8, 0.55 },
    { "below strip, passive ash", 1e-8, 0.45 },
    { "EIR side, passive ash",    1e-9, 0.30 },
  };
  for (auto& r : ash)
  {
    Real Y[MAX_SPECIES]; NSELikeComposition(Y);
    Y[SCASH] = 1.0;
    const Real w = pt->TransitionWeight(r.n, r.T, Y);
    const Real f = pt->TransitionFactor(r.n, r.T);
    char buf[96];
    std::snprintf(buf, sizeof buf, "weight %.3f  factor %.3f", w, f);
    Check(w == f, r.name, buf);
  }
  pt->SetAshForcesNSE(true);
}

int main()
{
  EOS<EOSTransition, ResetFloorTransition> eos;
  const char* env  = getenv("EOS_TEST_DATA");
  const std::string data =
    env ? std::string(env) + "/"
        : std::string(getenv("HOME")) +
            "/repos/NR/primitive-solver/tests/data/";
  try
  {
    eos.InitializeTables((data + "SFHo_full.h5").c_str(),
                         (data + "electron_table.h5").c_str(), 930.4117);
  }
  catch (std::exception& e)
  {
    std::printf("table init failed: %s\n", e.what());
    return 1;
  }
  eos.SetDensityFloor(1e-15);
  {
    Real ld_n, hd_n, ld_t, hd_t;
    eos.GetTableBoundaries(ld_n, hd_n, ld_t, hd_t);
    eos.SetTableBoundaries(ld_n, hd_n, ld_t, hd_t);
  }
  peos = &eos;

  TestRoundTrip();
  TestStripMonotonicity();
  std::printf("Sweep over the joint domain\n");
  TestSweep(&NSELikeComposition, "NSE-like", true);
  TestSweep(&GenericComposition, "out of equilibrium", false);
  TestConjugate();
  TestCaptureStep();
  TestTrappedIsentropy();
  TestAshBranch();
  TestAshPassive();

  std::printf("\n%d passed, %d failed\n", n_pass, n_fail);
  return n_fail != 0;
}
