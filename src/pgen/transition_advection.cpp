//! \file transition_advection.cpp
//  \brief 1D advection of a cold, pressure-balanced contact for the transition
//  EOS: uniform (n, T, composition, v_x); SCEB and v_y change across a tanh
//  contact. P does not depend on SCEB and v_y is tangential, so the exact
//  solution is rigid advection at v_x with T unchanged everywhere. Any change
//  of T measures the inconsistency between the tau and SCEB transport.
#include <cmath>
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../scalars/scalars.hpp"
#include "../z4c/z4c.hpp"

#if not Z4C_ENABLED
#error "This problem generator must be used with z4c"
#endif
#if !defined(USE_TRANSITION_EOS)
#error "transition_advection needs --eospolicy=eos_transition"
#endif

void MeshBlock::ProblemGenerator(ParameterInput* pin)
{
  const Real rho   = pin->GetReal("problem", "rho");    // code units
  const Real T     = pin->GetReal("problem", "T");      // MeV
  const Real vx    = pin->GetReal("problem", "vx");
  const Real vy_l  = pin->GetReal("problem", "vy_l");
  const Real vy_r  = pin->GetReal("problem", "vy_r");
  const Real eb_l  = pin->GetReal("problem", "sceb_l");
  const Real eb_r  = pin->GetReal("problem", "sceb_r");
  const Real x0    = pin->GetReal("problem", "x0");
  const Real width = pin->GetReal("problem", "width");
  Real Y[NSCALARS] = { 0.0 };
  Y[SCYE] = pin->GetReal("problem", "ye");
  Y[SCXN] = pin->GetReal("problem", "xn");
  Y[SCXP] = pin->GetReal("problem", "xp");
  Y[SCXA] = pin->GetReal("problem", "xa");
  Y[SCXH] = pin->GetReal("problem", "xh");
  Y[SCAH] = pin->GetReal("problem", "ah");
  // optional composition contact (defaults: same as left)
  Real Yr[NSCALARS];
  for (int l = 0; l < NSCALARS; ++l) Yr[l] = Y[l];
  Yr[SCXN] = pin->GetOrAddReal("problem", "xn_r", Y[SCXN]);
  Yr[SCXH] = pin->GetOrAddReal("problem", "xh_r", Y[SCXH]);
  Yr[SCAH] = pin->GetOrAddReal("problem", "ah_r", Y[SCAH]);
  // optional temperature contact (default: isothermal)
  const Real T_r = pin->GetOrAddReal("problem", "T_r", T);

  pz4c->ADMMinkowski(pz4c->storage.adm);
  pz4c->GaugeGeodesic(pz4c->storage.u);
  pz4c->ADMToZ4c(pz4c->storage.adm, pz4c->storage.u);

  const Real mb = peos->GetEOS().GetBaryonMass();
  AthenaArray<Real>& r = pscalars->r;
  for (int k = ks; k <= ke; ++k)
    for (int j = js; j <= je; ++j)
      for (int i = is; i <= ie; ++i)
      {
        const Real f = 0.5 * (1.0 + std::tanh((pcoord->x1v(i) - x0) / width));
        for (int l = 0; l < NSCALARS; ++l)
          r(l, k, j, i) = Y[l] + (Yr[l] - Y[l]) * f;
        r(SCEB, k, j, i) = eb_l + (eb_r - eb_l) * f;
        const Real vy = vy_l + (vy_r - vy_l) * f;
        Real Yc[NSCALARS];
        for (int l = 0; l < NSCALARS; ++l) Yc[l] = r(l, k, j, i);
        const Real Tc = T + (T_r - T) * f;
        const Real W = 1.0 / std::sqrt(1.0 - vx * vx - vy * vy);
        phydro->w(IDN, k, j, i) = rho;
        phydro->w(IPR, k, j, i) = peos->GetEOS().GetPressure(rho / mb, Tc, Yc);
        phydro->w(IVX, k, j, i) = W * vx;
        phydro->w(IVY, k, j, i) = W * vy;
        phydro->w(IVZ, k, j, i) = 0.0;
        phydro->derived_ms(IX_T, k, j, i) = Tc;
      }
  AthenaArray<Real> bb;
  bb.NewAthenaArray(3, ke + 1, je + 1, ie + 1);
  peos->PrimitiveToConserved(phydro->w, r, bb, phydro->u, pscalars->s, pcoord,
                             is, ie, js, je, ks, ke);
}
