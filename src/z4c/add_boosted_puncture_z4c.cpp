//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file puncture_z4c.cpp
//  \brief implementation of functions in the Z4c class for boosted punctures, based on
//         AddPuncture thorn from THCSupport, itself based on BBIDdataReader.c from BAM.

// C++ standard headers
#include <cmath>
#include <sstream>

// Athena++ headers
#include "z4c.hpp"
#include "z4c_macro.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"

// No need to reinvent the wheel -- pull in the right functionality from PrimitiveSolver
// so that we can invert the spatial metric.
//#include "primitive/geom_math.hpp"

//namespace {
void ConstructBoost(Real lam[4][4], Real ilam[4][4], Real vu[3]);

void SetPuncture(Real M, Real eps, Real px[3], Real pv[3], const Real r[3],
                 Real& psi4, Real dpsi4[3], Real& alpha, Real dalpha[3],
                 Real lam[4][4]);

void BoostPuncture(Real pv[3], const Real px[3], Real& psi4, Real dpsi4[3],
                   Real& alpha, Real dalpha[3], Real beta_u[3], Real g3d[3][3],
                   Real K3d[3][3], Real lam[4][4], Real ilam[4][4]);

void Invert4Matrix(Real md[4][4], Real mu[4][4], Real& detg);

void GetChristoffels(Real g[4][4], Real dg[4][4][4], Real Gamma[4][4][4]);
//} // Namespace

//----------------------------------------------------------------------------------------
// \!fn void Z4c::ADMOnePuncture(AthenaArray<Real> & u)
// \brief Add 

void Z4c::ADMAddBoostedPuncture(ParameterInput *pin, AthenaArray<Real>& u_adm,
                                AthenaArray<Real>& u_z4c, int index) {
  ADM_vars adm;
  Z4c_vars z4c;
  SetADMAliases(u_adm, adm);
  SetZ4cAliases(u_z4c, z4c);

  // Read in parameters
  // ADM mass
  std::stringstream ss;
  ss << "punc_ADM_mass" << index;
  Real ADM_mass = pin->GetOrAddReal("problem", ss.str(), 1.0);

  // Radial epsilon for setting puncture
  ss.str(std::string());
  ss << "punc_eps" << index;
  Real punc_eps = pin->GetOrAddReal("problem", ss.str(), 1e-15);

  // Velocity (v)
  ss.str(std::string());
  ss << "punc_vx" << index;
  Real punc_vx = pin->GetOrAddReal("problem", ss.str(), 0.0);
  ss.str(std::string());
  ss << "punc_vy" << index;
  Real punc_vy = pin->GetOrAddReal("problem", ss.str(), 0.0);
  ss.str(std::string());
  ss << "punc_vz" << index;
  Real punc_vz = pin->GetOrAddReal("problem", ss.str(), 0.0);
  Real v_u[3] = {-punc_vx, -punc_vy, -punc_vz};

  // Puncture location
  ss.str(std::string());
  ss << "punc_x" << index;
  Real punc_x = pin->GetOrAddReal("problem", ss.str(), 0.0);
  ss.str(std::string());
  ss << "punc_y" << index;
  Real punc_y = pin->GetOrAddReal("problem", ss.str(), 0.0);
  ss.str(std::string());
  ss << "punc_z" << index;
  Real punc_z = pin->GetOrAddReal("problem", ss.str(), 0.0);
  Real px[3] = {punc_x, punc_y, punc_z};

  // Necessary local variables for the puncture calculation
  Real lam[4][4], ilam[4][4], r[3], g3d[3][3], K3d[3][3];
  Real psi4, dpsi4[3], alpha, dalpha[3], beta_u[3];

  // Compute the boost
  ConstructBoost(lam, ilam, v_u);
  
  GLOOP2(k,j) {
    GLOOP1(i) {
      r[0] = mbi.x1(i);
      r[1] = mbi.x2(j);
      r[2] = mbi.x3(k);

      // Make the puncture, then boost it.
      SetPuncture(ADM_mass, punc_eps, px, v_u, r, psi4, dpsi4, alpha, dalpha, lam);
      BoostPuncture(v_u, r, psi4, dpsi4, alpha, dalpha, beta_u, g3d, K3d, lam, ilam);

      // Superimpose the puncture on top of the existing spacetime.
      // Also subtract off Minkowski space.
      // We note that the lapse tends toward zero, so if the PBH is too close to the
      // neutron star, it results in a negative lapse. Therefore, we put this janky
      // floor in here to fix that.
      z4c.alpha(k,j,i) = std::fmax(z4c.alpha(k,j,i) + alpha - 1.0, punc_eps);
      adm.psi4(k,j,i) += psi4;
      for (int a = 0; a < 3; a++) {
        z4c.beta_u(a,k,j,i) += beta_u[a];
        for (int b = a; b < 3; b++) {
          adm.g_dd(a, b, k, j, i) += g3d[a][b];
          adm.K_dd(a, b, k, j, i) += K3d[a][b];
        }
        adm.g_dd(a, a, k, j, i) -= 1.0;
      }
    }
  }
}

void ConstructBoost(Real lam[4][4], Real ilam[4][4], Real vu[3]) {
  Real vsq = std::sqrt(vu[0]*vu[0] + vu[1]*vu[1] + vu[2]*vu[2]);

  // If our velocity is zero, we return the identity matrix.
  if (vsq == 0.0 || vsq == -0.0) {
    // Zero out the boosts
    for (int a = 0; a < 4; a++) {
      for (int b = a; b < 4; b++) {
        lam[a][b] = lam[b][a] = 0.0;
        ilam[a][b] = ilam[b][a] = 0.0;
      }
    }
    lam[0][0] = lam[1][1] = lam[2][2] = lam[3][3] = 1.0;
    ilam[0][0] = ilam[1][1] = ilam[2][2] = ilam[3][3] = 1.0;
    return;
  }

  // Lorentz factor
  Real gamma = 1.0/std::sqrt(1.0 - vsq);

  lam[0][0] = gamma;
  lam[0][1] = lam[1][0] = vu[0]*gamma;
  lam[0][2] = lam[2][0] = vu[1]*gamma;
  lam[0][3] = lam[3][0] = vu[2]*gamma;
  lam[1][1] = 1.0 + (gamma - 1.0)*(vu[0]*vu[0])/vsq;
  lam[2][2] = 1.0 + (gamma - 1.0)*(vu[1]*vu[1])/vsq;
  lam[3][3] = 1.0 + (gamma - 1.0)*(vu[2]*vu[2])/vsq;
  lam[1][2] = lam[2][1] = (gamma - 1.0)*vu[0]*vu[1]/vsq;
  lam[1][3] = lam[3][1] = (gamma - 1.0)*vu[0]*vu[2]/vsq;
  lam[2][3] = lam[3][2] = (gamma - 1.0)*vu[1]*vu[2]/vsq;

  Real detg;
  Invert4Matrix(lam, ilam, detg);
}

void Invert4Matrix(Real md[4][4], Real mu[4][4], Real& detg) {
  // Copied from Mathematica
  Real a = md[0][0];
  Real b = md[0][1];
  Real c = md[0][2];
  Real d = md[0][3];
  Real e = md[1][1];
  Real f = md[1][2];
  Real g = md[1][3];
  Real h = md[2][2];
  Real i = md[2][3];
  Real j = md[3][3];

  detg = -(std::pow(c,2)*std::pow(g,2)) + a*std::pow(g,2)*h +
        std::pow(d,2)*(-std::pow(f,2) + e*h) + 2*b*c*g*i - 2*a*f*g*i -
        std::pow(b,2)*std::pow(i,2) + a*e*std::pow(i,2) +
        2*d*(c*f*g - b*g*h - c*e*i + b*f*i) + (std::pow(c,2)*e -
            2*b*c*f + a*std::pow(f,2) + std::pow(b,2)*h - a*e*h)*j;
  Real oodetg = 1./detg;

  mu[0][0] = oodetg*(std::pow(g,2)*h - 2*f*g*i + std::pow(f,2)*j + e*(std::pow(i,2) - h*j));
  mu[0][1] = oodetg*(-(d*g*h) + d*f*i + c*g*i - b*std::pow(i,2) - c*f*j + b*h*j);
  mu[0][2] = oodetg*(d*f*g - c*std::pow(g,2) - d*e*i + b*g*i + c*e*j - b*f*j);
  mu[0][3] = oodetg*(-(d*std::pow(f,2)) + c*f*g + d*e*h - b*g*h - c*e*i + b*f*i);
  mu[1][1] = oodetg*(std::pow(d,2)*h - 2*c*d*i + std::pow(c,2)*j + a*(std::pow(i,2) - h*j));
  mu[1][2] = oodetg*(-(std::pow(d,2)*f) + c*d*g + b*d*i - a*g*i - b*c*j + a*f*j);
  mu[1][3] = oodetg*(c*d*f - std::pow(c,2)*g - b*d*h + a*g*h + b*c*i - a*f*i);
  mu[2][2] = oodetg*(std::pow(d,2)*e - 2*b*d*g + std::pow(b,2)*j + a*(std::pow(g,2) - e*j));
  mu[2][3] = oodetg*(-(c*d*e) + b*d*f + b*c*g - a*f*g - std::pow(b,2)*i + a*e*i);
  mu[3][3] = oodetg*(std::pow(c,2)*e - 2*b*c*f + std::pow(b,2)*h + a*(std::pow(f,2) - e*h));

  mu[1][0] = mu[0][1];
  mu[2][0] = mu[0][2];
  mu[3][0] = mu[0][3];
  mu[2][1] = mu[1][2];
  mu[3][1] = mu[1][3];
  mu[3][2] = mu[2][3];
}

void SetPuncture(Real M, Real eps, Real px[3], Real pv[3], const Real r[3],
                 Real& psi4, Real dpsi4[3], Real& alpha, Real dalpha[3],
                 Real lam[4][4]) {
  // Set transformed coordinates
  Real x, y, z;
  x = lam[1][1]*(r[0] - px[0]) + lam[1][2]*(r[1] - px[1]) + lam[1][3]*(r[2] - px[2]);
  y = lam[2][1]*(r[0] - px[0]) + lam[2][2]*(r[1] - px[1]) + lam[2][3]*(r[2] - px[2]);
  z = lam[3][1]*(r[0] - px[0]) + lam[3][2]*(r[1] - px[1]) + lam[3][3]*(r[2] - px[2]);

  Real riso = std::sqrt(x*x + y*y + z*z);
  riso = std::fmax(riso, eps);

  // Choose precollapsed gauge.
  psi4 = std::pow(1.0 + 0.5*M/riso, 4.0);
  alpha = 1.0/std::sqrt(psi4);

  dpsi4[0] = -2.0*M*std::pow(1.0 + 0.5*M/riso,3.0)/(riso*riso)*x/riso;
  dpsi4[1] = -2.0*M*std::pow(1.0 + 0.5*M/riso,3.0)/(riso*riso)*y/riso;
  dpsi4[2] = -2.0*M*std::pow(1.0 + 0.5*M/riso,3.0)/(riso*riso)*z/riso;
  dalpha[0] = -0.5*alpha*alpha*alpha*dpsi4[0];
  dalpha[1] = -0.5*alpha*alpha*alpha*dpsi4[1];
  dalpha[2] = -0.5*alpha*alpha*alpha*dpsi4[2];

}

void BoostPuncture(Real pv[3], const Real px[3], Real& psi4, Real dpsi4[3],
                   Real& alpha, Real dalpha[3], Real beta_u[3], Real g3d[3][3],
                   Real K3d[3][3], Real lam[4][4], Real ilam[4][4]) {
  // If there's no boost, this is easy: just set the standard puncture metric and move on.
  if (pv[0] == 0.0 && pv[1] == 0.0 && pv[2] == 0.0) {
    alpha = 1.0/std::sqrt(psi4);
    beta_u[0] = 0.0;
    beta_u[1] = 0.0;
    beta_u[2] = 0.0;

    g3d[0][0] = g3d[1][1] = g3d[2][2] = psi4;
    g3d[0][1] = g3d[1][0] = 0.0;
    g3d[0][2] = g3d[2][0] = 0.0;
    g3d[1][2] = g3d[2][1] = 0.0;

    for (int a = 0; a < 3; a++) {
      for (int b = 0; b < 3; b++) {
        K3d[a][b] = 0.0;
      }
    }
    return;
  }

  // Pre-boost quantities
  Real g[4][4], u[4], delg[4][4][4], Gamma[4][4][4];
  // Post-boost quantities
  Real gp[4][4], up[4], delgp[4][4][4], Gammap[4][4][4];
  Real gip[4][4];

  // Construct the pre-boost metric
  g[0][0] = -alpha*alpha;
  for (int a = 1; a < 4; a++) {
    g[0][a] = g[a][0] = 0.0;
  }
  for (int a = 1; a < 4; a++) {
    for (int b = 1; b < 4; b++) {
      g[a][b] = psi4*(Real)(a == b);
    }
  }

  // Derivatives of pre-boost metric.
  // Time derivative
  for (int a = 0; a < 4; a++) {
    for (int b = 0; b < 4; b++) {
      delg[0][a][b] = 0.0;
    }
  }
  // Spatial derivatives
  for (int a = 1; a < 4; a++) {
    // Lapse
    delg[a][0][0] = -2.0*alpha*dalpha[a];
    // Shift
    for (int b = 1; b < 4; b++) {
      delg[a][b][0] = delg[a][0][b] = 0.0;
    }
    // Metric
    for (int b = 1; b < 4; b++) {
      for (int c = 1; c < 4; c++) {
        delg[a][b][c] = dpsi4[a]*(Real)(b==c);
      }
    }
  }

  // Construct the Christoffel symbols
  GetChristoffels(g, delg, Gamma);

  // Make coordinate transformation for u^\mu, g_\mu\nu, Gamma^\sigma_\mu\nu
  for (int a = 0; a < 4; a++) {
    for (int b = 0; b < 4; b++) {
      for (int c = 0; c < 4; c++) {
        up[a] = 0.0;
        gp[a][b] = 0.0;
        Gammap[a][b][c] = 0.0;
        for (int d = 0; d < 4; d++) {
          up[a] += ilam[d][a] * u[d];
          for (int e = 0; e < 4; e++) {
            gp[a][b] += lam[a][d] * lam[b][e] * g[d][e];
            for (int t = 0; t < 4; t++) {
              Gammap[a][b][c] += ilam[a][d]*lam[b][e]*lam[c][t]*Gamma[d][e][t];
              Gammap[a][b][c] += ilam[a][d]*0.; // 0 b/c lam is constnat.
            }
          }
        }
      }
    }
  }

  // compute inverse gp^{\mu\nu}
  Real detg;
  Invert4Matrix(gp, gip, detg);

  // Move transformed values into 3-metric.
  for (int a = 0; a < 3; a++) {
    for (int b = 0; b < 3; b++) {
      g3d[a][b] = gp[1+a][1+b];
    }
    beta_u[a] = -gip[0][a]/gip[0][0];
  }
  alpha = std::sqrt(-1.0/gip[0][0]);

  // Set extrinsic curvature
  for (int a = 0; a < 3; a++) {
    for (int b = 0; b < 3; b++) {
      K3d[a][b] = -alpha*Gammap[0][1+a][1+b];
    }
  }
}

void GetChristoffels(Real g[4][4], Real dg[4][4][4], Real Gamma[4][4][4]) {
  Real gi[4][4], detg;

  Invert4Matrix(g, gi, detg);

  for (int a = 0; a < 4; a++) {
    for (int b = 0; b < 4; b++) {
      for (int c = 0; c < 4; c++) {
        Gamma[a][b][c] = 0.0;
        for (int d = 0; d < 4; d++) {
          Gamma[a][b][c] += 0.5*gi[a][d]*(dg[c][b][d] + dg[b][c][d] - dg[d][b][c]);
        }
      }
    }
  }
}
