#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <string>

#include "../athena.hpp"
#include "../athena_aliases.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../mesh/mesh_refinement.hpp"
#include "../parameter_input.hpp"
#include "../utils/utils.hpp"
#include "../z4c/z4c.hpp"

#define UDENS (6.1762691458861632e+17)

using namespace std;
using namespace gra::aliases;

//========================================================================================
// Utilities
//========================================================================================
namespace {

void skipline(FILE *fptr);

class StellarProfile {
  enum vars {
    alp = 0,
    gxx = 1,
    mass = 2,
    vel = 3,
    rho = 4,
    temp = 5,
    ye = 6,
    press = 7,
    nvars = 8
  };

 public:
  StellarProfile(ParameterInput *pin);
  ~StellarProfile();

  Real Eval(int var, Real r) const;

 private:
  int siz;
  Real *pr;
  Real *pvars[nvars];
};

// Deleptonization scheme by astro-ph/0504072, updated fits from 1701.02752
class Deleptonization {
 public:
  Deleptonization(ParameterInput *pin);
  Real Ye_of_rho(Real rho) const;

 private:
  Real log10_rho1;
  Real log10_rho2;
  Real Ye_2;
  Real Ye_c;
  Real Ye_H;
};

Deleptonization *pdelept = nullptr;

int RefinementCondition(MeshBlock *pmb);

}  // namespace

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also
//  be used to initialize variables which are global to (and therefore can be
//  passed to) other functions in this file.  Called in Mesh constructor.
//========================================================================================
void Mesh::InitUserMeshData(ParameterInput *pin) {
  if (adaptive == true) EnrollUserRefinementCondition(RefinementCondition);

  pdelept = new Deleptonization(pin);
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Sets the initial conditions.
//========================================================================================
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // Read stellar profile
  StellarProfile *pstar = new StellarProfile(pin);

  // container with idx / grids pertaining z4c
  MB_info *mbi = &(pz4c->mbi);

  // --------------------------------------------------------------------------
  // Set some aliases for the variables.
  AT_N_sca alpha(pz4c->storage.u, Z4c::I_Z4c_alpha);
  AT_N_vec beta_u(pz4c->storage.u, Z4c::I_Z4c_betax);
  AT_N_sym g_dd(pz4c->storage.adm, Z4c::I_ADM_gxx);
  AT_N_sym K_dd(pz4c->storage.adm, Z4c::I_ADM_Kxx);

  beta_u.ZeroClear();
  g_dd.ZeroClear();
  K_dd.ZeroClear();

  // Interpolate quantities to the grid
  for (int k = 0; k < mbi->nn3; ++k) {
    for (int j = 0; j < mbi->nn2; ++j) {
      for (int i = 0; i < mbi->nn1; ++i) {
        Real const xp = mbi->x1(i);
        Real const yp = mbi->x2(j);
        Real const zp = mbi->x3(k);
        Real const rad = sqrt(xp * xp + yp * yp + zp * zp);

        alpha(k, j, i) = pstar->Eval(StellarProfile::alp, rad);
        Real A = pstar->Eval(StellarProfile::gxx, rad);
        g_dd(2, 2, k, j, i) = g_dd(1, 1, k, j, i) = g_dd(0, 0, k, j, i) = A;

        phydro->w(IDN, k, j, i) = pstar->Eval(StellarProfile::rho, rad);

        Real vr = pstar->Eval(StellarProfile::vel, rad);
        Real vx = vr * xp / rad;
        Real vy = vr * yp / rad;
        Real vz = vr * zp / rad;
        Real W = 1.0 / sqrt(1.0 - A * (vx * vx + vy * vy + vz * vz));
        phydro->w(IVX, k, j, i) = W * vx;
        phydro->w(IVY, k, j, i) = W * vy;
        phydro->w(IVZ, k, j, i) = W * vz;

        phydro->w(IPR, k, j, i) = pstar->Eval(StellarProfile::press, rad);
      }
    }
  }
  // DR: is this needed?
  phydro->w1 = phydro->w;

  pz4c->ADMToZ4c(pz4c->storage.adm, pz4c->storage.u);
  peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord, 0,
                             ncells1, 0, ncells2, 0, ncells3);
  pz4c->GetMatter(pz4c->storage.mat, pz4c->storage.adm, phydro->w, pscalars->r,
                  pfield->bcc);
  pz4c->ADMConstraints(pz4c->storage.con, pz4c->storage.adm, pz4c->storage.mat,
                       pz4c->storage.u);

  // Cleanup
  delete pstar;
}

//========================================================================================
//! \fn void Mesh::UserWorkInLoop()
//  \brief Here we should apply the deleptonization scheme
//========================================================================================
void Mesh::UserWorkInLoop() {
  // TODO: how do we add the deleptonization?
  // Y_e of rho scheme
  // tau_dot = - \alpha \sqrt{\gamma} \rho/mb W \dot{Y_e} E_\nu
}

//========================================================================================
namespace {
//========================================================================================

void skipline(FILE *fptr) {
  char c;
  do {
    c = fgetc(fptr);
  } while (c != '\n');
}

StellarProfile::StellarProfile(ParameterInput *pin) : siz(0) {
  string fname = pin->GetString("problem", "progenitor");

  FILE *starfile = fopen(fname.c_str(), "r");
  if (NULL == starfile) {
    stringstream msg;
    msg << "### FATAL ERROR problem/progenitor: " << string(fname) << " "
        << " could not be accessed.";
    ATHENA_ERROR(msg);
  }

  // Find out how large the profile is
  skipline(starfile);
  skipline(starfile);
  char c;
  while (EOF != (c = getc(starfile))) {
    if (c == '\n') {
      siz++;
    }
  }
  rewind(starfile);
  skipline(starfile);
  skipline(starfile);

  pr = new Real[siz];
  for (int vi = 0; vi < nvars; ++vi) {
    pvars[vi] = new Real[siz];
  }

  // Read the profile
  for (int i = 0; i < siz; ++i) {
    fscanf(starfile, "%lf %lf %lf %lf %lf %lf %lf", &pr[i], &pvars[mass][i],
           &pvars[vel][i], &pvars[rho][i], &pvars[temp][i], &pvars[ye][i],
           &pvars[press][i]);
  }

  // Compute the metric (Newtonian limit)
  for (int i = 0; i < siz; ++i) {
    Real Phi = pvars[mass][i] / pr[i];
    pvars[alp][i] = sqrt(1. + 2. * Phi);
    pvars[gxx][i] = 1. - 2. * Phi;
  }

  // Cleanup
  fclose(starfile);
}

StellarProfile::~StellarProfile() {
  delete[] pr;
  for (int vi = 0; vi < nvars; ++vi) {
    delete[] pvars[vi];
  }
}

Real StellarProfile::Eval(int vi, Real rad) const {
  assert(vi >= 0 && vi < nvars);
  if (rad <= pr[0]) {
    return pvars[vi][0];
  }
  if (rad >= pr[siz - 1]) {
    return pvars[vi][siz - 1];
  }
  int offset = lower_bound(pr, pr + siz, rad) - pr - 1;
  Real lam = (rad - pr[offset]) / (pr[offset + 1] - pr[offset]);
  return pvars[vi][offset] * (1 - lam) + pvars[vi][offset + 1] * lam;
}

Deleptonization::Deleptonization(ParameterInput *pin) {
  // Default values are SFHo from 1701.02752
  log10_rho1 = pin->GetOrAddReal("deleptonization", "log10_rho1", 7.795);
  log10_rho2 = pin->GetOrAddReal("deleptonization", "log10_rho2", 12.816);
  Ye_2 = pin->GetOrAddReal("deleptonization", "Ye_2", 0.308);
  Ye_c = pin->GetOrAddReal("deleptonization", "Ye_c", 0.0412);
  Ye_H = pin->GetOrAddReal("deleptonization", "Ye_H", 0.257);
}

Real Deleptonization::Ye_of_rho(Real rho) const {
  Real const Ye_1 = 0.5;
  Real const log10_rhoH = 15;

  Real const log10_rho = log10(rho * UDENS);
  Real const x = max(-1.0, min(1.0, (2 * log10_rho - log10_rho2 - log10_rho1) /
                                        (log10_rho2 - log10_rho1)));
  Real const m = (Ye_H - Ye_2) / (log10_rhoH - log10_rho2);

  if (log10_rho > log10_rho2) {
    return Ye_2 + m * (log10_rho - log10_rho2);
  } else {
    return 0.5 * (Ye_2 + Ye_1) + 0.5 * x * (Ye_2 - Ye_1) +
           Ye_c * (1 - abs(x) + 4 * abs(x) * (abs(x) - 0.5) * (abs(x) - 1));
  }
}

int RefinementCondition(MeshBlock *pmb) {
  // TODO: implement this
  // We probably want a refinement condition based on the rest mass in
  // each cell
}

}  // namespace