#ifndef TASK_LIST_GR_TIME_INTEGRATORS_HPP_
#define TASK_LIST_GR_TIME_INTEGRATORS_HPP_

// C/C++ headers

// Athena++ classes headers
#include "../athena.hpp"
#include "../athena_aliases.hpp"
#include "../mesh/mesh.hpp"
#include "task_list.hpp"

// ----------------------------------------------------------------------------
using namespace gra::aliases;
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
namespace TaskLists::Integrators {

class LowStorage
{
public:
  LowStorage(ParameterInput *pin, Mesh *pm);

  struct IntegratorWeights
  {
    // 2S or 3S* low-storage RK coefficients, Ketchenson (2010)

    // low-storage coefficients to avoid double F() evaluation per substage
    Real delta;
    // low-storage coeff for weighted ave of registers
    Real gamma_1, gamma_2, gamma_3;
    // coeff. from bidiagonal Shu-Osher form Beta matrix, -1 diagonal terms
    Real beta;
  };

  // Scaled coefficient for RHS time-advance within stage
  Real dt_scaled(const int stage, MeshBlock * pmb)
  {
    return (stage_wghts[(stage-1)].beta)*(pmb->pmy_mesh->dt);
  }

  // Time at beginning of stage for u()
  Real t_begin(const int stage, MeshBlock * pmb)
  {
    return pmb->pmy_mesh->time + pmb->stage_abscissae[stage-1][0];
  }

  // Time at end of the stage
  Real t_end(const int stage, MeshBlock * pmb)
  {
    return pmb->pmy_mesh->time + pmb->stage_abscissae[stage][0];
  }

  // Given overall timestep dt, compute the time abscissae for all
  // registers, stages
  void PrepareStageAbscissae(const int stage, MeshBlock * pmb);

  // data
  std::string integrator;
  // dt stability limit for the particular time integrator + spatial order
  Real cfl_limit;
  int nstages;

public:
  IntegratorWeights stage_wghts[MAX_NSTAGE];
};

class Butcher
{
public:
  Butcher(ParameterInput *pin, Mesh *pm);

  // data
  std::string integrator;
  // dt stability limit for the particular time integrator + spatial order
  Real cfl_limit;
  int nstages;

  // Scaled coefficient for RHS time-advance within stage
  Real dt_scaled(const int stage, MeshBlock * pmb)
  {
    // return (stage_wghts[(stage-1)].beta)*(pmb->pmy_mesh->dt);
    return pmb->pmy_mesh->dt;
  }

  // Time at beginning of stage for u()
  Real t_begin(const int stage, MeshBlock * pmb)
  {
    // return pmb->pmy_mesh->time + pmb->stage_abscissae[stage-1][0];
    return pmb->pmy_mesh->time;
  }

  // Time at end of the stage
  Real t_end(const int stage, MeshBlock * pmb)
  {
    // return pmb->pmy_mesh->time + pmb->stage_abscissae[stage][0];
    return pmb->pmy_mesh->time+pmb->pmy_mesh->dt;
  }

  void PrepareStageScratch(const int stage,
                           MeshBlock * pmb,
                           std::vector<AA> & bt_k,
                           const AthenaArray<Real> & u,
                           AthenaArray<Real> & rhs);

  void PutScratchBT_u(MeshBlock * pmb,
                      std::vector<AA> & bt_k,
                      const AthenaArray<Real> & u);

  void SumBT_ak(MeshBlock * pmb,
                const int stage,
                const Real dt,
                std::vector<AA> & bt_k,
                AthenaArray<Real> & u_out);

  void SumBT_bk(MeshBlock * pmb,
                const Real dt,
                std::vector<AA> & bt_k,
                AthenaArray<Real> & u_out);

public:
  // BT style integrator weights
  std::vector<std::vector<Real>> bt_a;
  std::vector<Real> bt_b, bt_c;
};

class integrators
{
public:
  integrators(ParameterInput *pin, Mesh *pm)
  {
    integrator = pin->GetOrAddString("time", "integrator", "vl2");
    is_lowstorage = !(integrator.find("bt_") != std::string::npos);

    if (is_lowstorage)
    {
      ls = new LowStorage(pin, pm);
      nstages = ls->nstages;
    }
    else
    {
      bt = new Butcher(pin, pm);
      nstages = bt->nstages;
    }
  }

  ~integrators()
  {
    if (is_lowstorage)
    {
      delete ls;
    }
    else
    {
      delete bt;
    }
  }

  // Scaled coefficient for RHS time-advance within stage
  Real dt_scaled(const int stage, MeshBlock * pmb)
  {
    if (is_lowstorage)
    {
      return ls->dt_scaled(stage, pmb);
    }
    else
    {
      return bt->dt_scaled(stage, pmb);
    }
  }

  // Time at beginning of stage for u()
  Real t_begin(const int stage, MeshBlock * pmb)
  {
    if (is_lowstorage)
    {
      return ls->t_begin(stage, pmb);
    }
    else
    {
      return bt->t_begin(stage, pmb);
    }
  }

  // Time at end of the stage
  Real t_end(const int stage, MeshBlock * pmb)
  {
    if (is_lowstorage)
    {
      return ls->t_end(stage, pmb);
    }
    else
    {
      return bt->t_end(stage, pmb);
    }
  }

  void Initialize(const int stage,
                  MeshBlock * pmb,
                  std::vector<AA> & bt_k,
                  const AthenaArray<Real> & u,
                  AthenaArray<Real> & rhs)
  {
    if (is_lowstorage)
    {
      if (stage == 1)
      {
        ls->PrepareStageAbscissae(stage, pmb);
      }
    }
    else
    {
      bt->PrepareStageScratch(stage, pmb, bt_k, u, rhs);
    }
  }

public:
  std::string integrator;

  bool is_lowstorage;
  int nstages;

  LowStorage * ls;
  Butcher * bt;
};

}  // TaskLists::Integrators

#endif  // TASK_LIST_GR_TIME_INTEGRATORS_HPP_

//
// :D
//
