#ifndef TASK_LIST_GR_TIME_INTEGRATORS_HPP_
#define TASK_LIST_GR_TIME_INTEGRATORS_HPP_

// C/C++ headers

// Athena++ classes headers
#include "../../athena.hpp"
#include "../../mesh/mesh.hpp"
#include "task_list.hpp"


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

}  // TaskLists::Integrators

#endif  // TASK_LIST_GR_TIME_INTEGRATORS_HPP_

//
// :D
//
