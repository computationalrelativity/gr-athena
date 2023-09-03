#ifndef TASK_LIST_M1_TASK_LIST_HPP_
#define TASK_LIST_M1_TASK_LIST_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file m1_task_list.hpp
//! \brief M1 task list

#include "task_list.hpp"

class M1IntegratorTaskList: public TaskList {
  public:
    Real cfl_limit = 1.0;
    int const nstages = 2;

    M1IntegratorTaskList(ParameterInput *pin, Mesh *pm);

    // functions
    TaskStatus ClearAllBoundary(MeshBlock *pmb, int stage);

    TaskStatus CalcFiducialVelocity(MeshBlock *pmb, int stage);
    TaskStatus CalcClosure(MeshBlock *pmb, int stage);
    TaskStatus CalcOpacity(MeshBlock *pmb, int stage);

    TaskStatus CalcFlux(MeshBlock *pmb, int stage);
    TaskStatus SendFlux(MeshBlock *pmb, int stage);
    TaskStatus ReceiveAndCorrectFlux(MeshBlock *pmb, int stage);

    TaskStatus CalcGRSources(MeshBlock *pmb, int stage);
    TaskStatus AddFluxDivergence(MeshBlock *pmb, int stage);

    TaskStatus CalcUpdate(MeshBlock *pmb, int stage);

    TaskStatus Send(MeshBlock *pmb, int stage);
    TaskStatus Receive(MeshBlock *pmb, int stage);

    TaskStatus SetBoundaries(MeshBlock *pmb, int stage);

    TaskStatus Prolongation(MeshBlock *pmb, int stage);
    TaskStatus PhysicalBoundary(MeshBlock *pmb, int stage);

    TaskStatus UserWork(MeshBlock *pmb, int stage);

    TaskStatus NewBlockTimeStep(MeshBlock *pmb, int stage);

    TaskStatus CheckRefinement(MeshBlock *pmb, int stage);

  private:
    Real const dt_fac[2]{0.5, 1.0}; // timestep factor for each stage
    void AddTask(const TaskID& id, const TaskID& dep) override;
    void StartupTaskList(MeshBlock *pmb, int stage) override;
};

//----------------------------------------------------------------------------------------
// 64-bit integers with "1" in different bit positions used to ID each z4c task.
namespace M1IntegratorTaskNames {
  const TaskID NONE(0);
  const TaskID CLEAR_ALLBND(1);
  const TaskID CALC_FIDU(2);
  const TaskID CALC_CLOSURE(3);
  const TaskID CALC_OPAC(4);
  const TaskID CALC_FLUX(5);
  const TaskID SEND_FLUX(6);
  const TaskID RECV_FLUX(7);
  const TaskID ADD_FLX_DIV(8);
  const TaskID CALC_GRSRC(9);
  const TaskID CALC_UPDATE(10);
  const TaskID SEND(11);
  const TaskID RECV(12);
  const TaskID SETB(13);
  const TaskID PROLONG(14);
  const TaskID PHY_BVAL(15);
  const TaskID USERWORK(16);
  const TaskID NEW_DT(17);
  const TaskID FLAG_AMR(18);
}  // namespace M1IntegratorTaskNames

#endif // TASK_LIST_M1_TASK_LIST_HPP_