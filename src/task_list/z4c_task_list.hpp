#ifndef Z4c_TASK_LIST_HPP_
#define Z4c_TASK_LIST_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//!   \file z4c_task_list.hpp
//    \brief task list for Z4c

#include "../athena.hpp"
#include "task_list.hpp"

class Z4cIntegratorTaskList: public TaskList {
public:
  Z4cIntegratorTaskList(ParameterInput *pin, Mesh *pm);
  ~Z4cIntegratorTaskList() {}

  // data
  std::string integrator;
  Real cfl_limit; // dt stability limit for the particular time integrator + spatial order
  struct IntegratorWeight stage_wghts[MAX_NSTAGE];

  void AddZ4cIntegratorTask(uint64_t id, uint64_t dep);

  // functions
  enum TaskStatus StartAllReceive(MeshBlock *pmb, int stage);
  enum TaskStatus ClearAllBoundary(MeshBlock *pmb, int stage);

  enum TaskStatus CalculateZ4cRHS(MeshBlock *pmb, int stage);
  enum TaskStatus Z4cIntegrate(MeshBlock *pmb, int stage);

  enum TaskStatus Z4cSend(MeshBlock *pmb, int stage);
  enum TaskStatus Z4cReceive(MeshBlock *pmb, int stage);

  enum TaskStatus Prolongation(MeshBlock *pmb, int stage);
  enum TaskStatus PhysicalBoundary(MeshBlock *pmb, int stage);
  enum TaskStatus EnforceAlgConstr(MeshBlock *pmb, int stage);
  enum TaskStatus Z4cToADM(MeshBlock *pmb, int stage);
  enum TaskStatus UserWork(MeshBlock *pmb, int stage);
  enum TaskStatus NewBlockTimeStep(MeshBlock *pmb, int stage);
  enum TaskStatus CheckRefinement(MeshBlock *pmb, int stage);

  enum TaskStatus StartupIntegrator(MeshBlock *pmb, int stage);
};

//----------------------------------------------------------------------------------------
// 64-bit integers with "1" in different bit positions used to ID each wave task.

namespace Z4cIntegratorTaskNames {
  const uint64_t NONE=0;
  const uint64_t START_ALLRECV=1LL<<0;
  const uint64_t CLEAR_ALLBND=1LL<<1;
  const uint64_t CALC_Z4CRHS=1LL<<2;
  const uint64_t INT_Z4C=1LL<<3;
  const uint64_t SEND_Z4C=1LL<<4;
  const uint64_t RECV_Z4C=1LL<<5;
  const uint64_t PROLONG =1LL<<6;
  const uint64_t CON2PRIM=1LL<<7;
  const uint64_t PHY_BVAL=1LL<<8;
  const uint64_t ALG_CONSTR=1LL<<9;
  const uint64_t Z4C_TO_ADM=1LL<<10;
  const uint64_t USERWORK=1LL<<11;
  const uint64_t NEW_DT  =1LL<<12;
  const uint64_t AMR_FLAG=1LL<<13;
  const uint64_t STARTUP_INT=1LL<<14;
} // namespace Z4cIntegratorTaskNames

#endif
