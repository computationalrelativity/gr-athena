#ifndef GR_TASK_LIST_HPP_
#define GR_TASK_LIST_HPP_

// C headers

// C++ headers

// Athena++ headers
#include "../../athena.hpp"
#include "../task_list.hpp"
#include "task_names.hpp"
#include "time_integrators.hpp"

namespace TaskLists::GeneralRelativity {

class GR_Z4c : public TaskList, TaskLists::Integrators::LowStorage {
public:
  GR_Z4c(ParameterInput *pin, Mesh *pm);

  TaskStatus ClearAllBoundary(MeshBlock *pmb, int stage);
  TaskStatus UserWork(MeshBlock *pmb, int stage);
  TaskStatus NewBlockTimeStep(MeshBlock *pmb, int stage);
  TaskStatus CalculateZ4cRHS(MeshBlock *pmb, int stage);
  TaskStatus IntegrateZ4c(MeshBlock *pmb, int stage);
  TaskStatus SendZ4c(MeshBlock *pmb, int stage);
  TaskStatus ReceiveZ4c(MeshBlock *pmb, int stage);
  TaskStatus SetBoundariesZ4c(MeshBlock *pmb, int stage);
  TaskStatus Prolongation(MeshBlock *pmb, int stage);
  TaskStatus PhysicalBoundary(MeshBlock *pmb, int stage);
  TaskStatus EnforceAlgConstr(MeshBlock *pmb, int stage);
  TaskStatus Z4cToADM(MeshBlock *pmb, int stage);
  TaskStatus ADM_Constraints(MeshBlock *pmb, int stage);
  TaskStatus CheckRefinement(MeshBlock *pmb, int stage);
  TaskStatus Z4c_Weyl(MeshBlock *pmb, int stage);
  TaskStatus WaveExtract(MeshBlock *pmb, int stage);
#if CCE_ENABLED
  TaskStatus CCEDump(MeshBlock *pmb, int stage);
#endif
  TaskStatus AssertFinite(MeshBlock *pmb, int stage);

  //---------------------------------------------------------------------------
  // Provide finer-grained control over tasklist
  // Note: If a parameter is zero related task(s) will be ignored
  struct aux_NextTimeStep{
    Real dt{0.};
    Real next_time{0.};
    Real to_update{false};
  };

  struct {
    aux_NextTimeStep adm;
    aux_NextTimeStep con;
    aux_NextTimeStep con_hst;
    aux_NextTimeStep assert_is_finite;
    aux_NextTimeStep wave_extraction;
    aux_NextTimeStep cce_dump;
  } TaskListTriggers;

  bool CurrentTimeCalculationThreshold(Mesh *pm,
                                       aux_NextTimeStep *variable);
  void UpdateTaskListTriggers();
  //---------------------------------------------------------------------------

public:
  using TaskList::nstages;

private:
  // TODO: remove the AddTask logic in favour of Add
  void AddTask(const TaskID& id, const TaskID& dep) override { };
  void StartupTaskList(MeshBlock *pmb, int stage) override;

private:
  // For slightly cleaner & more flexible, treatment of tasklist graph assembly
  void Add(
    const TaskID& id, const TaskID& dep,
    TaskStatus (GR_Z4c::*fcn)(MeshBlock*, int),
    bool lb_time=true)
  {
    TaskList::Add(id, dep, static_cast<TaskStatus (TaskList::*)(
      MeshBlock*, int
    )>(fcn));
  }
};

class PostAMR_Z4c : public TaskList
{
public:
  PostAMR_Z4c(ParameterInput *pin, Mesh *pm);

  TaskStatus ClearAllBoundary(MeshBlock *pmb, int stage);
  TaskStatus EnforceAlgConstr(MeshBlock *pmb, int stage);

  TaskStatus Z4cToADM(MeshBlock *pmb, int stage);
  TaskStatus UpdateSource(MeshBlock *pmb, int stage);
  TaskStatus ADM_Constraints(MeshBlock *pmb, int stage);

  TaskStatus Z4c_Weyl(MeshBlock *pmb, int stage);

private:
  // TODO: remove the AddTask logic in favour of Add
  void AddTask(const TaskID& id, const TaskID& dep) override { };
  void StartupTaskList(MeshBlock *pmb, int stage) override;

private:
  // For slightly cleaner & more flexible, treatment of tasklist graph assembly
  void Add(
    const TaskID& id, const TaskID& dep,
    TaskStatus (PostAMR_Z4c::*fcn)(MeshBlock*, int),
    bool lb_time=true)
  {
    TaskList::Add(id, dep, static_cast<TaskStatus (TaskList::*)(
      MeshBlock*, int
    )>(fcn));
  }
};

}  // namespace TaskLists::GeneralRelativity

#endif  // GR_TASK_LIST_HPP_
