#ifndef M1_TASK_LIST_HPP_
#define M1_TASK_LIST_HPP_

// C headers

// C++ headers

// Athena++ headers
#include "../../athena.hpp"
#include "../../main_triggers.hpp"
#include "../task_list.hpp"
#include "task_names.hpp"
#include "../time_integrators.hpp"

namespace TaskLists::M1 {

// homogeneous wave equation, second order
class M1N0 : public TaskList, TaskLists::Integrators::LowStorage
{
public:
  M1N0(ParameterInput *pin, Mesh *pm, gra::triggers::Triggers &trgs);

  TaskStatus ClearAllBoundary(MeshBlock *pmb, int stage);

  TaskStatus UpdateBackground(MeshBlock *pmb, int stage);

  TaskStatus CalcFiducialVelocity(MeshBlock *pmb, int stage);
  TaskStatus CalcClosure(MeshBlock *pmb, int stage);
  TaskStatus CalcFiducialFrame(MeshBlock *pmb, int stage);
  TaskStatus CalcOpacity(MeshBlock *pmb, int stage);

  TaskStatus CalcFlux(MeshBlock *pmb, int stage);
  TaskStatus SendFlux(MeshBlock *pmb, int stage);
  TaskStatus ReceiveAndCorrectFlux(MeshBlock *pmb, int stage);

  TaskStatus AddGRSources(MeshBlock *pmb, int stage);
  TaskStatus AddFluxDivergence(MeshBlock *pmb, int stage);
  TaskStatus CalcUpdate(MeshBlock *pmb, int stage);
  TaskStatus UpdateCoupling(MeshBlock *pmb, int stage);

  TaskStatus Send(MeshBlock *pmb, int stage);
  TaskStatus Receive(MeshBlock *pmb, int stage);

  TaskStatus SetBoundaries(MeshBlock *pmb, int stage);

  TaskStatus Prolongation(MeshBlock *pmb, int stage);
  TaskStatus PhysicalBoundary(MeshBlock *pmb, int stage);

  TaskStatus UserWork(MeshBlock *pmb, int stage);

  TaskStatus NewBlockTimeStep(MeshBlock *pmb, int stage);
  TaskStatus CheckRefinement(MeshBlock *pmb, int stage);
public:
  using TaskList::nstages;

private:
  // BD - TODO: remove the AddTask logic in favour of Add
  void AddTask(const TaskID& id, const TaskID& dep) override { };
  void StartupTaskList(MeshBlock *pmb, int stage) override;

  // Particular to M1N0
  Real const dt_fac[2]{0.5, 1.0}; // timestep factor for each stage

private:
  gra::triggers::Triggers & trgs;

private:
  // For slightly cleaner & more flexible, treatment of tasklist graph assembly
  void Add(
    const TaskID& id, const TaskID& dep,
    TaskStatus (M1N0::*fcn)(MeshBlock*, int),
    bool lb_time=true)
  {
    TaskList::Add(id, dep, static_cast<TaskStatus (TaskList::*)(
      MeshBlock*, int
    )>(fcn));
  }
};

}  // namespace TaskLists::M1N0

#endif  // M1_TASK_LIST_HPP_
