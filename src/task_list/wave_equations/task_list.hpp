#ifndef WAVE_EQUATIONS_TASK_LIST_HPP_
#define WAVE_EQUATIONS_TASK_LIST_HPP_

// C headers

// C++ headers

// Athena++ headers
#include "../../athena.hpp"
#include "../../main_triggers.hpp"
#include "../task_list.hpp"
#include "task_names.hpp"
#include "../time_integrators.hpp"

namespace TaskLists::WaveEquations {

// homogeneous wave equation, second order
class Wave_2O : public TaskList, TaskLists::Integrators::LowStorage
{
public:
  Wave_2O(ParameterInput *pin, Mesh *pm, gra::triggers::Triggers &trgs);

  TaskStatus ClearAllBoundary(MeshBlock *pmb, int stage);
  TaskStatus UserWork(MeshBlock *pmb, int stage);
  TaskStatus NewBlockTimeStep(MeshBlock *pmb, int stage);
  TaskStatus CheckRefinement(MeshBlock *pmb, int stage);
  TaskStatus CalculateWaveRHS(MeshBlock *pmb, int stage);
  TaskStatus IntegrateWave(MeshBlock *pmb, int stage);
  TaskStatus SendWave(MeshBlock *pmb, int stage);
  TaskStatus ReceiveWave(MeshBlock *pmb, int stage);

  TaskStatus SetBoundariesWave(MeshBlock *pmb, int stage);

  TaskStatus Prolongation(MeshBlock *pmb, int stage);
  TaskStatus PhysicalBoundary(MeshBlock *pmb, int stage);

public:
  using TaskList::nstages;

private:
  // BD - TODO: remove the AddTask logic in favour of Add
  void AddTask(const TaskID& id, const TaskID& dep) override { };
  void StartupTaskList(MeshBlock *pmb, int stage) override;

private:
  gra::triggers::Triggers & trgs;

private:
  // For slightly cleaner & more flexible, treatment of tasklist graph assembly
  void Add(
    const TaskID& id, const TaskID& dep,
    TaskStatus (Wave_2O::*fcn)(MeshBlock*, int),
    bool lb_time=true)
  {
    TaskList::Add(id, dep, static_cast<TaskStatus (TaskList::*)(
      MeshBlock*, int
    )>(fcn));
  }
};

}  // namespace TaskLists::WaveEquations

#endif  // WAVE_EQUATIONS_TASK_LIST_HPP_
