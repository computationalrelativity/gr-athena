#ifndef GR_TASK_LIST_HPP_
#define GR_TASK_LIST_HPP_

// C headers

// C++ headers

// Athena++ headers
#include "../../athena.hpp"
#include "../task_list.hpp"
#include "task_names.hpp"

namespace TaskLists::GeneralRelativity {

class PostAMR : public TaskList
{
public:
  PostAMR(ParameterInput *pin, Mesh *pm);

  TaskStatus ClearAllBoundary(MeshBlock *pmb, int stage);
  TaskStatus EnforceAlgConstr(MeshBlock *pmb, int stage);

  TaskStatus Z4cToADM(MeshBlock *pmb, int stage);
  TaskStatus UpdateSource(MeshBlock *pmb, int stage);
  TaskStatus ADM_Constraints(MeshBlock *pmb, int stage);

  TaskStatus Z4c_Weyl(MeshBlock *pmb, int stage);

private:
  void AddTask(const TaskID& id, const TaskID& dep) override { };
  void StartupTaskList(MeshBlock *pmb, int stage) override;

private:
  void Add(
    const TaskID& id, const TaskID& dep,
    TaskStatus (PostAMR::*fcn)(MeshBlock*, int),
    bool lb_time=true)
  {
    TaskList::Add(id, dep, static_cast<TaskStatus (TaskList::*)(
      MeshBlock*, int
    )>(fcn));
  }
};

}  // namespace TaskLists::GeneralRelativity

#endif  // GR_TASK_LIST_HPP_
