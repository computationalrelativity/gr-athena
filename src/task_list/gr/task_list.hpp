#ifndef GR_TASK_LIST_HPP_
#define GR_TASK_LIST_HPP_

// C headers

// C++ headers

// Athena++ headers
#include "../../athena.hpp"
#include "../../main_triggers.hpp"
#include "../task_list.hpp"
#include "task_names.hpp"
#include "../time_integrators.hpp"

namespace TaskLists::GeneralRelativity {

// Vacuum system tasklist
class GR_Z4c : public TaskList,
                      TaskLists::Integrators::integrators
{
public:
  GR_Z4c(ParameterInput *pin, Mesh *pm, gra::triggers::Triggers &trgs);

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
#if CCE_ENABLED
  TaskStatus CCEDump(MeshBlock *pmb, int stage);
#endif
  // BD: TODO- fix this
  TaskStatus AssertFinite(MeshBlock *pmb, int stage);

public:
  using TaskList::nstages;

private:
  // TODO: remove the AddTask logic in favour of Add
  void AddTask(const TaskID& id, const TaskID& dep) override { };
  void StartupTaskList(MeshBlock *pmb, int stage) override;

private:
  gra::triggers::Triggers & trgs;

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

// Coupled GRMHD system tasklist
class GRMHD_Z4c : public TaskList, TaskLists::Integrators::LowStorage
{
public:
  GRMHD_Z4c(ParameterInput *pin, Mesh *pm, gra::triggers::Triggers &trgs);

  TaskStatus ClearAllBoundary(MeshBlock *pmb, int stage);

  TaskStatus CalculateHydroFlux(MeshBlock *pmb, int stage);
  TaskStatus CalculateEMF(MeshBlock *pmb, int stage);

  TaskStatus SendFluxCorrectionHydro(MeshBlock *pmb, int stage);
  TaskStatus SendFluxCorrectionEMF(  MeshBlock *pmb, int stage);

  TaskStatus ReceiveAndCorrectHydroFlux(MeshBlock *pmb, int stage);
  TaskStatus ReceiveAndCorrectEMF(MeshBlock *pmb, int stage);

  TaskStatus IntegrateHydro(MeshBlock *pmb, int stage);
  TaskStatus IntegrateField(MeshBlock *pmb, int stage);

  TaskStatus AddSourceTermsHydro(MeshBlock *pmb, int stage);

  TaskStatus SendHydro(MeshBlock *pmb, int stage);
  TaskStatus SendField(MeshBlock *pmb, int stage);

  TaskStatus ReceiveHydro(MeshBlock *pmb, int stage);
  TaskStatus ReceiveField(MeshBlock *pmb, int stage);

  TaskStatus SetBoundariesHydro(MeshBlock *pmb, int stage);
  TaskStatus SetBoundariesField(MeshBlock *pmb, int stage);

  TaskStatus Prolongation_Hyd(MeshBlock *pmb, int stage);
  TaskStatus Primitives(MeshBlock *pmb, int stage);
  TaskStatus PhysicalBoundary_Hyd(MeshBlock *pmb, int stage);
  TaskStatus UserWork(MeshBlock *pmb, int stage);
  TaskStatus NewBlockTimeStep(MeshBlock *pmb, int stage);
  TaskStatus CheckRefinement(MeshBlock *pmb, int stage);

  TaskStatus CalculateScalarFlux(MeshBlock *pmb, int stage);
  TaskStatus SendScalarFlux(MeshBlock *pmb, int stage);
  TaskStatus ReceiveScalarFlux(MeshBlock *pmb, int stage);
  TaskStatus IntegrateScalars(MeshBlock *pmb, int stage);
  TaskStatus SendScalars(MeshBlock *pmb, int stage);
  TaskStatus ReceiveScalars(MeshBlock *pmb, int stage);
  TaskStatus SetBoundariesScalars(MeshBlock *pmb, int stage);

  TaskStatus CalculateZ4cRHS(MeshBlock *pmb, int stage);
  TaskStatus IntegrateZ4c(MeshBlock *pmb, int stage);
  TaskStatus SendZ4c(MeshBlock *pmb, int stage);
  TaskStatus ReceiveZ4c(MeshBlock *pmb, int stage);
  TaskStatus SetBoundariesZ4c(MeshBlock *pmb, int stage);
  TaskStatus Prolongation_Z4c(MeshBlock *pmb, int stage);
  TaskStatus PhysicalBoundary_Z4c(MeshBlock *pmb, int stage);
  TaskStatus EnforceAlgConstr(MeshBlock *pmb, int stage);
  TaskStatus Z4cToADM(MeshBlock *pmb, int stage);
  TaskStatus ADM_Constraints(MeshBlock *pmb, int stage);
  TaskStatus Z4c_Weyl(MeshBlock *pmb, int stage);
#if CCE_ENABLED
  TaskStatus CCEDump(MeshBlock *pmb, int stage);
#endif
  TaskStatus UpdateSource(MeshBlock *pmb, int stage);

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
    TaskStatus (GRMHD_Z4c::*fcn)(MeshBlock*, int),
    bool lb_time=true)
  {
    TaskList::Add(id, dep, static_cast<TaskStatus (TaskList::*)(
      MeshBlock*, int
    )>(fcn));
  }
};

// Split GRMHD system tasklist [MHD part]
class GRMHD_Z4c_Phase_MHD : public TaskList,
                                   TaskLists::Integrators::LowStorage
{
public:
  GRMHD_Z4c_Phase_MHD(ParameterInput *pin,
                      Mesh *pm,
                      gra::triggers::Triggers &trgs);

  TaskStatus ClearAllBoundary(MeshBlock *pmb, int stage);

  TaskStatus CalculateEMF(MeshBlock *pmb, int stage);

  TaskStatus SendFluxCorrectionHydro(MeshBlock *pmb, int stage);
  TaskStatus SendFluxCorrectionEMF(  MeshBlock *pmb, int stage);
  TaskStatus SendScalarFlux(MeshBlock *pmb, int stage);

  TaskStatus ReceiveAndCorrectHydroFlux(MeshBlock *pmb, int stage);
  TaskStatus ReceiveAndCorrectEMF(MeshBlock *pmb, int stage);
  TaskStatus ReceiveScalarFlux(MeshBlock *pmb, int stage);

  TaskStatus CalculateHydroScalarFlux(MeshBlock *pmb, int stage);
  TaskStatus IntegrateHydroScalars(MeshBlock *pmb, int stage);
  TaskStatus AddFluxDivergenceHydroScalars(MeshBlock *pmb, int stage);

  TaskStatus IntegrateField(MeshBlock *pmb, int stage);

  TaskStatus AddSourceTermsHydro(MeshBlock *pmb, int stage);

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
    TaskStatus (GRMHD_Z4c_Phase_MHD::*fcn)(MeshBlock*, int),
    bool lb_time=true)
  {
    TaskList::Add(id, dep, static_cast<TaskStatus (TaskList::*)(
      MeshBlock*, int
    )>(fcn));
  }
};

// Split GRMHD system tasklist [MHD part]
class GRMHD_Z4c_Phase_MHD_com : public TaskList,
                                   TaskLists::Integrators::LowStorage
{
public:
  GRMHD_Z4c_Phase_MHD_com(ParameterInput *pin,
                      Mesh *pm,
                      gra::triggers::Triggers &trgs);

  TaskStatus ClearAllBoundary(MeshBlock *pmb, int stage);

  TaskStatus PrimitivesPhysical(MeshBlock *pmb, int stage);

  TaskStatus SendHydro(MeshBlock *pmb, int stage);
  TaskStatus SendField(MeshBlock *pmb, int stage);
  TaskStatus SendScalars(MeshBlock *pmb, int stage);

  TaskStatus ReceiveHydro(MeshBlock *pmb, int stage);
  TaskStatus ReceiveField(MeshBlock *pmb, int stage);
  TaskStatus ReceiveScalars(MeshBlock *pmb, int stage);

  TaskStatus SetBoundariesHydro(MeshBlock *pmb, int stage);
  TaskStatus SetBoundariesField(MeshBlock *pmb, int stage);
  TaskStatus SetBoundariesScalars(MeshBlock *pmb, int stage);

  TaskStatus Prolongation_Hyd(MeshBlock *pmb, int stage);
  TaskStatus PhysicalBoundary_Hyd(MeshBlock *pmb, int stage);

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
    TaskStatus (GRMHD_Z4c_Phase_MHD_com::*fcn)(MeshBlock*, int),
    bool lb_time=true)
  {
    TaskList::Add(id, dep, static_cast<TaskStatus (TaskList::*)(
      MeshBlock*, int
    )>(fcn));
  }
};

// Split GRMHD system tasklist [Z4c part]
class GRMHD_Z4c_Phase_Z4c : public TaskList,
                                   TaskLists::Integrators::LowStorage
{
public:
  GRMHD_Z4c_Phase_Z4c(ParameterInput *pin,
                      Mesh *pm,
                      gra::triggers::Triggers &trgs);

  TaskStatus ClearAllBoundary(MeshBlock *pmb, int stage);

  TaskStatus CalculateZ4cRHS(MeshBlock *pmb, int stage);
  TaskStatus IntegrateZ4c(MeshBlock *pmb, int stage);
  TaskStatus SendZ4c(MeshBlock *pmb, int stage);
  TaskStatus ReceiveZ4c(MeshBlock *pmb, int stage);
  TaskStatus SetBoundariesZ4c(MeshBlock *pmb, int stage);
  TaskStatus Prolongation_Z4c(MeshBlock *pmb, int stage);
  TaskStatus PhysicalBoundary_Z4c(MeshBlock *pmb, int stage);
  TaskStatus EnforceAlgConstr(MeshBlock *pmb, int stage);
  TaskStatus Z4cToADM(MeshBlock *pmb, int stage);
#if CCE_ENABLED
  TaskStatus CCEDump(MeshBlock *pmb, int stage);
#endif

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
    TaskStatus (GRMHD_Z4c_Phase_Z4c::*fcn)(MeshBlock*, int),
    bool lb_time=true)
  {
    TaskList::Add(id, dep, static_cast<TaskStatus (TaskList::*)(
      MeshBlock*, int
    )>(fcn));
  }
};

// Split GRMHD system tasklist [Finalization]
class GRMHD_Z4c_Phase_Finalize : public TaskList,
                                        TaskLists::Integrators::LowStorage
{
public:
  GRMHD_Z4c_Phase_Finalize(ParameterInput *pin,
                           Mesh *pm,
                           gra::triggers::Triggers &trgs);

  TaskStatus PrimitivesGhosts(MeshBlock *pmb, int stage);
  TaskStatus UpdateSource(MeshBlock *pmb, int stage);

  TaskStatus ADM_Constraints(MeshBlock *pmb, int stage);
  TaskStatus Z4c_Weyl(MeshBlock *pmb, int stage);

  TaskStatus UserWork(MeshBlock *pmb, int stage);
  TaskStatus NewBlockTimeStep(MeshBlock *pmb, int stage);
  TaskStatus CheckRefinement(MeshBlock *pmb, int stage);

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
    TaskStatus (GRMHD_Z4c_Phase_Finalize::*fcn)(MeshBlock*, int),
    bool lb_time=true)
  {
    TaskList::Add(id, dep, static_cast<TaskStatus (TaskList::*)(
      MeshBlock*, int
    )>(fcn));
  }
};

// Auxiliary functions (post Vac./GRMHD exec.)
class Aux_Z4c : public TaskList
{
public:
  Aux_Z4c(ParameterInput *pin, Mesh *pm, gra::triggers::Triggers &trgs);

  TaskStatus WeylDecompose(MeshBlock *pmb, int stage);

  // Time at end of the stage
  Real t_end(const int stage, MeshBlock * pmb)
  {
    return pmb->pmy_mesh->time;
  }

public:
  int nstages;

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
    TaskStatus (Aux_Z4c::*fcn)(MeshBlock*, int),
    bool lb_time=true)
  {
    TaskList::Add(id, dep, static_cast<TaskStatus (TaskList::*)(
      MeshBlock*, int
    )>(fcn));
  }

};

// If using AMR, additional functionality is needed
class PostAMR_Z4c : public TaskList
{
public:
  PostAMR_Z4c(ParameterInput *pin, Mesh *pm, gra::triggers::Triggers &trgs);

  TaskStatus ClearAllBoundary(MeshBlock *pmb, int stage);

  TaskStatus ADM_Constraints(MeshBlock *pmb, int stage);
  TaskStatus Z4c_Weyl(MeshBlock *pmb, int stage);

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
