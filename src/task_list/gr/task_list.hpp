#ifndef GR_TASK_LIST_HPP_
#define GR_TASK_LIST_HPP_

// C headers

// C++ headers

// Athena++ headers
#include "../../athena.hpp"
#include "../task_list.hpp"
#include "task_names.hpp"

namespace TaskLists::GeneralRelativity {

class GR_Z4c : public TaskList {
public:
  GR_Z4c(ParameterInput *pin, Mesh *pm);

  //--------------------------------------------------------------------------------------
  //! \struct IntegratorWeight
  //  \brief weights used in time integrator tasks

  struct IntegratorWeight {
    // 2S or 3S* low-storage RK coefficients, Ketchenson (2010)
    Real delta; // low-storage coefficients to avoid double F() evaluation per substage
    Real gamma_1, gamma_2, gamma_3; // low-storage coeff for weighted ave of registers
    Real beta; // coeff. from bidiagonal Shu-Osher form Beta matrix, -1 diagonal terms
  };

  // data
  std::string integrator;
  Real cfl_limit; // dt stability limit for the particular time integrator + spatial order

  // functions
  TaskStatus ClearAllBoundary(MeshBlock *pmb, int stage);  // CLEAR_ALLBND [x]
  TaskStatus UserWork(MeshBlock *pmb, int stage);          // USERWORK     [x]
  TaskStatus NewBlockTimeStep(MeshBlock *pmb, int stage);  // NEW_DT       [x]
  TaskStatus CalculateZ4cRHS(MeshBlock *pmb, int stage);   // CALC_Z4CRHS  [x]
  TaskStatus IntegrateZ4c(MeshBlock *pmb, int stage);      // INT_Z4C      [x]
  TaskStatus SendZ4c(MeshBlock *pmb, int stage);           // SEND_Z4C     [x]
  TaskStatus ReceiveZ4c(MeshBlock *pmb, int stage);        // RECV_Z4C     [x]
  TaskStatus SetBoundariesZ4c(MeshBlock *pmb, int stage);  // SETB_Z4C     [x]
  TaskStatus Prolongation(MeshBlock *pmb, int stage);      // PROLONG      [x]
  TaskStatus PhysicalBoundary(MeshBlock *pmb, int stage);  // PHY_BVAL     [x]
  TaskStatus EnforceAlgConstr(MeshBlock *pmb, int stage);  // ALG_CONSTR   [x]
  TaskStatus Z4cToADM(MeshBlock *pmb, int stage);          // Z4C_TO_ADM   [x]
  TaskStatus ADM_Constraints(MeshBlock *pmb, int stage);   // ADM_CONSTR   [x]
  TaskStatus CheckRefinement(MeshBlock *pmb, int stage);   // FLAG_AMR     [x]
  TaskStatus Z4c_Weyl(MeshBlock *pmb, int stage);          // Z4C_WEYL     [x]
  TaskStatus WaveExtract(MeshBlock *pmb, int stage);       // WAVE_EXTR    [x]
#if CCE_ENABLED
  TaskStatus CCEDump(MeshBlock *pmb, int stage);           // CCE_DUMP     [x]
#endif
  TaskStatus AssertFinite(MeshBlock *pmb, int stage);      // ASSERT_FIN   [x]

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


private:
  IntegratorWeight stage_wghts[MAX_NSTAGE];

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
  // TODO: remove the AddTask logic in favour of Add
  void AddTask(const TaskID& id, const TaskID& dep) override { };
  void StartupTaskList(MeshBlock *pmb, int stage) override;

private:
  // For slightly cleaner & more flexible, treatment of tasklist graph assembly
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
