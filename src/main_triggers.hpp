#ifndef MAIN_TRIGGERS_HPP
#define MAIN_TRIGGERS_HPP

// C headers
#include <cassert>

// C++ headers
#include <functional>
#include <iostream>
#include <map>
#include <unordered_map>

// External libraries

// Athena++ headers
#include "athena.hpp"
#include "defs.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"

#include "outputs/outputs.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "utils/utils.hpp"

// triggers external to task-lists used in main -------------------------------
namespace gra::triggers {

struct Trigger
{
  Real dt = 0;  // step (trigger only activates if dt > 0)
  Real t_last;  // time of last activation
  Real t_next;  // next time
  bool allow_rescale_dt = true;
  bool force_first_iter = false;

  Mesh * pm;

  inline bool is_satisfied()
  {

    if (dt > 0)
    {
      if (force_first_iter)
      {
        return true;
      }

      const Real t_end_step = pm->time+pm->dt;

      // catch the last point(s) reducing trigger dt
      if (((pm->ncycle >= pm->nlim) &&
           (pm->nlim > 0)) ||
          (t_next > pm->tlim))
      {
        dt = pm->dt;
        return true;
      }

      return (t_end_step >= t_next)  ||
             (pm->time   >= pm->tlim);
    }
    return false;
  }

  inline void update()
  {
    t_last = t_next;

    if (force_first_iter)
    {
      // Note use of t_last here
      const int ndumps = (static_cast<int>(t_last / dt));
      t_next = (ndumps+1) * dt;
    }
    else
    {
      t_next = t_last+dt;
    }

    force_first_iter = false;
  }

  // if next step takes us past trigger, reduce mesh step
  inline bool reduce_mesh_dt()
  {
    // if (force_first_iter)
    // {
    //   return false;
    // }

    const Real t_end_step = pm->time+pm->dt;

    // BD: TODO - better way to handle this?
    // FP ops can cause miniscule steps to be set if t_next and pm->time
    // approximately coincide.
    const bool diff_tol = std::abs(1-t_next / pm->time) > 1e-13;
    // const bool diff_tol = true;

    if ((t_end_step > t_next) &&
        (pm->time < t_next)   &&
        diff_tol)
    {
      const Real dt_0 = pm->dt;
      const Real t_next_0 = t_next;

      pm->dt = t_next-pm->time;
      t_next = pm->time + pm->dt;
      t_last = t_next - dt;

      // debug ----------------------------------------------------------------
      // pm->dt = pm->dt - (t_end_step - t_next);

      // #pragma omp critical
      // if (pm->dt < 1e-12)
      // {
      //   std::cout << "pm->dt:               " << pm->dt << std::endl;
      //   std::cout << "dt_0:                 " << dt_0 << std::endl;
      //   std::cout << "pm->time:             " << pm->time << std::endl;
      //   std::cout << "t_end_step:           " << t_end_step << std::endl;
      //   std::cout << "t_next:               " << t_next << std::endl;
      //   std::cout << "t_next_0:             " << t_next_0 << std::endl;
      //   std::cout << "t_last:               " << t_last << std::endl;
      //   std::cout << "dt:                   " << dt << std::endl;
      //   std::cout << "|1-dt/dt_0|:          "
      //             << std::abs(1 - dt / dt_0) << std::endl;
      //   std::cout << "|1-t_next/pm->time|:  "
      //             << std::abs(1 - t_next / pm->time) << std::endl;
      //   std::cout << force_first_iter << std::endl;
      //   std::exit(0);
      // }
      // ----------------------------------------------------------------------

      return true;
    }
    return false;
  }
};

class Triggers
{
public:
  enum class TriggerVariant {
    tracker_extrema,
    Z4c_ADM_constraints,
    Z4c_tracker_punctures,
    Z4c_Weyl,
  };

  enum class OutputVariant {
    user,  // for e.g. tracker scalars
    rst,   // restarts
    hst,   // history file
    data   // general output / dumps
  };

public:
  Triggers(Mesh *pm, ParameterInput *pin, Outputs *pouts)
    : pm(pm), pin(pin), pouts(pouts)
  { };

// ----------------------------------------------------------------------------
// For unordered_map: use map instead to avoid hasher, though, it would be
// slower ...
public:
  typedef std::tuple<TriggerVariant, OutputVariant> TriggerMeta;

private:
  class tm_hash
  {
    public:
      size_t operator()(const TriggerMeta& tm) const
      {
        // hash is XOR of standard hash impl. on enum
        return (std::hash<TriggerVariant>()(std::get<0>(tm)) ^
                std::hash<OutputVariant >()(std::get<1>(tm)));
      }
  };

public:
  // std::unordered_map<TriggerVariant, Trigger> triggers;
  std::unordered_map<TriggerMeta, Trigger, tm_hash> triggers;
// ----------------------------------------------------------------------------

  inline static TriggerMeta MakeTriggerMeta(TriggerVariant tv,
                                            OutputVariant ov)
  {
    return {tv, ov};
  }

public:
  void Add(TriggerVariant tvar,
           OutputVariant ovar,
           const bool force_first_iter,
           const bool allow_rescale_dt)
  {
    Trigger tri;

    // populate trigger logic is generic, some triggers have additional
    // constraints
    switch (tvar)
    {
      case TriggerVariant::tracker_extrema:
      {
        Real dt = 0;
        switch (ovar)
        {
          case (Triggers::OutputVariant::user):
          {
            dt = pin->GetOrAddReal("task_triggers", "tracker_extrema", 0.0);
            break;
          }
          default:
          {
            assert(false);
          }
        }

        PopulateTrigger(tri, force_first_iter, allow_rescale_dt, dt);
        triggers[MakeTriggerMeta(tvar, ovar)] = tri;
        break;
      }
      case TriggerVariant::Z4c_ADM_constraints:
      {
        Real dt = 0;
        switch (ovar)
        {
          case (Triggers::OutputVariant::user):
          {
            dt = pin->GetOrAddReal("task_triggers",
                                   "dt_Z4c_ADM_constraints", 0.0);
            break;
          }
          case (Triggers::OutputVariant::hst):
          {
            dt = pouts->GetMinOutputTimeStepExhaustive("hst");
            break;
          }
          case (Triggers::OutputVariant::data):
          {
            dt = pouts->GetMinOutputTimeStepExhaustive("con");
            break;
          }
          default:
          {
            assert(false);
          }
        }

        PopulateTrigger(tri, force_first_iter, allow_rescale_dt, dt);
        triggers[MakeTriggerMeta(tvar, ovar)] = tri;
        break;
      }
      case TriggerVariant::Z4c_tracker_punctures:
      {
        Real dt = 0;
        switch (ovar)
        {
          case (Triggers::OutputVariant::user):
          {
            dt = pin->GetOrAddReal("task_triggers",
                                   "Z4c_tracker_punctures",
                                   0.0);
            break;
          }
          default:
          {
            assert(false);
          }
        }

        PopulateTrigger(tri, force_first_iter, allow_rescale_dt, dt);
        triggers[MakeTriggerMeta(tvar, ovar)] = tri;
        break;
      }
      case TriggerVariant::Z4c_Weyl:
      {
        Real dt = 0;
        switch (ovar)
        {
          case (Triggers::OutputVariant::user):
          {
            dt = pin->GetOrAddReal("task_triggers",
                                   "dt_Z4c_Weyl", 0.0);
            break;
          }
          case (Triggers::OutputVariant::data):
          {
            dt = pouts->GetMinOutputTimeStepExhaustive("Weyl");
            break;
          }
          default:
          {
            assert(false);
          }
        }

        PopulateTrigger(tri, force_first_iter, allow_rescale_dt, dt);
        triggers[MakeTriggerMeta(tvar, ovar)] = tri;
        break;
      }
      default:
      {
        assert(false);
      }
    }
  };

  // Check whether a trigger is satisfied; checks all output variants
  bool IsSatisfied(TriggerVariant tvar, OutputVariant ovar)
  {
    return triggers[MakeTriggerMeta(tvar, ovar)].is_satisfied();
  }

  // Check whether a trigger is satisfied; any output style
  bool IsSatisfied(TriggerVariant tvar)
  {
    bool satisfied = false;
    for (auto& [tvar_iter, tri] : triggers)
    {
      // Note:
      // TriggerVariant tvar_cur = std::get<0>(tvar_iter);
      // OutputVariant  ovar_cur = std::get<1>(tvar_iter);

      if ((std::get<0>(tvar_iter) == tvar))
      {
        bool was_satisfied = tri.is_satisfied();
        satisfied = satisfied || was_satisfied;
      }
    }
    return satisfied;
  }

  // Iterate over triggers, if any registered is allowed to reduce pm->dt then
  // do so. Return if this occurred.
  bool AdjustFromAny_mesh_dt()
  {
    bool adjusted = false;
    for (auto& [var, tri] : triggers)
    {
      if (tri.is_satisfied())
      {
        bool was_reduced = tri.reduce_mesh_dt();
        adjusted = adjusted || was_reduced;
      }
    }
    return adjusted;
  }

  // Iterate over triggers, updating for next time-step as required
  void Update()
  {
    for (auto& [var, tri] : triggers)
    {
      if (tri.is_satisfied())
      {
        tri.update();
      }
    }
  }

private:
  void PopulateTrigger(Trigger & tri,
                       const bool force_first_iter,
                       const bool allow_rescale_dt,
                       const Real dt)
  {
    tri.pm = pm;
    tri.dt = dt;

    // tri.t_last = pm->start_time;
    // use "time" as this gets "start_time" if specified
    // Further, it is retained on restarts

    const int ndumps = (dt > 0)
      ? (static_cast<int>(pm->time / dt))
      : 0;
    tri.t_last = ndumps * dt;
    tri.t_next = (ndumps + 1) * dt;

    if (force_first_iter)
    {
      // on restarts first iterate may exceed next permitted
      // step based on dump iter
      // tri.t_next = pm->time+pm->dt;
      tri.t_next = std::min(pm->time + pm->dt, tri.t_next);
      tri.force_first_iter = true;
    }

    tri.allow_rescale_dt = allow_rescale_dt;

  }

private:
  Mesh *pm;
  ParameterInput * pin;
  Outputs *pouts;

};

}  // namespace gra::triggers


#endif // MAIN_TRIGGERS_HPP
//
// :D
//
