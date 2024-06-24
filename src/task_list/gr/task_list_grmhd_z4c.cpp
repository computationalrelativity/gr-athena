// C/C++ headers
#include <iostream>   // endl
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ classes headers
#include "../../athena.hpp"
#include "../../bvals/bvals.hpp"
#include "../../eos/eos.hpp"
#include "../../field/field.hpp"
#include "../../gravity/gravity.hpp"
#include "../../hydro/hydro.hpp"
#include "../../hydro/srcterms/hydro_srcterms.hpp"
#include "../../mesh/mesh.hpp"
#include "../../z4c/z4c.hpp"
#include "../../z4c/wave_extract.hpp"
#include "../../z4c/puncture_tracker.hpp"
#include "../../trackers/extrema_tracker.hpp"
#include "../../reconstruct/reconstruction.hpp"
#include "../../scalars/scalars.hpp"
#include "task_list.hpp"
#include "task_names.hpp"


// ----------------------------------------------------------------------------
using namespace TaskLists::GeneralRelativity;
using namespace TaskLists::Integrators;
using namespace TaskNames::GeneralRelativity::GRMHD_Z4c;

using namespace gra::triggers;
typedef Triggers::TriggerVariant TriggerVariant;
// ----------------------------------------------------------------------------

GRMHD_Z4c::GRMHD_Z4c(ParameterInput *pin,
                     Mesh *pm,
                     Triggers &trgs)
  : LowStorage(pin, pm),
    trgs(trgs)
{
  // Take the number of stages from the integrator
  nstages = LowStorage::nstages;

  //---------------------------------------------------------------------------

  const bool multilevel = pm->multilevel;  // for SMR or AMR logic
  const bool adaptive   = pm->adaptive;    // AMR

  // reference task-list (from older branches)
#ifdef DBG_USE_REFERENCE_TASKLISTS
  {
    // (M)HD sub-system logic -------------------------------------------------
    Add(CALC_HYDFLX, NONE, &GRMHD_Z4c::CalculateHydroFlux);

    if (NSCALARS > 0)
    {
      Add(CALC_SCLRFLX, CALC_HYDFLX, &GRMHD_Z4c::CalculateScalarFlux);
    }

    if (multilevel)
    {
      Add(SEND_HYDFLX, CALC_HYDFLX, &GRMHD_Z4c::SendFluxCorrectionHydro);
      Add(RECV_HYDFLX, CALC_HYDFLX, &GRMHD_Z4c::ReceiveAndCorrectHydroFlux);
      Add(INT_HYD,     RECV_HYDFLX, &GRMHD_Z4c::IntegrateHydro);
    }
    else
    {
      Add(INT_HYD, CALC_HYDFLX, &GRMHD_Z4c::IntegrateHydro);
    }

    Add(SRCTERM_HYD, INT_HYD,     &GRMHD_Z4c::AddSourceTermsHydro);
    Add(SEND_HYD,    SRCTERM_HYD, &GRMHD_Z4c::SendHydro);
    Add(RECV_HYD,    INT_HYD,     &GRMHD_Z4c::ReceiveHydro);

    Add(SETB_HYD, (RECV_HYD | SRCTERM_HYD), &GRMHD_Z4c::SetBoundariesHydro);

    if (NSCALARS > 0)
    {
      if (multilevel)
      {
        Add(SEND_SCLRFLX, CALC_SCLRFLX, &GRMHD_Z4c::SendScalarFlux);
        Add(RECV_SCLRFLX, CALC_SCLRFLX, &GRMHD_Z4c::ReceiveScalarFlux);
        Add(INT_SCLR,     RECV_SCLRFLX, &GRMHD_Z4c::IntegrateScalars);
      }
      else
      {
        Add(INT_SCLR, CALC_SCLRFLX, &GRMHD_Z4c::IntegrateScalars);
      }

      Add(SEND_SCLR, INT_SCLR, &GRMHD_Z4c::SendScalars);
      Add(RECV_SCLR, NONE,     &GRMHD_Z4c::ReceiveScalars);

      Add(SETB_SCLR, (RECV_SCLR | INT_SCLR), &GRMHD_Z4c::SetBoundariesScalars);
    }

    if (MAGNETIC_FIELDS_ENABLED)
    {
      // compute MHD fluxes, integrate field
      Add(CALC_FLDFLX, CALC_HYDFLX, &GRMHD_Z4c::CalculateEMF);
      Add(SEND_FLDFLX, CALC_FLDFLX, &GRMHD_Z4c::SendFluxCorrectionEMF);
      Add(RECV_FLDFLX, SEND_FLDFLX, &GRMHD_Z4c::ReceiveAndCorrectEMF);
      Add(INT_FLD,     RECV_FLDFLX, &GRMHD_Z4c::IntegrateField);

      Add(SEND_FLD, INT_FLD, &GRMHD_Z4c::SendField);
      Add(RECV_FLD, NONE,    &GRMHD_Z4c::ReceiveField);
      Add(SETB_FLD, (RECV_FLD | INT_FLD), &GRMHD_Z4c::SetBoundariesField);

      // prolongate, compute new primitives
      if (multilevel)
      {
        if (NSCALARS > 0)
        {
          Add(PROLONG_HYD,
              (SEND_HYD  | SETB_HYD  | SEND_FLD | SETB_FLD |
               SEND_SCLR | SETB_SCLR | Z4C_TO_ADM),
              &GRMHD_Z4c::Prolongation_Hyd);
        }
        else
        {
          Add(PROLONG_HYD,
              (SEND_HYD | SETB_HYD | SEND_FLD | SETB_FLD |
               Z4C_TO_ADM),
              &GRMHD_Z4c::Prolongation_Hyd);
        }
        Add(CONS2PRIM, (PROLONG_HYD | Z4C_TO_ADM), &GRMHD_Z4c::Primitives);
      }
      else
      {
        if (NSCALARS > 0)
        {
          Add(CONS2PRIM, (SETB_HYD | SETB_FLD | SETB_SCLR),
              &GRMHD_Z4c::Primitives);
        }
        else
        {
          Add(CONS2PRIM, (SETB_HYD | SETB_FLD | Z4C_TO_ADM),
              &GRMHD_Z4c::Primitives);
        }
      }
    }
    else  // otherwise GRHD
    {
      // prolongate, compute new primitives
      if (multilevel)
      {
        if (NSCALARS > 0)
        {
          Add(PROLONG_HYD,
              (SEND_HYD | SETB_HYD | SETB_SCLR | SEND_SCLR | Z4C_TO_ADM),
              &GRMHD_Z4c::Prolongation_Hyd);
        }
        else
        {
          Add(PROLONG_HYD,
              (SEND_HYD | SETB_HYD | Z4C_TO_ADM),
              &GRMHD_Z4c::Prolongation_Hyd);
        }
        Add(CONS2PRIM,(PROLONG_HYD | Z4C_TO_ADM), &GRMHD_Z4c::Primitives);
      }
      else
      {

        if (NSCALARS > 0)
        {
          Add(CONS2PRIM, (SETB_HYD | SETB_SCLR), &GRMHD_Z4c::Primitives);
        }
        else
        {
          Add(CONS2PRIM,
              (SETB_HYD | Z4C_TO_ADM),
              &GRMHD_Z4c::Primitives);
        }

      }
    }

    Add(UPDATE_SRC,   (CONS2PRIM | CALC_Z4CRHS),
        &GRMHD_Z4c::UpdateSource);
    Add(PHY_BVAL_HYD, CONS2PRIM, &GRMHD_Z4c::PhysicalBoundary_Hyd);

    // Z4c sub-system logic ---------------------------------------------------
    Add(CALC_Z4CRHS, NONE,        &GRMHD_Z4c::CalculateZ4cRHS);
    Add(INT_Z4C,     CALC_Z4CRHS, &GRMHD_Z4c::IntegrateZ4c);

    Add(SEND_Z4C, INT_Z4C, &GRMHD_Z4c::SendZ4c);

    if(MAGNETIC_FIELDS_ENABLED)
    {
      Add(RECV_Z4C, (INT_Z4C | RECV_HYD | RECV_FLD | RECV_FLDFLX),
          &GRMHD_Z4c::ReceiveZ4c);
    }
    else
    {
      Add(RECV_Z4C, (INT_Z4C | RECV_HYD), &GRMHD_Z4c::ReceiveZ4c);
    }

    Add(SETB_Z4C, (RECV_Z4C | INT_Z4C), &GRMHD_Z4c::SetBoundariesZ4c);

    if (multilevel)
    {
      Add(PROLONG_Z4C,  (SEND_Z4C | SETB_Z4C), &GRMHD_Z4c::Prolongation_Z4c);
      Add(PHY_BVAL_Z4C, PROLONG_Z4C,
          &GRMHD_Z4c::PhysicalBoundary_Z4c);
    }
    else
    {
      Add(PHY_BVAL_Z4C, SETB_Z4C, &GRMHD_Z4c::PhysicalBoundary_Z4c);
    }

    Add(ALG_CONSTR, PHY_BVAL_Z4C, &GRMHD_Z4c::EnforceAlgConstr);

    if (MAGNETIC_FIELDS_ENABLED)
    {
      Add(Z4C_TO_ADM, (ALG_CONSTR | INT_HYD | INT_FLD),
          &GRMHD_Z4c::Z4cToADM);
    }
    else
    {
      Add(Z4C_TO_ADM, (ALG_CONSTR | INT_HYD), &GRMHD_Z4c::Z4cToADM);
    }

    Add(ADM_CONSTR, (Z4C_TO_ADM | UPDATE_SRC), &GRMHD_Z4c::ADM_Constraints);

    Add(Z4C_WEYL,  Z4C_TO_ADM, &GRMHD_Z4c::Z4c_Weyl);

    Add(USERWORK, (ADM_CONSTR | PHY_BVAL_HYD), &GRMHD_Z4c::UserWork);
    Add(NEW_DT,   USERWORK,                    &GRMHD_Z4c::NewBlockTimeStep);

    if (adaptive)
    {
      Add(FLAG_AMR,     USERWORK, &GRMHD_Z4c::CheckRefinement);
      Add(CLEAR_ALLBND, FLAG_AMR, &GRMHD_Z4c::ClearAllBoundary);
    }
    else
    {
      Add(CLEAR_ALLBND, NEW_DT, &GRMHD_Z4c::ClearAllBoundary);
    }

  }
#endif // DBG_USE_REFERENCE_TASKLISTS

  // basic GRHD AMR
  if (0) {
    // (M)HD sub-system logic -------------------------------------------------
    Add(CALC_HYDFLX, NONE, &GRMHD_Z4c::CalculateHydroFlux);

    Add(SEND_HYDFLX, CALC_HYDFLX, &GRMHD_Z4c::SendFluxCorrectionHydro);
    Add(RECV_HYDFLX, CALC_HYDFLX, &GRMHD_Z4c::ReceiveAndCorrectHydroFlux);
    Add(INT_HYD,     RECV_HYDFLX, &GRMHD_Z4c::IntegrateHydro);

    Add(SRCTERM_HYD, INT_HYD,     &GRMHD_Z4c::AddSourceTermsHydro);

    Add(SEND_HYD,    SRCTERM_HYD, &GRMHD_Z4c::SendHydro);
    Add(RECV_HYD,    INT_HYD,     &GRMHD_Z4c::ReceiveHydro);

    Add(SETB_HYD, RECV_HYD, &GRMHD_Z4c::SetBoundariesHydro);

    {
      // prolongate, compute new primitives
      if (multilevel)
      {
        Add(PROLONG_HYD,
            (SEND_HYD | SETB_HYD | Z4C_TO_ADM),
            &GRMHD_Z4c::Prolongation_Hyd);
        Add(CONS2PRIM,(PROLONG_HYD | Z4C_TO_ADM), &GRMHD_Z4c::Primitives);
      }
      else
      {

        Add(CONS2PRIM,
            (SETB_HYD | Z4C_TO_ADM),
            &GRMHD_Z4c::Primitives);

      }
    }

    Add(PHY_BVAL_HYD, CONS2PRIM, &GRMHD_Z4c::PhysicalBoundary_Hyd);
    Add(UPDATE_SRC, PHY_BVAL_HYD, &GRMHD_Z4c::UpdateSource);

    // Z4c sub-system logic ---------------------------------------------------
    Add(CALC_Z4CRHS, NONE,        &GRMHD_Z4c::CalculateZ4cRHS);
    Add(INT_Z4C,     CALC_Z4CRHS, &GRMHD_Z4c::IntegrateZ4c);

    Add(SEND_Z4C, INT_Z4C, &GRMHD_Z4c::SendZ4c);
    Add(RECV_Z4C, NONE,    &GRMHD_Z4c::ReceiveZ4c);
    Add(SETB_Z4C, RECV_Z4C, &GRMHD_Z4c::SetBoundariesZ4c);

    Add(PROLONG_Z4C,  (SEND_Z4C | SETB_Z4C), &GRMHD_Z4c::Prolongation_Z4c);
    Add(PHY_BVAL_Z4C, PROLONG_Z4C, &GRMHD_Z4c::PhysicalBoundary_Z4c);
    Add(ALG_CONSTR, PHY_BVAL_Z4C, &GRMHD_Z4c::EnforceAlgConstr);

    Add(Z4C_TO_ADM, ALG_CONSTR, &GRMHD_Z4c::Z4cToADM);

    Add(ADM_CONSTR, UPDATE_SRC, &GRMHD_Z4c::ADM_Constraints);
    Add(Z4C_WEYL,  Z4C_TO_ADM, &GRMHD_Z4c::Z4c_Weyl);

    Add(USERWORK, ADM_CONSTR, &GRMHD_Z4c::UserWork);
    Add(NEW_DT,   USERWORK,   &GRMHD_Z4c::NewBlockTimeStep);

    Add(FLAG_AMR,     USERWORK, &GRMHD_Z4c::CheckRefinement);
    Add(CLEAR_ALLBND, FLAG_AMR, &GRMHD_Z4c::ClearAllBoundary);

  }

  // change ordering of tasks to better reflect state-vector time
  if (0) {
    // (M)HD sub-system logic -------------------------------------------------
    Add(CALC_HYDFLX, NONE, &GRMHD_Z4c::CalculateHydroFlux);

    Add(SEND_HYDFLX, CALC_HYDFLX, &GRMHD_Z4c::SendFluxCorrectionHydro);
    Add(RECV_HYDFLX, CALC_HYDFLX, &GRMHD_Z4c::ReceiveAndCorrectHydroFlux);
    Add(INT_HYD,     RECV_HYDFLX, &GRMHD_Z4c::IntegrateHydro);

    Add(SRCTERM_HYD, INT_HYD,     &GRMHD_Z4c::AddSourceTermsHydro);

    // now need advanced z4c(ADM) state
    Add(CALC_Z4CRHS, SRCTERM_HYD,        &GRMHD_Z4c::CalculateZ4cRHS);
    Add(INT_Z4C,     CALC_Z4CRHS, &GRMHD_Z4c::IntegrateZ4c);

    Add(SEND_Z4C, INT_Z4C, &GRMHD_Z4c::SendZ4c);
    Add(RECV_Z4C, NONE,    &GRMHD_Z4c::ReceiveZ4c);
    Add(SETB_Z4C, RECV_Z4C, &GRMHD_Z4c::SetBoundariesZ4c);

    Add(PROLONG_Z4C,  (SEND_Z4C | SETB_Z4C), &GRMHD_Z4c::Prolongation_Z4c);
    Add(PHY_BVAL_Z4C, PROLONG_Z4C, &GRMHD_Z4c::PhysicalBoundary_Z4c);
    Add(ALG_CONSTR, PHY_BVAL_Z4C, &GRMHD_Z4c::EnforceAlgConstr);
    Add(Z4C_TO_ADM, ALG_CONSTR, &GRMHD_Z4c::Z4cToADM);

    // now can send hydro


    Add(SEND_HYD,    Z4C_TO_ADM, &GRMHD_Z4c::SendHydro);
    Add(RECV_HYD,    Z4C_TO_ADM,     &GRMHD_Z4c::ReceiveHydro);

    Add(SETB_HYD, RECV_HYD, &GRMHD_Z4c::SetBoundariesHydro);

    {
      // prolongate, compute new primitives
      Add(PROLONG_HYD,
          (SEND_HYD | SETB_HYD),
          &GRMHD_Z4c::Prolongation_Hyd);
      Add(CONS2PRIM,(PROLONG_HYD), &GRMHD_Z4c::Primitives);
    }

    Add(PHY_BVAL_HYD, CONS2PRIM, &GRMHD_Z4c::PhysicalBoundary_Hyd);
    Add(UPDATE_SRC, PHY_BVAL_HYD, &GRMHD_Z4c::UpdateSource);

    // Z4c sub-system logic ---------------------------------------------------



    Add(ADM_CONSTR, UPDATE_SRC, &GRMHD_Z4c::ADM_Constraints);
    Add(Z4C_WEYL,  Z4C_TO_ADM, &GRMHD_Z4c::Z4c_Weyl);

    Add(USERWORK, ADM_CONSTR, &GRMHD_Z4c::UserWork);
    Add(NEW_DT,   USERWORK,   &GRMHD_Z4c::NewBlockTimeStep);

    Add(FLAG_AMR,     USERWORK, &GRMHD_Z4c::CheckRefinement);
    Add(CLEAR_ALLBND, FLAG_AMR, &GRMHD_Z4c::ClearAllBoundary);

  }

  // metricial data is advanced in hydro sources
  if (0) {
    // (M)HD sub-system logic -------------------------------------------------
    Add(CALC_HYDFLX, NONE, &GRMHD_Z4c::CalculateHydroFlux);

    Add(SEND_HYDFLX, CALC_HYDFLX, &GRMHD_Z4c::SendFluxCorrectionHydro);
    Add(RECV_HYDFLX, CALC_HYDFLX, &GRMHD_Z4c::ReceiveAndCorrectHydroFlux);
    Add(INT_HYD,     RECV_HYDFLX, &GRMHD_Z4c::IntegrateHydro);

    // now need advanced z4c(ADM) state
    Add(CALC_Z4CRHS, INT_HYD,        &GRMHD_Z4c::CalculateZ4cRHS);
    Add(INT_Z4C,     CALC_Z4CRHS, &GRMHD_Z4c::IntegrateZ4c);

    Add(SEND_Z4C, INT_Z4C, &GRMHD_Z4c::SendZ4c);
    Add(RECV_Z4C, NONE,    &GRMHD_Z4c::ReceiveZ4c);
    Add(SETB_Z4C, RECV_Z4C, &GRMHD_Z4c::SetBoundariesZ4c);

    Add(PROLONG_Z4C,  (SEND_Z4C | SETB_Z4C), &GRMHD_Z4c::Prolongation_Z4c);
    Add(PHY_BVAL_Z4C, PROLONG_Z4C, &GRMHD_Z4c::PhysicalBoundary_Z4c);
    Add(ALG_CONSTR, PHY_BVAL_Z4C, &GRMHD_Z4c::EnforceAlgConstr);
    Add(Z4C_TO_ADM, ALG_CONSTR, &GRMHD_Z4c::Z4cToADM);

    // compute hydro source terms
    Add(SRCTERM_HYD, Z4C_TO_ADM,     &GRMHD_Z4c::AddSourceTermsHydro);

    // now can send hydro
    Add(SEND_HYD, SRCTERM_HYD, &GRMHD_Z4c::SendHydro);
    Add(RECV_HYD, SRCTERM_HYD,     &GRMHD_Z4c::ReceiveHydro);

    Add(SETB_HYD, RECV_HYD, &GRMHD_Z4c::SetBoundariesHydro);

    {
      // prolongate, compute new primitives
      Add(PROLONG_HYD,
          (SEND_HYD | SETB_HYD),
          &GRMHD_Z4c::Prolongation_Hyd);
      Add(CONS2PRIM,(PROLONG_HYD), &GRMHD_Z4c::Primitives);
    }

    Add(PHY_BVAL_HYD, CONS2PRIM, &GRMHD_Z4c::PhysicalBoundary_Hyd);
    Add(UPDATE_SRC, PHY_BVAL_HYD, &GRMHD_Z4c::UpdateSource);

    // Z4c sub-system logic ---------------------------------------------------



    Add(ADM_CONSTR, UPDATE_SRC, &GRMHD_Z4c::ADM_Constraints);
    Add(Z4C_WEYL,  Z4C_TO_ADM, &GRMHD_Z4c::Z4c_Weyl);

    Add(USERWORK, ADM_CONSTR, &GRMHD_Z4c::UserWork);
    Add(NEW_DT,   USERWORK,   &GRMHD_Z4c::NewBlockTimeStep);

    Add(FLAG_AMR,     USERWORK, &GRMHD_Z4c::CheckRefinement);
    Add(CLEAR_ALLBND, FLAG_AMR, &GRMHD_Z4c::ClearAllBoundary);

  }

#ifndef DBG_USE_REFERENCE_TASKLISTS
  #ifndef DBG_USE_CONS_BC
  // refactored task-list
  {
    // (M)HD sub-system logic -------------------------------------------------
    Add(CALC_HYDFLX, NONE, &GRMHD_Z4c::CalculateHydroFlux);

    if (NSCALARS > 0)
    {
      Add(CALC_SCLRFLX, CALC_HYDFLX, &GRMHD_Z4c::CalculateScalarFlux);
    }

    if (multilevel)
    {
      Add(SEND_HYDFLX, CALC_HYDFLX, &GRMHD_Z4c::SendFluxCorrectionHydro);
      Add(RECV_HYDFLX, CALC_HYDFLX, &GRMHD_Z4c::ReceiveAndCorrectHydroFlux);
      Add(INT_HYD,     RECV_HYDFLX, &GRMHD_Z4c::IntegrateHydro);
    }
    else
    {
      Add(INT_HYD, CALC_HYDFLX, &GRMHD_Z4c::IntegrateHydro);
    }

    Add(SRCTERM_HYD, INT_HYD,     &GRMHD_Z4c::AddSourceTermsHydro);
    Add(SEND_HYD,    SRCTERM_HYD, &GRMHD_Z4c::SendHydro);
    Add(RECV_HYD,    NONE,     &GRMHD_Z4c::ReceiveHydro);

    Add(SETB_HYD, (RECV_HYD | SRCTERM_HYD), &GRMHD_Z4c::SetBoundariesHydro);

    if (NSCALARS > 0)
    {
      if (multilevel)
      {
        Add(SEND_SCLRFLX, CALC_SCLRFLX, &GRMHD_Z4c::SendScalarFlux);
        Add(RECV_SCLRFLX, CALC_SCLRFLX, &GRMHD_Z4c::ReceiveScalarFlux);
        Add(INT_SCLR,     RECV_SCLRFLX, &GRMHD_Z4c::IntegrateScalars);
      }
      else
      {
        Add(INT_SCLR, CALC_SCLRFLX, &GRMHD_Z4c::IntegrateScalars);
      }

      Add(SEND_SCLR, INT_SCLR, &GRMHD_Z4c::SendScalars);
      Add(RECV_SCLR, NONE,     &GRMHD_Z4c::ReceiveScalars);

      Add(SETB_SCLR, (RECV_SCLR | INT_SCLR), &GRMHD_Z4c::SetBoundariesScalars);
    }

    if (MAGNETIC_FIELDS_ENABLED)
    {
      // compute MHD fluxes, integrate field
      Add(CALC_FLDFLX, CALC_HYDFLX, &GRMHD_Z4c::CalculateEMF);
      Add(SEND_FLDFLX, CALC_FLDFLX, &GRMHD_Z4c::SendFluxCorrectionEMF);
      Add(RECV_FLDFLX, CALC_FLDFLX, &GRMHD_Z4c::ReceiveAndCorrectEMF);
      Add(INT_FLD,     RECV_FLDFLX, &GRMHD_Z4c::IntegrateField);

      Add(SEND_FLD, INT_FLD, &GRMHD_Z4c::SendField);
      Add(RECV_FLD, NONE,    &GRMHD_Z4c::ReceiveField);
      Add(SETB_FLD, (RECV_FLD | INT_FLD), &GRMHD_Z4c::SetBoundariesField);

      // prolongate, compute new primitives
      if (multilevel)
      {
        if (NSCALARS > 0)
        {
          Add(PROLONG_HYD, (Z4C_TO_ADM | SEND_HYD | SEND_FLD | SEND_SCLR),
              &GRMHD_Z4c::Prolongation_Hyd);
        }
        else
        {
          Add(PROLONG_HYD, (Z4C_TO_ADM | SEND_HYD | SEND_FLD),
              &GRMHD_Z4c::Prolongation_Hyd);
        }
        Add(CONS2PRIM, PROLONG_HYD, &GRMHD_Z4c::Primitives);
      }
      else
      {
        Add(CONS2PRIM, Z4C_TO_ADM, &GRMHD_Z4c::Primitives);
      }
    }
    else  // otherwise GRHD
    {
      // prolongate, compute new primitives
      if (multilevel)
      {
        if (NSCALARS > 0)
        {
          Add(PROLONG_HYD, (Z4C_TO_ADM | SEND_HYD| SEND_SCLR),
              &GRMHD_Z4c::Prolongation_Hyd);
        }
        else
        {
          Add(PROLONG_HYD, (Z4C_TO_ADM | SEND_HYD),
              &GRMHD_Z4c::Prolongation_Hyd);
        }
        Add(CONS2PRIM, PROLONG_HYD, &GRMHD_Z4c::Primitives);
      }
      else
      {
        Add(CONS2PRIM, Z4C_TO_ADM, &GRMHD_Z4c::Primitives);
      }
    }

    if (!multilevel)
    {
      if(MAGNETIC_FIELDS_ENABLED)
      {
        if (NSCALARS > 0)
        {
          Add(PHY_BVAL_HYD, (CONS2PRIM | SEND_HYD | SEND_FLD | SEND_SCLR),
              &GRMHD_Z4c::PhysicalBoundary_Hyd);
        }
        else
        {
          Add(PHY_BVAL_HYD, (CONS2PRIM | SEND_HYD | SEND_FLD),
              &GRMHD_Z4c::PhysicalBoundary_Hyd);
        }
      }
      else
      {
        if (NSCALARS > 0)
        {
          Add(PHY_BVAL_HYD, (CONS2PRIM | SEND_HYD | SEND_SCLR),
              &GRMHD_Z4c::PhysicalBoundary_Hyd);
        }
        else
        {
          Add(PHY_BVAL_HYD, (CONS2PRIM | SEND_HYD),
              &GRMHD_Z4c::PhysicalBoundary_Hyd);
        }
      }
    }
    else
    {
      Add(PHY_BVAL_HYD, CONS2PRIM, &GRMHD_Z4c::PhysicalBoundary_Hyd);
    }

    Add(UPDATE_SRC, PHY_BVAL_HYD, &GRMHD_Z4c::UpdateSource);

    // collect all MHD-scalar communication into blocking task ----------------
    // In principle this should not be required, there is somewhere MPI issue
    TaskID MAT_COM = SEND_HYD | RECV_HYD;
    if (multilevel)
    {
      MAT_COM = MAT_COM | SEND_HYDFLX | RECV_HYDFLX;
    }

    if (NSCALARS > 0)
    {
      MAT_COM = MAT_COM | SEND_SCLR | RECV_SCLR;
      if (multilevel)
      {
        MAT_COM = MAT_COM | SEND_SCLRFLX | RECV_SCLRFLX;
      }
    }

    if (MAGNETIC_FIELDS_ENABLED)
    {
      MAT_COM = MAT_COM | SEND_FLD | RECV_FLD | SEND_FLDFLX | RECV_FLDFLX;
    }

    // Z4c sub-system logic ---------------------------------------------------
    Add(CALC_Z4CRHS, NONE,        &GRMHD_Z4c::CalculateZ4cRHS);
    Add(INT_Z4C,     CALC_Z4CRHS, &GRMHD_Z4c::IntegrateZ4c);

    // Should be able to do this
    // Add(SEND_Z4C, INT_Z4C, &GRMHD_Z4c::SendZ4c);
    // Add(RECV_Z4C, NONE, &GRMHD_Z4c::ReceiveZ4c);

    // Do instead this
    Add(SEND_Z4C, (INT_Z4C | MAT_COM), &GRMHD_Z4c::SendZ4c);
    Add(RECV_Z4C, MAT_COM,             &GRMHD_Z4c::ReceiveZ4c);

    Add(SETB_Z4C, (RECV_Z4C | INT_Z4C), &GRMHD_Z4c::SetBoundariesZ4c);

    if (multilevel)
    {
      Add(PROLONG_Z4C,  SETB_Z4C, &GRMHD_Z4c::Prolongation_Z4c);
      Add(PHY_BVAL_Z4C, PROLONG_Z4C, &GRMHD_Z4c::PhysicalBoundary_Z4c);
    }
    else
    {
      Add(PHY_BVAL_Z4C, SETB_Z4C, &GRMHD_Z4c::PhysicalBoundary_Z4c);
    }

    Add(ALG_CONSTR, PHY_BVAL_Z4C, &GRMHD_Z4c::EnforceAlgConstr);

    if (MAGNETIC_FIELDS_ENABLED)
    {
      if (NSCALARS > 0)
      {
        Add(Z4C_TO_ADM, (ALG_CONSTR | SETB_HYD | SETB_FLD | SETB_SCLR),
            &GRMHD_Z4c::Z4cToADM);
      }
      else
      {
        Add(Z4C_TO_ADM, (ALG_CONSTR | SETB_HYD | SETB_FLD),
            &GRMHD_Z4c::Z4cToADM);
      }
    }
    else
    {
      if (NSCALARS > 0)
      {
        Add(Z4C_TO_ADM, (ALG_CONSTR | SETB_HYD | SETB_SCLR),
            &GRMHD_Z4c::Z4cToADM);
      }
      else
      {
        Add(Z4C_TO_ADM, (ALG_CONSTR | SETB_HYD), &GRMHD_Z4c::Z4cToADM);
      }
    }

    Add(ADM_CONSTR, UPDATE_SRC, &GRMHD_Z4c::ADM_Constraints);

    Add(Z4C_WEYL,  Z4C_TO_ADM, &GRMHD_Z4c::Z4c_Weyl);

    Add(USERWORK, ADM_CONSTR, &GRMHD_Z4c::UserWork);
    Add(NEW_DT,   USERWORK,   &GRMHD_Z4c::NewBlockTimeStep);

    if (adaptive)
    {
      Add(FLAG_AMR,     USERWORK, &GRMHD_Z4c::CheckRefinement);
      Add(CLEAR_ALLBND, FLAG_AMR, &GRMHD_Z4c::ClearAllBoundary);
    }
    else
    {
      Add(CLEAR_ALLBND, NEW_DT, &GRMHD_Z4c::ClearAllBoundary);
    }

  }
  #else
  // refactored (for cons_bc) task-list
  {
    // (M)HD sub-system logic -------------------------------------------------
    Add(CALC_HYDFLX, NONE, &GRMHD_Z4c::CalculateHydroFlux);

    if (NSCALARS > 0)
    {
      Add(CALC_SCLRFLX, CALC_HYDFLX, &GRMHD_Z4c::CalculateScalarFlux);
    }

    if (multilevel)
    {
      Add(SEND_HYDFLX, CALC_HYDFLX, &GRMHD_Z4c::SendFluxCorrectionHydro);
      Add(RECV_HYDFLX, CALC_HYDFLX, &GRMHD_Z4c::ReceiveAndCorrectHydroFlux);
      Add(INT_HYD,     RECV_HYDFLX, &GRMHD_Z4c::IntegrateHydro);
    }
    else
    {
      Add(INT_HYD, CALC_HYDFLX, &GRMHD_Z4c::IntegrateHydro);
    }

    Add(SRCTERM_HYD, INT_HYD,     &GRMHD_Z4c::AddSourceTermsHydro);
    Add(SEND_HYD,    SRCTERM_HYD, &GRMHD_Z4c::SendHydro);
    Add(RECV_HYD,    NONE,     &GRMHD_Z4c::ReceiveHydro);

    Add(SETB_HYD, (RECV_HYD | SRCTERM_HYD), &GRMHD_Z4c::SetBoundariesHydro);

    if (NSCALARS > 0)
    {
      if (multilevel)
      {
        Add(SEND_SCLRFLX, CALC_SCLRFLX, &GRMHD_Z4c::SendScalarFlux);
        Add(RECV_SCLRFLX, CALC_SCLRFLX, &GRMHD_Z4c::ReceiveScalarFlux);
        Add(INT_SCLR,     RECV_SCLRFLX, &GRMHD_Z4c::IntegrateScalars);
      }
      else
      {
        Add(INT_SCLR, CALC_SCLRFLX, &GRMHD_Z4c::IntegrateScalars);
      }

      Add(SEND_SCLR, INT_SCLR, &GRMHD_Z4c::SendScalars);
      Add(RECV_SCLR, NONE,     &GRMHD_Z4c::ReceiveScalars);

      Add(SETB_SCLR, (RECV_SCLR | INT_SCLR), &GRMHD_Z4c::SetBoundariesScalars);
    }

    if (MAGNETIC_FIELDS_ENABLED)
    {
      // compute MHD fluxes, integrate field
      Add(CALC_FLDFLX, CALC_HYDFLX, &GRMHD_Z4c::CalculateEMF);
      Add(SEND_FLDFLX, CALC_FLDFLX, &GRMHD_Z4c::SendFluxCorrectionEMF);
      Add(RECV_FLDFLX, CALC_FLDFLX, &GRMHD_Z4c::ReceiveAndCorrectEMF);
      Add(INT_FLD,     RECV_FLDFLX, &GRMHD_Z4c::IntegrateField);

      Add(SEND_FLD, INT_FLD, &GRMHD_Z4c::SendField);
      Add(RECV_FLD, NONE,    &GRMHD_Z4c::ReceiveField);
      Add(SETB_FLD, (RECV_FLD | INT_FLD), &GRMHD_Z4c::SetBoundariesField);

      // prolongate, compute new primitives
      if (multilevel)
      {
        if (NSCALARS > 0)
        {
          Add(PROLONG_HYD, (SETB_HYD | SETB_FLD | SETB_SCLR),
              &GRMHD_Z4c::Prolongation_Hyd);
        }
        else
        {
          Add(PROLONG_HYD, (SETB_HYD | SETB_FLD),
              &GRMHD_Z4c::Prolongation_Hyd);
        }
        Add(PHY_BVAL_HYD, PROLONG_HYD, &GRMHD_Z4c::Primitives);
      }
      else
      {
        Add(PHY_BVAL_HYD, (SETB_HYD | SETB_FLD), &GRMHD_Z4c::Primitives);
      }
    }
    else  // otherwise GRHD
    {
      // prolongate, compute new primitives
      if (multilevel)
      {
        if (NSCALARS > 0)
        {
          Add(PROLONG_HYD, (SETB_HYD | SETB_SCLR),
              &GRMHD_Z4c::Prolongation_Hyd);
        }
        else
        {
          // DEBUG: add back the send dep.
          // Add(PROLONG_HYD, (SETB_HYD | SEND_HYD), &GRMHD_Z4c::Prolongation_Hyd);
          Add(PROLONG_HYD, SETB_HYD, &GRMHD_Z4c::Prolongation_Hyd);
        }
        Add(PHY_BVAL_HYD, PROLONG_HYD, &GRMHD_Z4c::Primitives);
      }
      else
      {
        Add(PHY_BVAL_HYD, SETB_HYD, &GRMHD_Z4c::Primitives);
      }
    }

    Add(CONS2PRIM, (PHY_BVAL_HYD | Z4C_TO_ADM),
        &GRMHD_Z4c::PhysicalBoundary_Hyd);
    Add(UPDATE_SRC, CONS2PRIM, &GRMHD_Z4c::UpdateSource);

    // collect all MHD-scalar communication into blocking task ----------------
    // In principle this should not be required, there is somewhere MPI issue
    TaskID MAT_COM = SEND_HYD | RECV_HYD;
    if (multilevel)
    {
      MAT_COM = MAT_COM | SEND_HYDFLX | RECV_HYDFLX;
    }

    if (NSCALARS > 0)
    {
      MAT_COM = MAT_COM | SEND_SCLR | RECV_SCLR;
      if (multilevel)
      {
        MAT_COM = MAT_COM | SEND_SCLRFLX | RECV_SCLRFLX;
      }
    }

    if (MAGNETIC_FIELDS_ENABLED)
    {
      MAT_COM = MAT_COM | SEND_FLD | RECV_FLD | SEND_FLDFLX | RECV_FLDFLX;
    }

    // collect all MHD-scalar sources (that need old state vector) ------------
    TaskID MAT_SRC = SRCTERM_HYD;

    if (NSCALARS > 0)
    {
      // BD: TODO -
      // not currently needed; would be needed if sourced.
    }

    if (MAGNETIC_FIELDS_ENABLED)
    {
      MAT_SRC = MAT_SRC | CALC_FLDFLX;
    }

    // Z4c sub-system logic ---------------------------------------------------
    Add(CALC_Z4CRHS, NONE,        &GRMHD_Z4c::CalculateZ4cRHS);
    Add(INT_Z4C,     CALC_Z4CRHS, &GRMHD_Z4c::IntegrateZ4c);

    // Should be able to do this
    // Add(SEND_Z4C, INT_Z4C, &GRMHD_Z4c::SendZ4c);
    // Add(RECV_Z4C, NONE, &GRMHD_Z4c::ReceiveZ4c);

    // Do instead this
    Add(SEND_Z4C, (INT_Z4C | MAT_COM), &GRMHD_Z4c::SendZ4c);
    Add(RECV_Z4C, MAT_COM,             &GRMHD_Z4c::ReceiveZ4c);

    Add(SETB_Z4C, (RECV_Z4C | INT_Z4C), &GRMHD_Z4c::SetBoundariesZ4c);

    if (multilevel)
    {
      Add(PROLONG_Z4C,  SETB_Z4C, &GRMHD_Z4c::Prolongation_Z4c);
      Add(PHY_BVAL_Z4C, PROLONG_Z4C, &GRMHD_Z4c::PhysicalBoundary_Z4c);
    }
    else
    {
      Add(PHY_BVAL_Z4C, SETB_Z4C, &GRMHD_Z4c::PhysicalBoundary_Z4c);
    }

    Add(ALG_CONSTR, PHY_BVAL_Z4C, &GRMHD_Z4c::EnforceAlgConstr);

    // Force all matter requiring ADM at old step to complete first
    Add(Z4C_TO_ADM, (ALG_CONSTR | MAT_SRC), &GRMHD_Z4c::Z4cToADM);

    Add(ADM_CONSTR, UPDATE_SRC, &GRMHD_Z4c::ADM_Constraints);

    Add(Z4C_WEYL,  Z4C_TO_ADM, &GRMHD_Z4c::Z4c_Weyl);

    Add(USERWORK, ADM_CONSTR, &GRMHD_Z4c::UserWork);
    Add(NEW_DT,   USERWORK,   &GRMHD_Z4c::NewBlockTimeStep);

    if (adaptive)
    {
      Add(FLAG_AMR,     USERWORK, &GRMHD_Z4c::CheckRefinement);
      Add(CLEAR_ALLBND, FLAG_AMR, &GRMHD_Z4c::ClearAllBoundary);
    }
    else
    {
      Add(CLEAR_ALLBND, NEW_DT, &GRMHD_Z4c::ClearAllBoundary);
    }

  }
  #endif // DBG_USE_CONS_BC
#endif  // DBG_USE_REFERENCE_TASKLISTS


}

// ----------------------------------------------------------------------------
void GRMHD_Z4c::StartupTaskList(MeshBlock *pmb, int stage)
{
  BoundaryValues *pb   = pmb->pbval;
  Z4c            *pz4c = pmb->pz4c;

  // Application of Sommerfeld boundary conditions
  pz4c->Z4cBoundaryRHS(pz4c->storage.u, pz4c->storage.mat, pz4c->storage.rhs);

  const Real t_end = this->t_end(stage, pmb);
  const Real dt_scaled = this->dt_scaled(stage, pmb);

  FCN_CC_CX_VC(
    pb->ApplyPhysicalBoundaries,
    pb->ApplyPhysicalCellCenteredXBoundaries,
    pb->ApplyPhysicalVertexCenteredBoundaries
  )(t_end, dt_scaled);

  if (stage == 1)
  {
    if (integrator == "ssprk5_4")
    {
      std::stringstream msg;
      msg << "### FATAL ERROR in GRMHD_Z4c::StartupTaskList\n"
          << "integrator=" << integrator << " is currently incompatible with GRMHD"
          << std::endl;
      ATHENA_ERROR(msg);
    }

    // Initialize time abscissae
    PrepareStageAbscissae(stage, pmb);

    // Initialize storage registers
    Hydro *ph = pmb->phydro;
    ph->u1.ZeroClear();

    if (MAGNETIC_FIELDS_ENABLED)
    {
      Field *pf = pmb->pfield;
      pf->b1.x1f.ZeroClear();
      pf->b1.x2f.ZeroClear();
      pf->b1.x3f.ZeroClear();
    }

    if (NSCALARS > 0)
    {
      PassiveScalars *ps = pmb->pscalars;
      ps->s1.ZeroClear();
    }

    // Auxiliary var u1 needs 0-init at the beginning of each cycle
    pz4c->storage.u1.ZeroClear();
  }

  pb->StartReceiving(BoundaryCommSubset::all);
  return;
}

//-----------------------------------------------------------------------------
// Functions to end MPI communication
TaskStatus GRMHD_Z4c::ClearAllBoundary(MeshBlock *pmb, int stage)
{
  BoundaryValues *pb = pmb->pbval;
  pb->ClearBoundary(BoundaryCommSubset::all);
  return TaskStatus::success;
}

//-----------------------------------------------------------------------------
// Functions to calculates fluxes
TaskStatus GRMHD_Z4c::CalculateHydroFlux(MeshBlock *pmb, int stage)
{

  if (stage <= nstages)
  {
    Hydro *ph = pmb->phydro;
    Field *pf = pmb->pfield;
    Reconstruction * pr = pmb->precon;

    int xorder = pmb->precon->xorder;

    if ((stage == 1) && (integrator == "vl2"))
    {
      xorder = 1;
    }

    ph->CalculateFluxes(ph->w, pf->b, pf->bcc, xorder);

    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

//-----------------------------------------------------------------------------
TaskStatus GRMHD_Z4c::CalculateEMF(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    Hydro *ph = pmb->phydro;
    Field *pf = pmb->pfield;

    pf->ComputeCornerE(ph->w, pf->bcc);
    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

//-----------------------------------------------------------------------------
// Communicate fluxes between MeshBlocks for flux correction with AMR
TaskStatus GRMHD_Z4c::SendFluxCorrectionHydro(MeshBlock *pmb, int stage)
{
  Hydro *ph = pmb->phydro;
  ph->hbvar.SendFluxCorrection();
  return TaskStatus::success;
}

TaskStatus GRMHD_Z4c::SendFluxCorrectionEMF(MeshBlock *pmb, int stage)
{
  Field *pf = pmb->pfield;
  pf->fbvar.SendFluxCorrection();
  return TaskStatus::success;
}

//-----------------------------------------------------------------------------
// Functions to receive fluxes between MeshBlocks
TaskStatus GRMHD_Z4c::ReceiveAndCorrectHydroFlux(MeshBlock *pmb, int stage)
{
  Hydro * ph = pmb->phydro;

  if (ph->hbvar.ReceiveFluxCorrection())
  {
    return TaskStatus::next;
  }
  else
  {
    return TaskStatus::fail;
  }
}

TaskStatus GRMHD_Z4c::ReceiveAndCorrectEMF(MeshBlock *pmb, int stage)
{
  Field * pf = pmb->pfield;

  if (pf->fbvar.ReceiveFluxCorrection())
  {
    return TaskStatus::next;
  } else {
    return TaskStatus::fail;
  }
}

//-----------------------------------------------------------------------------
// Functions to integrate conserved variables
TaskStatus GRMHD_Z4c::IntegrateHydro(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    Hydro *ph = pmb->phydro;
    PassiveScalars *ps = pmb->pscalars;
    Field *pf = pmb->pfield;

    // See IntegrateField
    Real ave_wghts[3];
    ave_wghts[0] = 1.0;
    ave_wghts[1] = stage_wghts[stage-1].delta;
    ave_wghts[2] = 0.0;
    pmb->WeightedAveCC(ph->u1, ph->u, ph->u2, ave_wghts);

    ave_wghts[0] = stage_wghts[stage-1].gamma_1;
    ave_wghts[1] = stage_wghts[stage-1].gamma_2;
    ave_wghts[2] = stage_wghts[stage-1].gamma_3;

    if (ave_wghts[0] == 0.0 && ave_wghts[1] == 1.0 && ave_wghts[2] == 0.0)
    {
      ph->u.SwapAthenaArray(ph->u1);
    }
    else
    {
      pmb->WeightedAveCC(ph->u, ph->u1, ph->u2, ave_wghts);
    }

    const Real dt_scaled = this->dt_scaled(stage, pmb);
    ph->AddFluxDivergence(dt_scaled, ph->u);

    return TaskStatus::next;
  }
  return TaskStatus::fail;
}


TaskStatus GRMHD_Z4c::IntegrateField(MeshBlock *pmb, int stage)
{
  Field *pf = pmb->pfield;

  if (stage <= nstages)
  {
    Real ave_wghts[3];
    ave_wghts[0] = 1.0;
    ave_wghts[1] = stage_wghts[stage-1].delta;
    ave_wghts[2] = 0.0;
    pmb->WeightedAveFC(pf->b1, pf->b, pf->b2, ave_wghts);

    ave_wghts[0] = stage_wghts[stage-1].gamma_1;
    ave_wghts[1] = stage_wghts[stage-1].gamma_2;
    ave_wghts[2] = stage_wghts[stage-1].gamma_3;

    if (ave_wghts[0] == 0.0 && ave_wghts[1] == 1.0 && ave_wghts[2] == 0.0)
    {
      pf->b.x1f.SwapAthenaArray(pf->b1.x1f);
      pf->b.x2f.SwapAthenaArray(pf->b1.x2f);
      pf->b.x3f.SwapAthenaArray(pf->b1.x3f);
    }
    else
    {
      pmb->WeightedAveFC(pf->b, pf->b1, pf->b2, ave_wghts);
    }

    const Real dt_scaled = this->dt_scaled(stage, pmb);
    pf->CT(dt_scaled, pf->b);

    return TaskStatus::next;
  }

  return TaskStatus::fail;
}

//-----------------------------------------------------------------------------
// Functions to add source terms
TaskStatus GRMHD_Z4c::AddSourceTermsHydro(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    Hydro * ph = pmb->phydro;
    PassiveScalars *ps = pmb->pscalars;
    Field * pf = pmb->pfield;
    Coordinates * pc = pmb->pcoord;

    const Real dt_scaled = this->dt_scaled(stage, pmb);

    // add coordinate (geometric) source terms
#if USETM
    pc->AddCoordTermsDivergence(dt_scaled, ph->flux, ph->w,
                                ps->r, pf->bcc, ph->u);
#else
    pc->AddCoordTermsDivergence(dt_scaled, ph->flux, ph->w, pf->bcc, ph->u);
#endif

    return TaskStatus::next;
  }

  return TaskStatus::fail;
}


//-----------------------------------------------------------------------------
// Functions to communicate conserved variables between MeshBlocks
TaskStatus GRMHD_Z4c::SendHydro(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    Hydro * ph = pmb->phydro;

    // Swap Hydro quantity in BoundaryVariable interface back to conserved var
    // formulation (also needed in SetBoundariesHydro(), since the tasks are
    // independent)
#ifndef DBG_USE_CONS_BC
    ph->hbvar.SwapHydroQuantity(ph->u, HydroBoundaryQuantity::cons);
#endif  // DBG_USE_CONS_BC
    ph->hbvar.SendBoundaryBuffers();

    return TaskStatus::success;
  }

  return TaskStatus::fail;
}

TaskStatus GRMHD_Z4c::SendField(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    Field * pf = pmb->pfield;

    pf->fbvar.SendBoundaryBuffers();
    return TaskStatus::success;
  }

  return TaskStatus::fail;
}

//-----------------------------------------------------------------------------
// Functions to receive conserved variables between MeshBlocks
TaskStatus GRMHD_Z4c::ReceiveHydro(MeshBlock *pmb, int stage)
{
  bool ret;
  if (stage <= nstages)
  {
    ret = pmb->phydro->hbvar.ReceiveBoundaryBuffers();
  }
  else
  {
    return TaskStatus::fail;
  }

  if (ret) {
    return TaskStatus::success;
  } else {
    return TaskStatus::fail;
  }
}


TaskStatus GRMHD_Z4c::ReceiveField(MeshBlock *pmb, int stage)
{
  bool ret;
  if (stage <= nstages) {
    ret = pmb->pfield->fbvar.ReceiveBoundaryBuffers();
  } else {
    return TaskStatus::fail;
  }
  if (ret) {
    return TaskStatus::success;
  } else {
    return TaskStatus::fail;
  }
}

TaskStatus GRMHD_Z4c::SetBoundariesHydro(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    Hydro * ph = pmb->phydro;

#ifndef DBG_USE_CONS_BC
    ph->hbvar.SwapHydroQuantity(ph->u, HydroBoundaryQuantity::cons);
#endif  // DBG_USE_CONS_BC
    ph->hbvar.SetBoundaries();
    return TaskStatus::success;
  }
  return TaskStatus::fail;
}


TaskStatus GRMHD_Z4c::SetBoundariesField(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    Field * pf = pmb->pfield;

    pf->fbvar.SetBoundaries();
    return TaskStatus::success;
  }
  return TaskStatus::fail;
}


//-----------------------------------------------------------------------------
// Functions for everything else
TaskStatus GRMHD_Z4c::Prolongation_Hyd(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    BoundaryValues * pb = pmb->pbval;

    const Real t_end = this->t_end(stage, pmb);
    const Real dt_scaled = this->dt_scaled(stage, pmb);

    pb->ProlongateHydroBoundaries(t_end, dt_scaled);

    return TaskStatus::success;
  }

  return TaskStatus::fail;
}


TaskStatus GRMHD_Z4c::Primitives(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    Hydro *ph = pmb->phydro;
    Field *pf = pmb->pfield;
    PassiveScalars *ps = pmb->pscalars;
    BoundaryValues *pb = pmb->pbval;

    int il = pmb->is, iu = pmb->ie;
    int jl = pmb->js, ju = pmb->je;
    int kl = pmb->ks, ku = pmb->ke;

    if (pb->nblevel[1][1][0] != -1) il -= NGHOST;
    if (pb->nblevel[1][1][2] != -1) iu += NGHOST;
    if (pb->nblevel[1][0][1] != -1) jl -= NGHOST;
    if (pb->nblevel[1][2][1] != -1) ju += NGHOST;
    if (pb->nblevel[0][1][1] != -1) kl -= NGHOST;
    if (pb->nblevel[2][1][1] != -1) ku += NGHOST;

#ifdef DBG_USE_CONS_BC
    il = 0;
    iu = pmb->ncells1-1;
    jl = 0;
    ju = pmb->ncells2-1;
    kl = 0;
    ku = pmb->ncells3-1;
#endif // DBG_USE_CONS_BC

    // At beginning of this task, ph->w contains previous stage's W(U) output
    // and ph->w1 is used as a register to store the current stage's output.
    // For the second order integrators VL2 and RK2, the prim_old initial guess for the
    // Newton-Raphson solver in GR EOS uses the following abscissae:
    // stage=1: W at t^n and
    // stage=2: W at t^{n+1/2} (VL2) or t^{n+1} (RK2)
    pmb->peos->ConservedToPrimitive(ph->u, ph->w, pf->b, ph->w1,
#if USETM
                                    ps->s, ps->r,
#endif
                                    pf->bcc, pmb->pcoord,
                                    il, iu, jl, ju, kl, ku,0);
#if !USETM
    if (NSCALARS > 0)
    {
      // r1/r_old for GR is currently unused:
      pmb->peos->PassiveScalarConservedToPrimitive(ps->s, ph->w1, // ph->u, (updated rho)
                                                   ps->r, ps->r,
                                                   pmb->pcoord, il, iu, jl, ju, kl, ku);
    }
#endif

    // swap AthenaArray data pointers so that w now contains the updated w_out
    ph->w.SwapAthenaArray(ph->w1);
    // r1/r_old for GR is currently unused:
    // ps->r.SwapAthenaArray(ps->r1);

    return TaskStatus::success;
  }

  return TaskStatus::fail;
}


TaskStatus GRMHD_Z4c::PhysicalBoundary_Hyd(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    Hydro *ph = pmb->phydro;
    PassiveScalars *ps = pmb->pscalars;
    BoundaryValues *pbval = pmb->pbval;

    const Real t_end = this->t_end(stage, pmb);
    const Real dt_scaled = this->dt_scaled(stage, pmb);

    // Swap Hydro and (possibly) passive scalar quantities in BoundaryVariable
    // interface from conserved to primitive formulations:
#ifndef DBG_USE_CONS_BC
    ph->hbvar.SwapHydroQuantity(ph->w, HydroBoundaryQuantity::prim);
    if (NSCALARS > 0)
    {
      ps->sbvar.var_cc = &(ps->r);
    }
#endif // DBG_USE_CONS_BC

    pbval->ApplyPhysicalBoundaries(t_end, dt_scaled);

    return TaskStatus::success;
  }

  return TaskStatus::fail;
}


TaskStatus GRMHD_Z4c::UserWork(MeshBlock *pmb, int stage)
{
  if (stage != nstages) return TaskStatus::success; // only do on last stage

  pmb->UserWorkInLoop();

  // TODO: BD- this should be shifted to its own task
  pmb->ptracker_extrema_loc->TreatCentreIfLocalMember();

  return TaskStatus::success;
}


TaskStatus GRMHD_Z4c::CheckRefinement(MeshBlock *pmb, int stage)
{
  if (stage != nstages) return TaskStatus::success; // only do on last stage

  pmb->pmr->CheckRefinementCondition();
  return TaskStatus::success;
}


TaskStatus GRMHD_Z4c::CalculateScalarFlux(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    PassiveScalars *ps = pmb->pscalars;

    if ((stage == 1) && (integrator == "vl2"))
    {
      ps->CalculateFluxes(ps->r, 1);
      return TaskStatus::next;
    }
    else
    {
      ps->CalculateFluxes(ps->r, pmb->precon->xorder);
      return TaskStatus::next;
    }
  }
  return TaskStatus::fail;
}


TaskStatus GRMHD_Z4c::SendScalarFlux(MeshBlock *pmb, int stage)
{
  PassiveScalars * ps = pmb->pscalars;

  ps->sbvar.SendFluxCorrection();
  return TaskStatus::success;
}


TaskStatus GRMHD_Z4c::ReceiveScalarFlux(MeshBlock *pmb, int stage)
{
  PassiveScalars * ps = pmb->pscalars;

  if (ps->sbvar.ReceiveFluxCorrection())
  {
    return TaskStatus::next;
  }
  else
  {
    return TaskStatus::fail;
  }
}


TaskStatus GRMHD_Z4c::IntegrateScalars(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    PassiveScalars * ps = pmb->pscalars;

    // This time-integrator-specific averaging operation logic is identical to
    // IntegrateHydro, IntegrateField
    Real ave_wghts[3];
    ave_wghts[0] = 1.0;
    ave_wghts[1] = stage_wghts[stage-1].delta;
    ave_wghts[2] = 0.0;
    pmb->WeightedAveCC(ps->s1, ps->s, ps->s2, ave_wghts);

    ave_wghts[0] = stage_wghts[stage-1].gamma_1;
    ave_wghts[1] = stage_wghts[stage-1].gamma_2;
    ave_wghts[2] = stage_wghts[stage-1].gamma_3;

    if (ave_wghts[0] == 0.0 && ave_wghts[1] == 1.0 && ave_wghts[2] == 0.0)
    {
      ps->s.SwapAthenaArray(ps->s1);
    }
    else
    {
      pmb->WeightedAveCC(ps->s, ps->s1, ps->s2, ave_wghts);
    }

    const Real dt_scaled = this->dt_scaled(stage, pmb);
    ps->AddFluxDivergence(dt_scaled, ps->s);

    return TaskStatus::next;
  }
  return TaskStatus::fail;
}


TaskStatus GRMHD_Z4c::SendScalars(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    PassiveScalars * ps = pmb->pscalars;

    // Swap PassiveScalars quantity in BoundaryVariable interface back to
    // conserved var formulation (also needed in SetBoundariesScalars() since
    // the tasks are independent)
#ifndef DBG_USE_CONS_BC
    ps->sbvar.var_cc = &(ps->s);
#endif // DBG_USE_CONS_BC
    ps->sbvar.SendBoundaryBuffers();
  }
  else
  {
    return TaskStatus::fail;
  }
  return TaskStatus::success;
}


TaskStatus GRMHD_Z4c::ReceiveScalars(MeshBlock *pmb, int stage)
{
  bool ret;
  if (stage <= nstages)
  {
    PassiveScalars * ps = pmb->pscalars;
    ret = ps->sbvar.ReceiveBoundaryBuffers();
  }
  else
  {
    return TaskStatus::fail;
  }

  if (ret)
  {
    return TaskStatus::success;
  }
  else
  {
    return TaskStatus::fail;
  }
  return TaskStatus::success;
}


TaskStatus GRMHD_Z4c::SetBoundariesScalars(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    PassiveScalars * ps = pmb->pscalars;

    // Set PassiveScalars quantity in BoundaryVariable interface to cons var
    // formulation
#ifndef DBG_USE_CONS_BC
    ps->sbvar.var_cc = &(ps->s);
#endif // DBG_USE_CONS_BC
    ps->sbvar.SetBoundaries();
    return TaskStatus::success;
  }
  return TaskStatus::fail;
}


TaskStatus GRMHD_Z4c::CalculateZ4cRHS(MeshBlock *pmb, int stage)
{
  Z4c * pz4c = pmb->pz4c;

 // PunctureTracker: interpolate beta at puncture position before evolution
  if (stage == 1)
  {
    for (auto ptracker : pmb->pmy_mesh->pz4c_tracker)
    {
      ptracker->InterpolateShift(pmb, pz4c->storage.u);
    }
  }

  if (stage <= nstages)
  {
    pz4c->Z4cRHS(pz4c->storage.u, pz4c->storage.mat, pz4c->storage.rhs);

    // Sommerfeld boundary conditions
    pz4c->Z4cBoundaryRHS(pz4c->storage.u,
                         pz4c->storage.mat,
                         pz4c->storage.rhs);

    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

//-----------------------------------------------------------------------------
// Functions to integrate variables
TaskStatus GRMHD_Z4c::IntegrateZ4c(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    Z4c *pz4c = pmb->pz4c;
    Hydro *ph = pmb->phydro;
    Field *pf = pmb->pfield;

    // See IntegrateField
    Real ave_wghts[3];
    ave_wghts[0] = 1.0;
    ave_wghts[1] = stage_wghts[stage-1].delta;
    ave_wghts[2] = 0.0;
    pz4c->WeightedAve(pz4c->storage.u1,
                      pz4c->storage.u,
                      pz4c->storage.u2,
                      ave_wghts);

    ave_wghts[0] = stage_wghts[stage-1].gamma_1;
    ave_wghts[1] = stage_wghts[stage-1].gamma_2;
    ave_wghts[2] = stage_wghts[stage-1].gamma_3;

    // BD: TODO - why does this give a slightly different result?
    // if (ave_wghts[0] == 0.0 && ave_wghts[1] == 1.0 && ave_wghts[2] == 0.0)
    // {
    //   // pz4c->storage.u.SwapAthenaArray(pz4c->storage.u1);
    //   std::swap(pz4c->storage.u, pz4c->storage.u1);
    // }
    // else
    // {
    //   pz4c->WeightedAve(pz4c->storage.u,
    //                     pz4c->storage.u1,
    //                     pz4c->storage.u2,
    //                     ave_wghts);
    // }
    pz4c->WeightedAve(pz4c->storage.u,
                      pz4c->storage.u1,
                      pz4c->storage.u2,
                      ave_wghts);

    const Real dt_scaled = this->dt_scaled(stage, pmb);
    pz4c->AddZ4cRHS(pz4c->storage.rhs,
                    dt_scaled,
                    pz4c->storage.u);

    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

//-----------------------------------------------------------------------------
// Functions to communicate conserved variables between MeshBlocks
TaskStatus GRMHD_Z4c::SendZ4c(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    Z4c *pz4c = pmb->pz4c;

    pz4c->ubvar.SendBoundaryBuffers();
    return TaskStatus::success;
  }
  return TaskStatus::fail;
}

//-----------------------------------------------------------------------------
// Functions to receive conserved variables between MeshBlocks
TaskStatus GRMHD_Z4c::ReceiveZ4c(MeshBlock *pmb, int stage)
{
  bool ret;

  if (stage <= nstages)
  {
    Z4c *pz4c = pmb->pz4c;

    ret = pz4c->ubvar.ReceiveBoundaryBuffers();
  }
  else
  {
    return TaskStatus::fail;
  }

  if (ret)
  {
    return TaskStatus::success;
  }
  else
  {
    return TaskStatus::fail;
  }
}

TaskStatus GRMHD_Z4c::SetBoundariesZ4c(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    Z4c *pz4c = pmb->pz4c;

    pz4c->ubvar.SetBoundaries();
    return TaskStatus::success;
  }
  return TaskStatus::fail;
}

//-----------------------------------------------------------------------------
TaskStatus GRMHD_Z4c::Prolongation_Z4c(MeshBlock *pmb, int stage)
{

  if (stage <= nstages)
  {
    BoundaryValues *pbval = pmb->pbval;

    const Real t_end = this->t_end(stage, pmb);
    const Real dt_scaled = this->dt_scaled(stage, pmb);

    // Prolongate z4c vars
    pbval->ProlongateBoundaries(t_end, dt_scaled);
  }
  else
  {
    return TaskStatus::fail;
  }

  return TaskStatus::success;
}

TaskStatus GRMHD_Z4c::PhysicalBoundary_Z4c(MeshBlock *pmb, int stage)
{

  if (stage <= nstages)
  {
    BoundaryValues *pbval = pmb->pbval;

    const Real t_end = this->t_end(stage, pmb);
    const Real dt_scaled = this->dt_scaled(stage, pmb);

    // switch based on sampling
    FCN_CC_CX_VC(
        pbval->ApplyPhysicalBoundaries,
        pbval->ApplyPhysicalCellCenteredXBoundaries,
        pbval->ApplyPhysicalVertexCenteredBoundaries
    )(t_end, dt_scaled);

  } else {
    return TaskStatus::fail;
  }
  return TaskStatus::success;
}

TaskStatus GRMHD_Z4c::EnforceAlgConstr(MeshBlock *pmb, int stage)
{
#ifndef DBG_ALGCONSTR_ALL
  if (stage != nstages) return TaskStatus::success; // only do on last stage
#endif // DBG_ALGCONSTR_ALL

  Z4c *pz4c = pmb->pz4c;
  pz4c->AlgConstr(pz4c->storage.u);

  return TaskStatus::success;
}

TaskStatus GRMHD_Z4c::Z4cToADM(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    Z4c *pz4c = pmb->pz4c;
    pz4c->Z4cToADM(pz4c->storage.u, pz4c->storage.adm);
    return TaskStatus::success;
  }

  return TaskStatus::fail;
}

TaskStatus GRMHD_Z4c::Z4c_Weyl(MeshBlock *pmb, int stage)
{
  if (stage != nstages) return TaskStatus::success;

  Mesh *pm   = pmb->pmy_mesh;
  Z4c  *pz4c = pmb->pz4c;

  if (trgs.IsSatisfied(TriggerVariant::Z4c_Weyl))
  {
    pmb->pz4c->Z4cWeyl(pmb->pz4c->storage.adm,
                       pmb->pz4c->storage.mat,
                       pmb->pz4c->storage.weyl);
  }

  return TaskStatus::success;
}

TaskStatus GRMHD_Z4c::ADM_Constraints(MeshBlock *pmb, int stage)
{
  if (stage != nstages) return TaskStatus::success;

  Mesh *pm   = pmb->pmy_mesh;
  Z4c  *pz4c = pmb->pz4c;


  if (trgs.IsSatisfied(TriggerVariant::Z4c_ADM_constraints))
  {
    pz4c->ADMConstraints(pz4c->storage.con, pz4c->storage.adm,
                         pz4c->storage.mat, pz4c->storage.u);

  }
  return TaskStatus::success;
}

// new dt ---------------------------------------------------------------------
TaskStatus GRMHD_Z4c::NewBlockTimeStep(MeshBlock *pmb, int stage)
{
  if (stage != nstages) return TaskStatus::success; // only do on last stage

  //NB using the Z4C version of this fn rather than fluid - potential issue?
  Z4c *pz4c = pmb->pz4c;
  pz4c->NewBlockTimeStep();
  return TaskStatus::success;
}

// Recouple ADM sources -------------------------------------------------------
TaskStatus GRMHD_Z4c::UpdateSource(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    Z4c   *pz4c = pmb->pz4c;
    Hydro *ph   = pmb->phydro;
    Field *pf   = pmb->pfield;

#if USETM
    PassiveScalars * ps = pmb->pscalars;

    pz4c->GetMatter(pz4c->storage.mat,
                    pz4c->storage.adm,
                    ph->w,
                    ps->r,
                    pf->bcc);
#else
    pz4c->GetMatter(pz4c->storage.mat, pz4c->storage.adm, ph->w, pf->bcc);
#endif

    ph->w1.Fill(NAN);
    return TaskStatus::success;
  }
  return TaskStatus::fail;
}

//
// :D
//
