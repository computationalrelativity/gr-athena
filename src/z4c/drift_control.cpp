//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file drift_control.cpp
//  \brief implementation of the DriftControl class

#include <algorithm>
#include <cstring>
#include <sstream>

#include "../athena.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../trackers/extrema_tracker.hpp"
#include "drift_control.hpp"
#include "puncture_tracker.hpp"

DriftControl::DriftControl(Mesh* pmesh, ParameterInput* pin)
    : pmesh(pmesh)
{
  std::string const type_str =
      pin->GetOrAddString("z4c", "dc_tracker_type", "puncture");

  if (type_str == "extrema") {
    dc_tracker_type = TrackerType::Extrema;
  } else if (type_str == "puncture") {
    dc_tracker_type = TrackerType::Puncture;
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in DriftControl constructor" << std::endl
        << "Unknown dc_tracker_type '" << type_str
        << "'. Valid options: 'puncture', 'extrema'.";
    throw std::runtime_error(msg.str());
  }

  std::string const variety_str =
      pin->GetOrAddString("z4c", "dc_variety", "oscillator");

  if (variety_str == "pid") {
    dc_variety = Variety::PID;
  } else if (variety_str == "relaxation") {
    dc_variety = Variety::Relaxation;
  } else if (variety_str == "oscillator") {
    dc_variety = Variety::Oscillator;
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in DriftControl constructor" << std::endl
        << "Unknown dc_variety '" << variety_str
        << "'. Valid options: 'oscillator', 'pid', 'relaxation'.";
    throw std::runtime_error(msg.str());
  }

  dc_tracker_index = pin->GetOrAddInteger("z4c", "dc_tracker_index", 0);
  dc_first_step    = true;
  dc_vel_cap       = pin->GetOrAddReal("z4c", "dc_vel_cap", 1.0);
  dc_integral_cap  = pin->GetOrAddReal("z4c", "dc_integral_cap", 5.0);

  dc_fixed[0]      = pin->GetOrAddReal("z4c", "dc_fixed_x", 0.0);
  dc_fixed[1]      = pin->GetOrAddReal("z4c", "dc_fixed_y", 0.0);
  dc_fixed[2]      = pin->GetOrAddReal("z4c", "dc_fixed_z", 0.0);

  for (int a = 0; a < NDIM; ++a) {
    dc_pos[a]       = 0.0;
    dc_pos_old[a]   = 0.0;
    dc_vel[a]       = 0.0;
    dc_integral[a]  = 0.0;
    dc_prev_error[a]= 0.0;
  }
}

void DriftControl::Evolve()
{
  Real tracker_pos[NDIM] = {0.0, 0.0, 0.0};

  if (dc_tracker_type == TrackerType::Puncture) {
    int const idx = dc_tracker_index;
    if (idx < 0 ||
        static_cast<std::size_t>(idx) >= pmesh->pz4c_tracker.size()) {
      std::stringstream msg;
      msg << "### FATAL ERROR in DriftControl::Evolve" << std::endl
          << "dc_tracker_index " << idx << " is out of range for "
          << pmesh->pz4c_tracker.size() << " PunctureTrackers.";
      throw std::runtime_error(msg.str());
    }
    PunctureTracker* pt = pmesh->pz4c_tracker[idx];
    for (int a = 0; a < NDIM; ++a) {
      tracker_pos[a] = pt->GetPos(a);
    }
  } else {
    ExtremaTracker* et = pmesh->ptracker_extrema;
    int const idx = dc_tracker_index;
    if (et == nullptr || idx < 0 || idx >= et->N_tracker) {
      std::stringstream msg;
      msg << "### FATAL ERROR in DriftControl::Evolve" << std::endl
          << "dc_tracker_index " << idx << " is out of range for "
          << (et ? et->N_tracker : 0) << " ExtremaTracker slots, or "
          << "ptracker_extrema is null.";
      throw std::runtime_error(msg.str());
    }
    if constexpr (NDIM >= 1) tracker_pos[0] = et->c_x1(idx);
    if constexpr (NDIM >= 2) tracker_pos[1] = et->c_x2(idx);
    if constexpr (NDIM >= 3) tracker_pos[2] = et->c_x3(idx);
  }

  for (int a = 0; a < NDIM; ++a) {
    dc_pos_old[a] = dc_pos[a];
    dc_pos[a]     = tracker_pos[a];
  }

  if (dc_first_step) {
    for (int a = 0; a < NDIM; ++a) {
      dc_vel[a]      = 0.0;
      dc_integral[a] = 0.0;
      dc_prev_error[a] = dc_pos[a] - dc_fixed[a];
    }
    dc_first_step = false;
  } else if (dc_variety == Variety::PID) {
    for (int a = 0; a < NDIM; ++a) {
      Real const e = dc_pos[a] - dc_fixed[a];
      dc_integral[a] += e * pmesh->dt;
      dc_integral[a] = std::clamp(dc_integral[a],
                                  -dc_integral_cap, dc_integral_cap);
      Real const vel_raw = (e - dc_prev_error[a]) / pmesh->dt;
      dc_vel[a] = std::clamp(vel_raw, -dc_vel_cap, dc_vel_cap);
      dc_prev_error[a] = e;
    }
  } else {
    for (int a = 0; a < NDIM; ++a) {
      Real const vel_raw = (dc_pos[a] - dc_pos_old[a]) / pmesh->dt;
      dc_vel[a] = std::clamp(vel_raw, -dc_vel_cap, dc_vel_cap);
    }
  }
}
