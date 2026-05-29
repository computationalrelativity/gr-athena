#ifndef DRIFT_CONTROL_HPP
#define DRIFT_CONTROL_HPP

//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file drift_control.hpp
//  \brief definitions for the DriftControl class

#include <string>

#include "../athena.hpp"

class Mesh;
class ParameterInput;
class PunctureTracker;
class ExtremaTracker;

//! \class DriftControl
//! \brief Prevents coordinate drift of a fixed point during evolution.
//!
//! Tracks the position of a fixed point (e.g. post-merger remnant centre)
//! via a PunctureTracker or ExtremaTracker and adds a damped-oscillator
//! restoring force to the shift vector RHS to pull it back toward the
//! desired origin.
class DriftControl {
 public:
  enum class TrackerType { Puncture, Extrema };

  DriftControl(Mesh *pmesh, ParameterInput *pin);
  ~DriftControl() = default;

  void Evolve();

  Real GetPos(int a) const { return dc_pos[a]; }
  Real GetVel(int a) const { return dc_vel[a]; }

 private:
  Mesh const *pmesh;
  TrackerType dc_tracker_type;
  int dc_tracker_index;
  Real dc_pos[NDIM];
  Real dc_pos_old[NDIM];
  Real dc_vel[NDIM];
  bool dc_first_step;
  Real dc_vel_cap;
};

#endif
