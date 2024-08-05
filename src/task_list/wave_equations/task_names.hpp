#ifndef WAVE_EQUATIONS_TASK_NAMES_HPP_
#define WAVE_EQUATIONS_TASK_NAMES_HPP_

// C headers

// C++ headers

// Athena++ headers
#include "../../athena.hpp"
#include "../task_list.hpp"

namespace TaskNames::WaveEquations::WE_2O {

const TaskID NONE(0);
const TaskID CLEAR_ALLBND(1);

const TaskID CALC_WAVERHS(2);
const TaskID INT_WAVE(3);
const TaskID SEND_WAVE(4);
const TaskID RECV_WAVE(5);
const TaskID SETB_WAVE(6);
const TaskID PROLONG(7);
const TaskID PHY_BVAL(8);
const TaskID USERWORK(9);
const TaskID NEW_DT(10);
const TaskID FLAG_AMR(11);

}  // TaskNames::GeneralRelativity::GR_Z4c

#endif  // WAVE_EQUATIONS_TASK_NAMES_HPP_
