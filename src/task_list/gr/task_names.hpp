#ifndef GR_TASK_NAMES_HPP_
#define GR_TASK_NAMES_HPP_

// C headers

// C++ headers

// Athena++ headers
#include "../../athena.hpp"
#include "../task_list.hpp"

namespace TaskNames::GeneralRelativity::GRMHD_Z4c {

const TaskID NONE(0);
const TaskID CLEAR_ALLBND(1);
const TaskID UPDATE_MET(2);
const TaskID UPDATE_SRC(3);

const TaskID CALC_HYDFLX(4);
const TaskID CALC_FLDFLX(5);
const TaskID CALC_RADFLX(6);
const TaskID CALC_CHMFLX(7);

const TaskID SEND_HYDFLX(8);
const TaskID SEND_FLDFLX(9);
// const TaskID SEND_RADFLX(8);
// const TaskID SEND_CHMFLX(9);

const TaskID RECV_HYDFLX(10);
const TaskID RECV_FLDFLX(11);
// const TaskID RECV_RADFLX(12);
// const TaskID RECV_CHMFLX(13);
const TaskID ALG_CONSTR(12);
const TaskID Z4C_TO_ADM(13);

const TaskID SRCTERM_HYD(14);
// const TaskID SRCTERM_FLD(15);
// const TaskID SRCTERM_RAD(16);
// const TaskID SRCTERM_CHM(17);
const TaskID ADM_CONSTR(15);
const TaskID ASSERT_FIN(16);
const TaskID Z4C_WEYL(17);

const TaskID INT_HYD(18);
const TaskID INT_FLD(19);
const TaskID WAVE_EXTR(20);
// const TaskID INT_RAD(20);
// const TaskID INT_CHM(21);

const TaskID SEND_HYD(22);
const TaskID SEND_FLD(23);
// const TaskID SEND_RAD(24);
// const TaskID SEND_CHM(25);

const TaskID RECV_HYD(26);
const TaskID RECV_FLD(27);
// const TaskID RECV_RAD(28);
// const TaskID RECV_CHM(29);

const TaskID SETB_HYD(30);
const TaskID SETB_FLD(31);
// const TaskID SETB_RAD(32);
// const TaskID SETB_CHM(33);

const TaskID PROLONG_HYD(34);
const TaskID CONS2PRIM(35);
const TaskID PHY_BVAL_HYD(36);
const TaskID USERWORK(37);
const TaskID NEW_DT(38);
const TaskID FLAG_AMR(39);

const TaskID SEND_HYDSH(40);
const TaskID SEND_EMFSH(41);
const TaskID SEND_FLDSH(42);
const TaskID RECV_HYDSH(43);
const TaskID RECV_EMFSH(44);
const TaskID RECV_FLDSH(45);
const TaskID RMAP_EMFSH(46);

const TaskID DIFFUSE_HYD(47);
const TaskID DIFFUSE_FLD(48);

const TaskID CALC_SCLRFLX(49);
const TaskID SEND_SCLRFLX(50);
const TaskID RECV_SCLRFLX(51);
const TaskID INT_SCLR(52);
const TaskID SEND_SCLR(53);
const TaskID RECV_SCLR(54);
const TaskID SETB_SCLR(55);
const TaskID DIFFUSE_SCLR(56);

const TaskID CALC_Z4CRHS(57);
const TaskID INT_Z4C(58);
const TaskID SEND_Z4C(59);
const TaskID RECV_Z4C(60);
const TaskID SETB_Z4C(61);

const TaskID PROLONG_Z4C(62);
const TaskID PHY_BVAL_Z4C(63);

}  // namespace TaskNames::GeneralRelativity::GRMHD_Z4c

namespace TaskNames::GeneralRelativity::GR_Z4c {

const TaskID NONE(0);
const TaskID CLEAR_ALLBND(1);
const TaskID CALC_Z4CRHS(2);
const TaskID INT_Z4C(3);
const TaskID SEND_Z4C(4);
const TaskID RECV_Z4C(5);
const TaskID SETB_Z4C(6);

const TaskID PROLONG(7);
const TaskID PHY_BVAL(8);

const TaskID ALG_CONSTR(9);
const TaskID Z4C_TO_ADM(10);
const TaskID USERWORK(11);
const TaskID NEW_DT(12);

const TaskID ADM_CONSTR(13);
const TaskID FLAG_AMR(14);

const TaskID ASSERT_FIN(15);
const TaskID Z4C_WEYL(16);
const TaskID WAVE_EXTR(17);
const TaskID CCE_DUMP(18);

}  // TaskNames::GeneralRelativity::GR_Z4c

namespace TaskNames::GeneralRelativity::PostAMR {

const TaskID NONE(0);
const TaskID CLEAR_ALLBND(1);

const TaskID ALG_CONSTR(5);
const TaskID Z4C_TO_ADM(6);
const TaskID UPDATE_SRC(7);
const TaskID ADM_CONSTR(8);

const TaskID Z4C_WEYL(9);

}  // namespace TaskNames::GeneralRelativity::PostAMR

#endif  // GR_TASK_NAMES_HPP_
