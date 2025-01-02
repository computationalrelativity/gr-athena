module quintic_hermite
  implicit none
  private
  public :: psi0, dpsi0, psi1, dpsi1, psi2, dpsi2, ddpsi0, ddpsi1, ddpsi2, h5, h3

contains

  function psi0(z) result(res)
    real(8), intent(in) :: z
    real(8) :: res
    res = z**3 * (z * (-6.0d0*z + 15.0d0) - 10.0d0) + 1.0d0
  end function psi0

  function dpsi0(z) result(res)
    real(8), intent(in) :: z
    real(8) :: res
    res = z**2 * (z * (-30.0d0*z + 60.0d0) - 30.0d0)
  end function dpsi0

  function psi1(z) result(res)

    real(8), intent(in) :: z
    real(8) :: res
    res = z * (z**2 * (z * (-3.0d0*z + 8.0d0) - 6.0d0) + 1.0d0)
  end function psi1

  function dpsi1(z) result(res)
    real(8), intent(in) :: z
    real(8) :: res
    res = z*z * (z * (-15.0d0*z + 32.0d0) - 18.0d0) + 1.0d0
  end function dpsi1

  function psi2(z) result(res)
    real(8), intent(in) :: z
    real(8) :: res
    res = 0.5d0 * z*z * (z * (z * (-z + 3.0d0) - 3.0d0) + 1.0d0)
  end function psi2

  function dpsi2(z) result(res)
    real(8), intent(in) :: z
    real(8) :: res
    res = 0.5d0 * z * (z * (z * (-5.0d0*z + 12.0d0) - 9.0d0) + 2.0d0)
  end function dpsi2

  function ddpsi0(z) result(res)
    real(8), intent(in) :: z
    real(8) :: res
    res = z* ( z*( -120.0d0*z + 180.0d0) -60.0d0)
  end function ddpsi0

  function ddpsi1(z) result(res)
    real(8), intent(in) :: z
    real(8) :: res
    res = z * (z * (-60.0d0*z + 96.0d0) -36.0d0)
  end function ddpsi1

  function ddpsi2(z) result(res)
    real(8), intent(in) :: z
    real(8) :: res
    res = 0.5d0 * (z * (z * (-20.0d0*z + 36.0d0) - 18.0d0) + 2.0d0)
  end function ddpsi2

  function h5(fi, w0t, w1t, w2t, w0mt, w1mt, w2mt, w0d, w1d, w2d, w0md, w1md, w2md) result(res)
    real(8), intent(in) :: fi(36)
    real(8), intent(in) :: w0t, w1t, w2t, w0mt, w1mt, w2mt
    real(8), intent(in) :: w0d, w1d, w2d, w0md, w1md, w2md
    real(8) :: res

    res = fi(1)  * w0d*w0t   + fi(2)  * w0md*w0t &
        + fi(3)  * w0d*w0mt  + fi(4)  * w0md*w0mt &
        + fi(5)  * w0d*w1t   + fi(6)  * w0md*w1t &
        + fi(7)  * w0d*w1mt  + fi(8)  * w0md*w1mt &
        + fi(9)  * w0d*w2t   + fi(10) * w0md*w2t &
        + fi(11) * w0d*w2mt  + fi(12) * w0md*w2mt &
        + fi(13) * w1d*w0t   + fi(14) * w1md*w0t &
        + fi(15) * w1d*w0mt  + fi(16) * w1md*w0mt &
        + fi(17) * w2d*w0t   + fi(18) * w2md*w0t &
        + fi(19) * w2d*w0mt  + fi(20) * w2md*w0mt &
        + fi(21) * w1d*w1t   + fi(22) * w1md*w1t &
        + fi(23) * w1d*w1mt  + fi(24) * w1md*w1mt &
        + fi(25) * w2d*w1t   + fi(26) * w2md*w1t &
        + fi(27) * w2d*w1mt  + fi(28) * w2md*w1mt &
        + fi(29) * w1d*w2t   + fi(30) * w1md*w2t &
        + fi(31) * w1d*w2mt  + fi(32) * w1md*w2mt &
        + fi(33) * w2d*w2t   + fi(34) * w2md*w2t &
        + fi(35) * w2d*w2mt  + fi(36) * w2md*w2mt
  end function h5

  function h3(fi, w0t,w1t,w0mt,w1mt,w0d,w1d,w0md,w1md) result(res)
    real(8), intent(in) :: fi(36)
    real(8), intent(in) :: w0t, w1t, w0mt, w1mt
    real(8), intent(in) :: w0d, w1d, w0md, w1md
    real(8) :: res

    res = fi(1)  * w0d*w0t   +  fi(2)  * w0md*w0t &
        + fi(3)  * w0d*w0mt  +  fi(4)  * w0md*w0mt &
        + fi(5)  * w0d*w1t   +  fi(6)  * w0md*w1t &
        + fi(7)  * w0d*w1mt  +  fi(8)  * w0md*w1mt &
        + fi(9)  * w1d*w0t   +  fi(10) * w1md*w0t &
        + fi(11) * w1d*w0mt  +  fi(12) * w1md*w0mt &
        + fi(13) * w1d*w1t   +  fi(14) * w1md*w1t &
        + fi(15) * w1d*w1mt  +  fi(16) * w1md*w1mt
  end function h3
 
end module quintic_hermite
