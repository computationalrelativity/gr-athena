"""
Defines universal physical constants and unit conversions
Significant digits/uncertainties NOT defined
Certainly not more than 3-4 sig figs since G is not known better than 4 sig figs

Sources: Google, Wikipedia, NIST
Jeff Kaplan <jeffkaplan@caltech.edu>
February 2, 2013
"""
 
#################################
# Physical constants
#
# G - Gravitational constant
# C - Speed of light in vacuum
# AMU - Atomic mass unit
# H - Plank's constant
# HBAR - Plank's constant/2pi
# K  -  Boltzmann constant
 
#################################
# Astronomical constants
#
# MSUN - Mass of the sun
# RSUN - Radius of the sun
# LSUN - Luminosity of the sun
# TSUN - Surface temperature of the sun
# M_EARTH - Mass of the earth
# R_EARTH - Radius of the earth
# AU  - Astronomical unit
# PC  - Parsec
# LYR - Light year
 
 
##===================================================================
## Meters kilograms seconds (SI)
##===================================================================
MKS_G = 6.67384e-11 # m^3 kg^-1 s^-2
MKS_C = 299792458.0 # m s^-1
MKS_AMU = 1.660538921e-27 # kg
MKS_H = 6.62606957e-34     # J s = m^2 kg / s
MKS_HBAR = 1.054571726e-34 # J s = m^2 kg / s
MKS_K = 1.3806488e-23  # J K^-1
 
MKS_MSUN = 1.98855e30  # kg
MKS_RSUN = 6.955e8     # m
MKS_LSUN = 3.933e26    # W -Watt
MKS_TSUN = 5.780e3     # K
MKS_M_EARTH = 5.972e24 # kg
MKS_R_EARTH = 6478100.0 # m
MKS_AU   = 1.49597871e11 # m
MKS_PC   = 3.08567758e16  # m
MKS_LYR  = 9.4605284e15  # M
 
#################################
# Conversion constants
#
# EV - Electron volt
# CM - Centimeter: cgs length unit
# GRAM - Gram:   cgs mass unit
# ERG - Erg:   cgs energy unit
# DYNE - Dyne: cgs force unit
# GRAM_PER_CM_CUBED: density conversion to cgs
# DYNE_PER_CM_SQUARED: pressure conversion to cgs
MKS_EV = 1.60217646-19 # J
MKS_CM = 0.01          # m
MKS_GRAM = 0.001       # kg
MKS_ERG  = 1.0e-7      # J
MKS_DYNE = 1.0e-5      # N = m  kg s^-2
MKS_GRAM_PER_CM_CUBED = 1000.0 # kg m ^-3
MKS_DYNE_PER_CM_SQUARED = 0.1 # Pa = N m^-2  = kg m^-1 s^-2
 
##===================================================================
## Centimeters grams seconds
##===================================================================
CGS_G = 6.67384e-8 # cm^3 g^-1 s^-2
CGS_C = 29979245800. # cm s^-1
CGS_AMU = 1.660538921e-24 # g
CGS_H = 6.62606957e-27     # erg s = cm^2 g / s
CGS_HBAR = 1.054571726e-27 # erg s = cm^2 g / s
CGS_K = 1.3806488e-16  # erg K^-1
 
CGS_MSUN = 1.98855e33  # g
CGS_RSUN = 6.955e10     # cm
CGS_LSUN = 3.933e33    # erg s^-1
CGS_TSUN = 5.780e3     # K
CGS_M_EARTH = 5.972e27 # g
CGS_R_EARTH = 647810000. # cm
CGS_AU   = 1.49597871e13 # cm
CGS_PC   = 3.08567758e18  # cm
CGS_LYR  = 9.4605284e17  # cm
CGS_sigmaT = 6.6524e-25 # cm^2
mp = 1.6726231e-24 # g

#################################
# Conversion constants
#
# EV - Electron volt
# M - Meter: mks length unit
# KG - Kilogram:   mks mass unit
# J - Joule:   mks energy unit
# N - Newton: mks force unit
# KG_PER_M_CUBED: density conversion to mks
# N_PER_M_SQUARED: pressure conversion to mks
CGS_EV = 1.60217646-12 # erg
CGS_M = 100.0          # cn
CGS_KG = 1000.0       # g
CGS_J  = 1.0e+7      # erg
CGS_N = 1.0e+5      # dyne = cm  g s^-2
CGS_KG_PER_M_CUBED = 0.001 # g cm ^-3
CGS_N_PER_M_SQUARED = 10.0 # ba = dyne cm^-2  = g cm^-1 s^-2
 
##===================================================================
## Augmented geometric units G=c=Msun=1
#  Units are dimensionless, except Boltzmann K which has Kelvin^-1
#  S - second     W - Watt
##===================================================================
AGEO_G = 1.0
AGEO_C = 1.0
AGEO_AMU = 8.3505012e-58
AGEO_K  = 7.7229851e-71   # K^-1
 
AGEO_MSUN = 1.0
AGEO_RSUN = 470934.0
AGEO_LSUN = 4.4659113e-16
 
#################################
# Conversion constants
AGEO_EV = 8.9621524e-67
AGEO_M = 6.77116916e-4
AGEO_CM = 6.77116916e-6
AGEO_S  = 202994.545
AGEO_KG = 5.02739933e-31
AGEO_GRAM =5.02739933e-34
AGEO_KG_PER_M_CUBED = 1.6193935e-21
AGEO_GRAM_PER_CM_CUBED = 1.6193935e-18
AGEO_N_PER_M_SQUARED = 1.80181827e-38
AGEO_DYNE_PER_CM_SQUARED = 1.80181827e-37
AGEO_J = 5.59373614e-48
AGEO_ERG = 5.59373614e-55
AGEO_W  = 1.13549741e-42
AGEO_ERGS_PER_S = 1.13549741e-49
 
#################################
# Inverse conversion constants
# Gives you a feel for what '1' is for quantities in AGEO units
AGEO_LENGTH_IN_M = 1.47684983e3  # m
AGEO_LENGTH_IN_CM = 1.47684983e5  # cm Approx 1.5 km
AGEO_TIME_IN_S = 4.92624076e-3   # s   Approx 5 ms
AGEO_DENSITY_IN_MKS = 6.17515138e20  # kg m^-3
AGEO_DENSITY_IN_CGS = 6.17515138e17  # g cm^-3  Approx 6000x nuclear density
AGEO_PRESSURE_IN_MKS = 5.54994928e37 # Pa
AGEO_PRESSURE_IN_CGS = 5.54994928e36 # ba = dyne cm^-2
AGEO_ENERGY_IN_MKS = 1.78771393e47 # Joules
AGEO_ENERGY_IN_CGS = 1.78771393e54 # ergs
AGEO_POWER_IN_MKS  = 8.80671314e41 # W
AGEO_POWER_IN_CGS  = 8.80671314e48 # erg s^-1
