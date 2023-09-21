<job>
problem_id      = dummy_3d   # Basename of output filenames

<problem>


<time>
cfl_number      = 0.5       # The Courant, Friedrichs, & Lewy (CFL) number
tlim            = 0       # time limit
integrator      = rk4       # time integrator

<dummy>
# amr with spheres
sphere_zone_number = 2

# should roughly match the standard box-in-box

# sphere_zone_level_0    = 5
# sphere_zone_radius_0   = 100
# sphere_zone_puncture_0 = -1  # not pinned to a puncture
# sphere_zone_center1_0  = 0   # fix center
# sphere_zone_center2_0  = 0
# sphere_zone_center3_0  = 0

sphere_zone_level_1    = [[S_L]]
sphere_zone_radius_1   = [[S_R]]
sphere_zone_puncture_1 = -1  # not pinned to a puncture
sphere_zone_center1_1  = -[[S_AB]]   # fix center
sphere_zone_center2_1  = -[[S_AB]]
sphere_zone_center3_1  = -[[S_AB]]


sphere_zone_level_2    = [[S_L]]
sphere_zone_radius_2   = [[S_R]]
# sphere_zone_puncture_2 = 0  # pin to puncture
sphere_zone_puncture_2 = -1  # not pinned to a puncture
sphere_zone_center1_2  = [[S_AB]]   # fix center
sphere_zone_center2_2  = [[S_AB]]
sphere_zone_center3_2  = [[S_AB]]


<mesh>
refinement      = adaptive    # Static mesh refinement

nx1             = [[N_M_1]]      # Number of zones in X1-direction
x1min           = -[[X_M]]      # minimum value of X1
x1max           = [[X_M]]      # maximum value of X1
ix1_bc          = outflow  # inner-X1 boundary flag
ox1_bc          = outflow  # outer-X1 boundary flag

nx2             = [[N_M_2]]         # Number of zones in X2-direction
x2min           = -[[X_M]]      # minimum value of X2
x2max           = [[X_M]]      # maximum value of X2
ix2_bc          = outflow  # inner-X2 boundary flag
ox2_bc          = outflow  # outer-X2 boundary flag

nx3             = [[N_M_3]]         # Number of zones in X3-direction
x3min           = -[[X_M]]      # minimum value of X3
x3max           = [[X_M]]      # maximum value of X3
ix3_bc          = outflow  # inner-X3 boundary flag
ox3_bc          = outflow  # outer-X3 boundary flag

numlevel = 32  # <refinement> level + 1
num_threads = 1        # Number of OpenMP threads per process

<meshblock>
nx1        = [[N_B]]        # Number of zones per MeshBlock in X1-direction
nx2        = [[N_B]]        # Number of zones per MeshBlock in X2-direction
nx3        = [[N_B]]        # Number of zones per MeshBlock in X3-direction

num_threads = 1        # Number of OpenMP threads per process
