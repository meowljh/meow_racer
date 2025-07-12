'''
Racing environment configuration file'''
import pdb

from dataclasses import dataclass
from numpy import array
import sys 
sys.path.insert(0, '/home/ryuhhh/dev/apps/vti_poc/src/vti_poc/mpcc/controller_side/')
from common.readDataFcn import getTrack

@dataclass
class Environment_Parameters():
    # track_filename : str = 'CL_NoElev_rel_coord_Nam-C_new_880.txt'
    # track_filename : str = 'CL_NoElev_rel_coord_Nam-C_new_880_kappa_smooth.txt'
    # track_filename : str = 'CL_NoElev_Nam-C_20241202.txt'
    track_filename : str = 'CL_NoElev_Nam-C_20241211.txt'
    # track_filename : str = 'CL_NoElev_rel_coord_Nam-C_new_880_kappa_smooth_mf.txt'
    track_width : float = 8
    
@dataclass
class Vehicle_Parameters_NE():

    # Static params
    m   : float = 1840
    Iz  : float = 3966.731
    lf  : float = 1.571
    lr  : float = 1.429
    
    # Lateral dynamics
    Br  : float = 15.0693841577676
    Cr  : float = 1.44323056767426
    Dr  : float = 9326.44447146823

    CLr : float = 198686.5
    CLf : float = 166478.6046833368

    Bf  : float = 13.163204541541
    Cf  : float = 1.45027155063759
    Df  : float = 9032.91975562235
    
    alpha_f_max: float = 0.1411
    alpha_r_max: float = 0.1311

    # Longitudinal dynamics
    rw : float = 0.36875
    mu = 0.9
    gravity = 9.81

    rolling_coef_fx = 0.00685303788632066 
    rolling_coef_rx = 0.00364373791962774

    acados_tire_force_model : str = "pacejka_R" # "pacejka_R"
    acados_tire_force_eps : float = 2.0
    acados_avoid_singular : str = "sqrt"
    
    gear_ratio = 10.65
    
@dataclass
class Vehicle_Parameters_JW():

    # Static params
    m   : float = 1840 # 1985
    Iz  : float = 3837.368
    lf  : float = 1.419
    lr  : float = 1.481
    gear_ratio : float = 10.65
    
    # Lateral dynamics
    Br  : float = 11.128742028254646
    Cr  : float = 1.975900152498267
    Dr  : float = 9009.643015277969

    CLr : float = 198686.5
    CLf : float = 166478.6046833368

    Bf  : float = 11.731124202829513
    Cf  : float = 1.672883132833314
    Df  : float = 9427.441324099917

    alpha_f_max: float = 0.1411
    alpha_r_max: float = 0.1311

    # Longitudinal dynamics
    rw : float = 0.369
    mu = 1.0
    gravity = 9.81

    rolling_coef_fx = 0.006724669206530 
    rolling_coef_rx = 0.002439521729772

    acados_tire_force_model : str = "pacejka_R"
    acados_tire_force_eps : float = 2.0
    acados_avoid_singular : str = "sqrt"

    

@dataclass
class PP_MPCC_Params():
    
    # Constraints
    
    # Model bounds
    track_width : float = 3.5             # half

    vx_min : float = 30/3.6
    vx_max : float = 200/3.6 #50 
    
    # input bounds
    tau_min : float = -18000.0 # -7000.0
    tau_max : float =  3727.5 # 350 * 10.65(gear ratio)
    delta_min : float = -0.4#-0.3 # [rad]
    delta_max : float = 0.4# 0.3

    # control inputs bounds
    dtau_min : float = tau_min*5 # -9000 #-8000.0
    dtau_max : float = tau_max*5 # 9000 # 8000.0
    ddelta_min : float = delta_min*5 # -1.0
    ddelta_max : float =  delta_max*5 #1.0

    # constraints
    alpha_max : float = 0.2
    alpha_min : float = -0.2

    # [x] : Need to confirm values of lateral & longitudinal forces
    # Ffy_min : float = -9000*1.1#-7000
    # Ffy_max : float = 9000*1.1# 7000
    # Fry_min : float = -9000*1.1#-7000
    # Fry_max : float = 9000*1.1# 7000

    # Ffx_min : float = -7000*1.1#-7000#-6000
    # Ffx_max : float = 100*1.1#100# 1000
    # Frx_min : float = -5000*1.1#-5000#-5000
    # Frx_max : float = 8500*1.1#7000# 5000
    

    ### trackbounds, Ffy, Fry, Ffx, Frx, fmax(tau_r,0) - drivetrqmax, alpha_f, alpha_r
    # zl : array = array([1e-5, 1e-5, 1e-5, 1e-5, 1e3, 1e3])
    # zu : array = array([1e-5, 1e-5, 1e-5, 1e-5, 1e3, 1e3])
    # Zl : array = array([1e-5, 1e-5, 1e-5, 1e-5, 1e3, 1e3])
    # Zu : array = array([1e-5, 1e-5, 1e-5, 1e-5, 1e3, 1e3])
    # zl : array = array([1e3, 1e3])
    # zu : array = array([1e3, 1e3])
    # Zl : array = array([1e3, 1e3])
    # Zu : array = array([1e3, 1e3])
    
    zl : array = array([1e2, 1e2])
    zu : array = array([1e2, 1e2])
    Zl : array = array([1e2, 1e2])
    Zu : array = array([1e2, 1e2])
    

    [track_sref,track_xref,track_yref,track_psiref,track_kapparef,_,_] = getTrack(Environment_Parameters.track_filename)
    
    # x = vertcat(theta, e_c, e_phi, vx, vy, omega, tau_r, delta)
    # init_point = 45045
    init_point = 0 # 1100 #100
    
    vx_init = 180/3.6 #120/3.6
    initial_state : array = array(
        [init_point, 0.0, 0.0, vx_init, 0.0, 0.0, 0.0, 0.0]
        ) 
    
    initial_state_cartesian : array = array(
        [track_xref[init_point], track_yref[init_point], track_psiref[init_point], vx_init, 0.0, 0.0, 0.0, 0.0]
    )
    
    integrator_type : str = "RK45"  # ["RK4", "RK45", "RK23", "DOP853", "radau", "BDF", "LSODA"]

    ### long horizon length 
    ref_horizon_length : float = 300.0 #200 ~ 400
    N  : int   = 25    # 100 ~ 150
    dt : float = 0.2 # 0.1
    Tf : float = N*dt



    # # aggressive
    # mode__ = 'agg'
    qe_theta = 650
    qe_c = 10**5


    # moderate
    # mode__ = 'mod'
    # qe_theta = 400
    # qe_c = 10**6.5
    
    
    # conservative
    # mode__ = 'consv'
    # qe_theta = 200
    # qe_c = 10**7
    
    # qe_theta = 700
    # qe_c = 10**5

    qtau = 1.5e0     # 1e3~1e5
    qdelta = 5e4    # 5e3~5e4
    
    qdtau = 1e-4    # 1e-10~1e-8
    qddelta = 1e9   # 1e3~1e5
    
        
    # # # driver model + highly conservative - inje_track.txt
    # qe_theta = 2e3
    # qe_c = 9e5      # 1e2 ~ 1e5

    # qtau = 1e0     # 1e3~1e5
    # qdtau = 1e-4    # 1e-10~1e-8
    
    # qdelta = 5e4    # 5e3~5e4
    # qddelta = 1e9   # 1e3~1e5
    
    # # highly conservative - inje_track.txt
    # qe_theta = 3e3
    # qe_c = 8e5      # 1e2 ~ 1e5

    # qtau = 1e0     # 1e3~1e5
    # qdtau = 1e-4    # 1e-10~1e-8
    
    # qdelta = 5e4    # 5e3~5e4
    # qddelta = 1e9   # 1e3~1e5
    
    # # less conservative - inje_track.txt
    # qe_theta = 5e3
    # qe_c = 5e5      # 1e2 ~ 1e5

    # qtau = 1e0     # 1e3~1e5
    # qdtau = 1e-4    # 1e-10~1e-8
    
    # qdelta = 5e4    # 5e3~5e4
    # qddelta = 1e9   # 1e3~1e5
    
@dataclass
class Solver_Parameters():
    qp_solver : str = "PARTIAL_CONDENSING_HPIPM" # "PARTIAL_CONDENSING_HPIPM"
    qp_solver_iter_max   : int = 200
    qp_solver_warm_start : int = 1
    nlp_solver_type : str = "SQP_RTI"
    integrator_type : str = "ERK"
    sim_method_num_stages : int = 4
    sim_method_num_steps  : int = 1
    hpipm_mode : str = "BALANCE" # BALANCE, SPEED_ABS, SPEED, ROBUST
    hessian_approx  : str = "GAUSS_NEWTON"
    nlp_solver_max_iter : int = 50
    tol : float = 1e-5
    print_level : int = 0       # 0~4
    globalization : str = 'MERIT_BACKTRACKING'
    levenberg_marquardt: float = 0.25
    nlp_solver_step_length : float = 0.5

@dataclass
class mpcc_Config():

    env : Environment_Parameters = Environment_Parameters()
    # vehicle : Vehicle_Parameters_NE = Vehicle_Parameters_NE()
    vehicle : Vehicle_Parameters_JW = Vehicle_Parameters_JW()
    pp_mpcc : PP_MPCC_Params = PP_MPCC_Params()
    solver : Solver_Parameters = Solver_Parameters()