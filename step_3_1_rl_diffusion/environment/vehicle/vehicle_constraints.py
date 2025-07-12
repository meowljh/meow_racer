from dataclasses import dataclass

@dataclass
class PP_MPCC_Params():
    
    # Constraints
    
    # Model bounds
    track_width : float = 3.5             # half
    
    ### this velocity should be given as action constraints or penalty ##
    vx_min : float = 0/3.6
    vx_max : float = 200/3.6 #400/3.6 #200/3.6 ##KPH -> MS
    
    vy_min : float = -20/3.6
    vy_max : float = 20/3.6
    
    ax_min : float = -10.
    ax_max : float = +10.
    
    ay_min : float = -10.
    ay_max : float = +10.

    
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