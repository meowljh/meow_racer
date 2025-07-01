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
