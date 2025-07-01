'''
@C_roll_f: 전륜 구름 저항 계수
@C_roll_r: 후륜 구름 저항 계수
@C_b: 전/후륜 브레이킹 토크 비율 계수
@D_f, B_f, C_f: 전륜 S-PMF 계수
@D_r, B_r, C_r: 후륜 S-PMF 계수
@alpha_f, alpha_r: 전/후륜 타이어 슬립 각
''' 
class Vehicle_Parameters_NE():

    # Static params
    m   : float = 1840 # vehicle mass #
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
    
    Cb  : float = 0.7
    
    alpha_f_max: float = 0.1411 # 전륜 타이어 슬립 각 #
    alpha_r_max: float = 0.1311 # 후륜 타이어 슬립 각 #

    # Longitudinal dynamics
    rw : float = 0.36875 # 휠 반경 #
    mu = 0.9
    gravity = 9.81

    rolling_coef_fx = 0.00685303788632066 # 전륜 구름 저항 계수 #
    rolling_coef_rx = 0.00364373791962774 # 후륜 구름 저항 계수 #
  
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

    Cb  : float = 0.75
    
    alpha_f_max: float = 0.1411 # 전륜 타이어 슬립 각 #
    alpha_r_max: float = 0.1311 # 후륜 타이어 슬립 각 #

    # Longitudinal dynamics
    rw : float = 0.369 ## 휠 반경 ##
    mu = 1.0
    gravity = 9.81

    rolling_coef_fx = 0.006724669206530 # 전륜 구름 저항 계수 #
    rolling_coef_rx = 0.002439521729772 # 후륜 구름 저항 계수 #
    
    # constraints #
    delta_min : float = -0.5
    delta_max : float = 0.5
    torque_min : float = -18000.0
    torque_max : float = 3727.5