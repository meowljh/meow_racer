import yaml

class vehicleModel:
    '''vehicleModel
    simple object class for to hold the configuration values of the certain vehicle model
    Actually, the Vehicle Parameter Class
    
    [TO BE DONE]: will be changing the vehicle parameter values to a more precise
    and predictive DeepDynamics model.
    '''
    def __init__(self, config_yaml_path):
        
        with open(config_yaml_path, 'r', encoding='utf-8') as file:
            try:
                data = yaml.safe_load(file)
            except yaml.YAMLError as exception:
                print(f"Error in parsing YAML : {exception}")
                data = None
                
        self.config = data
            

        self.m                  = self.config['mass']
        self.Iz                 = self.config['Iz']
        self.lf                 = self.config['lf']
        self.lr                 = self.config['lr']

        self.rw                 = self.config['rw']
        self.mu                 = self.config['mu']
        self.gravity            = self.config['gravity']
        self.rolling_coef_fx    = self.config['rolling_coef_fx']
        self.rolling_coef_rx    = self.config['rolling_coef_rx']
        
        self.Bf                 = self.config['Bf']
        self.Cf                 = self.config['Cf']
        self.Df                 = self.config['Df']

        self.Br                 = self.config['Br']
        self.Cr                 = self.config['Cr']
        self.Dr                 = self.config['Dr']

        self.CLr                = self.config['CLr']
        self.CLf                = self.config['CLf']
        
        self.alpha_f_max        = self.config['alpha_f_max']
        self.alpha_r_max        = self.config['alpha_r_max']
        
        # self.W                  = 0.82225 
        
        self.Cb                 = self.config['Cb']
        
        #CarMaker > Vehicle Data Set > Outer Shell에서 확인한 값 
        # 1.89, 4.515가 실차 CM값 기반이고, 1.25는 이전에 Box2D 사용할 때에 지정했던 트랙간의 너비 비율에 맞춰본것#
        self.body_width         = 1.89 #1.25
        self.body_height        = 4.515 #2.38
        
        
    
        self.dx_max = 3.5 + self.body_width 
        self.dx_min = self.dx_max * -1
        self.dy_max = 3.5 + self.body_width 
        self.dy_min = self.dy_max * -1