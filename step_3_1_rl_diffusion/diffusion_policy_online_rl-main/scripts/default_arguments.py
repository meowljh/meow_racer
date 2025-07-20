class SDAC_Default(object):
    def __init__(self):
        self.num_timesteps = 20
        self.num_particles = 4
        self.noise_scale = 0.05
        self.target_entropy_scale = 0.9
        self.beta_schedule_scale = 0.3
        self.beta_schedule_type = 'cosine'
        self.time_dim = 16
        self.activation_fn = 'mish'

        self.gamma = 0.99
        self.lr = 3e-4
        self.lr_schedule_end = 3e-5
        self.alpha_lr = 3e-2
        self.tau = 0.005
        self.delay_alpha_update = 250
        self.delay_update = 2
        self.reward_scale = 0.2
        self.num_samples = 200