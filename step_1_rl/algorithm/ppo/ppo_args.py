import argparse

class Actor_Conf(object):
    def __init__(self):
        self.name = "adam"
        self.lr = 3e-4
        self.betas = [0.9, 0.999]
        self.eps = 1e-8
        self.weight_decay = 0

    def __call__(self):
        return self.__dict__
    
class Critic_Conf(object):
    def __init__(self):
        self.name = "adam"
        self.lr = 3e-4
        self.betas = [0.9, 0.999]
        self.eps = 1e-8
        self.weight_decay = 0
    
    def __call__(self):
        return self.__dict__
    
def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--xml_file", default='None', type=str)
    parser.add_argument("--train_iteration", type=int, default=1000000)
    parser.add_argument("--eval_iteration", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=10000)

    parser.add_argument("--actor_hidden_dims", nargs="+", type=int)
    parser.add_argument("--critic_hidden_dims", nargs="+", type=int)

    parser.add_argument("--lr_decay", action="store_true", default=False)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=128, help="number of trajectory data to sample from the buffer to update the parameter for one single step")
    parser.add_argument("--discount_factor", type=float, default=0.99, help="gamma factor for calculating the discounted value function")
    parser.add_argument("--lamb_da", default=0.95, type=float, help="GAE parameter")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--critic_loss_weight", default=1.0, type=float, help="weighting parameter on the critic network's loss value")
    parser.add_argument("--entropy_loss_weight", default=0.1, type=float, help="weighting paramter on the entropy of the predicted gaussian distribution")

    parser.add_argument("--critic_activation", type=str, default="tanh")
    parser.add_argument("--actor_activation", type=str, default="tanh")

    args = parser.parse_args()

    if args.xml_file == 'None':
        args.xml_file = None

    return args


if __name__ == "__main__":
    conf = Actor_Conf
    print(conf())