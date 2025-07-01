import os, sys
from tqdm import tqdm
import numpy as np
import pickle

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(root)

from common.common_utils import seed_all, load_activation
from ppo.ppo_args import Actor_Conf, Critic_Conf, load_args
from ppo.ppo_algorithm import PPO
from eval_main import load_env

def evaluate(env, agent, seed, num_iterations):
    loop = tqdm(range(num_iterations))
    scores = []

    for iter in loop:
        state, _ = env.reset(seed=seed+iter+42)
        env.render()

        terminated, truncated = False, False
        score = 0
        while not (terminated or truncated):
            action = agent.act(state, training=False)
            s_prime, reward, terminated, truncated, _ = env.step(action['action'])
            score += reward
            state = s_prime
        scores.append(score)
 

    return round(np.mean(scores), 4)
    
def main(args):
    seed_all(args.seed) 

    actor_optimizer_conf = Actor_Conf().__dict__
    critic_optimizer_conf = Critic_Conf().__dict__
    env = load_env(args=args)
    print(f"=====> Successfully Loaded the Environment {args.env_name}")

    agent = PPO(
        env=env,

        actor_hidden_dims=args.actor_hidden_dims,
        critic_hidden_dims=args.critic_hidden_dims,
        actor_optimizer_config=actor_optimizer_conf,
        critic_optimizer_config=critic_optimizer_conf,

        lr_decay=args.lr_decay,
        clip_epsilon=args.clip_epsilon,
        batch_size=args.batch_size,
        discount_factor=args.discount_factor,
        lamb_da=args.lamb_da,
        num_epochs=args.num_epochs,
        critic_loss_weight=args.critic_loss_weight,

        hidden_activation_actor=load_activation(args.actor_activation),
        hidden_activation_critic=load_activation(args.critic_activation)
    )

    print("=====> Successfully Loaded the Agent PPO")

    logger = []
    (state, _) = env.reset(seed=args.seed)
    terminated, truncated = False, False

    print("=====> Sucessfully Reset Environment")
 

    # for t in tqdm(1, args.train_iteration+1):
    loop = tqdm(range(1, args.train_iteration+1))

    for t in loop:
        env.render()
        a = agent.act(state, training=True)
 
        s_prime, reward, terminated, truncated, _ = env.step(action=a['action'])
        
        is_done = 1 if terminated or truncated else 0
        ##(1, dim)
        transitions = {
            'state': state.reshape(1, -1), # .detach().cpu().numpy(),
            'action': a['action'].reshape(1,-1), # .detach().cpu().numpy(),
            'reward': np.array([reward]).reshape(1, -1), # .detach().cpu().numpy(),
            'state_prime': s_prime.reshape(1, -1), # .detach().cpu().numpy(),
            'done': np.array([is_done]).reshape(1, -1)
        } 

        result = agent.step([transitions], step=t)
        state = s_prime

        if result is not None:
            logger.append([t, 'critic_loss', result['critic_loss']])
            logger.append([t, 'actor_loss', result['actor_loss']])
            logger.append([t, 'entropy_bonus', result['entropy_bonus']])

        if terminated or truncated:
            (state, _) = env.reset()
            terminated, truncated = False, False
        
        if t % args.eval_interval == 0:
            score = evaluate(env=env, agent=agent, seed=args.seed, num_iterations=args.eval_iteration)
            logger.append([t, 'Eval return', score])
            print("=" * 100)
            print(f"Evaluate score for timestep {t}/{args.train_iteration} : {score}")
            print("=" * 100)
    return logger

if __name__ == "__main__":
    args = load_args()
    logger = main(args=args)

    root = f"{os.path.dirname(os.path.abspath(__file__))}/{args.save}"
    os.makedirs(root, exist_ok=True)

    pickle.dump(logger, f"{root}/logs.pkl", "wb")
