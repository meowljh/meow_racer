# 🏎️ Racedemia : Learning Style-Conditioned Race Driving Policies via Diffusion in Online Reinforcement Learning
## 📂 Repository Structure
```
diffusion_soft_actor_critic/
├── soft_dac/ : implementation of the soft diffusion actor critic policy
│    ├── diffusion/
│        ├── actor_diffusion.py : diffusion policy
│        ├── scheduler.py       : noise scheduler (linear_beta_scheduler, cosine_beta_scheduler, vp_beta_scheduler)
│    ├── networks/
│        ├── critic.py          : Q1, Q1 network as the SAC
│        ├── module.py          : Time embedding layer, etc
├── notebooks/
├── policy/ : implementation of the default SAC and Multi-Style conditioned SAC
│    ├── main.py
│    ├── model.py
│    ├── replay_memory.py
├── sota/
│    ├── DACER/
│    ├── DIPO/
│    ├── DPPO/
│    ├── QVPO/
│    ├── SQM/
└── ReadMe.md
```
## ⚙️ Implementation Details
### 
