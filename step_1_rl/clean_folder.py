import os
from glob import glob
import shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
fig_root = f"{ROOT}/debug_figures"
png_file_paths = glob(ROOT + "/*.png")
for f in  png_file_paths:
    shutil.move(
        src=f.replace('\\', '/'),
        dst=f"{fig_root}/{os.path.basename(f)}"
    )
# EXP_ROOT = f"{ROOT}/experiments"

# exp_names = os.listdir(EXP_ROOT)

# png_files = glob(ROOT + "/*.png")
# png_files = [p.replace('\\', '/') for p in png_files]
# for ei, exp_name in enumerate(exp_names):
    
#     for pf in png_files:
#         if exp_name in pf:
#             fig_root = f"{EXP_ROOT}/{exp_name}/traj_figures"
#             os.makedirs(fig_root, exist_ok=True)
            
#             shutil.move(src=pf.replace('\\', '/'),
#                         dst=f"{fig_root}/{os.path.basename(pf)}")
