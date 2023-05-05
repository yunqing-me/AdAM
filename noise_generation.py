import os
import torch
from tqdm import tqdm


# (1)
### create folder for noise generation
if not os.path.exists('./_noise'):
    os.makedirs('./_noise', exist_ok=True)

# (2)
### generation random noise for importance probings
for i in tqdm(range(250)):
    noise = torch.randn(4, 512)
    torch.save(noise, f'/_noise/{str(i).zfill(4)}.pt')

