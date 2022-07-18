import os
import sys
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from argparse import Namespace
from .restyle.models.psp import pSp
# from .restyle.models.e4e import e4e
from .util import *
from .restyle.utils.inference_utils import *


@torch.no_grad()
def projection(img, name, device, n_iter=3):
    model_path = 'JoJo_GAN/models/restyle_psp_ffhq_encode.pt'
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    opts= Namespace(**opts)
    opts.n_iters_per_batch = n_iter
    net = pSp(opts, device).eval().to(device)
    # net = e4e(opts, device).eval().to(device)

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    img = transform(img).unsqueeze(0).to(device)
    avg_image = get_average_image(net, device)
    _, result_latents = run_on_batch(img.to(device), net, opts, avg_image)
    result_file = {}
    result_file['latent'] = result_latents[0][-1]
    torch.save(result_file, name)
    return torch.Tensor(result_latents[0][-1]).to(device)