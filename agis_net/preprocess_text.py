import torch
from torchvision.transforms import transforms
from PIL import Image
import random
import numpy as np
from .util.util import tensor2im
from .models.networks import define_G


def __patch_instance_norm_state_dict(state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):
            if module.__class__.__name__.startswith("InstanceNorm") and \
                (key == "running_mean" or key == "running_var"):
                if getattr(module, key) is None:
                    state_dict.pop(".".join(keys))
            if module.__class__.__name__.startswith("InstanceNorm") and \
                (key == "num_batches_tracked"):
                state_dict.pop(".".join(keys))
        else:
            __patch_instance_norm_state_dict(
            state_dict, getattr(module, key), keys, i + 1)
            

def load_saved_generator(texture_name, device):
    G_path = f"AGIS_Net/pretrained_models/{texture_name}.pth"
    gpu_ids = [] if device == "cpu" else [0]
    net_G = define_G(
        input_nc=3, output_nc=3, nz=8, ngf=32, nencode=4, netG="agisnet",
        norm="instance", nl="relu", use_dropout=True, init_type="xavier",
        gpu_ids=gpu_ids, where_add="all", upsample="basic"
    )
    if isinstance(net_G, torch.nn.DataParallel):
        net_G = net_G.module
    state_dict = torch.load(G_path, map_location=device)
    if hasattr(state_dict, "_metadata"):
        del state_dict._metadata
    for key in list(state_dict.keys()):
        __patch_instance_norm_state_dict(state_dict, net_G, key.split("."))
    net_G.load_state_dict(state_dict)
    return net_G


def get_content(content_path, device):
    content = Image.open(content_path).convert("RGB")
    content = transforms.ToTensor()(content)
    content = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(content)
    content = content.unsqueeze(0).to(device)
    return content


def get_textures(ABC_path_list, few_size, n_encode, device):
    shuffled_alphas = ['0', '1', '2', '3', '4', '5', '6', '7']
    few_alphas = shuffled_alphas[:few_size]
    random.shuffle(few_alphas)
    chars_random = few_alphas[:n_encode]
    textures = []
    for char in chars_random:
        ABC_path_list[-5] = char
        style_path = "".join(ABC_path_list).replace("train", "style")
        style = Image.open(style_path).convert("RGB")
        w3, h = style.size
        w = w3 // 3
        textures.append(style.crop((2 * w, 0, 3 * w, h)))
    textures = list(map(lambda c: transforms.Normalize(
        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(transforms.ToTensor()(c)), textures))
    textures = torch.cat(textures)
    textures = textures.unsqueeze(0).to(device)
    return textures


def generate_char(texture_name, char, net_G, few_size, n_encode, device):
    content_path = f"AGIS_Net/datasets/base/{char}.png"
    ABC_path = f"AGIS_Net/datasets/{texture_name}/train/{char}.png"
    ABC_path_list = list(ABC_path)
    content = get_content(content_path, device)
    textures = get_textures(ABC_path_list, few_size, n_encode, device)
    with torch.no_grad():
        fake_C, _ = net_G(content, textures)
    fake_C = tensor2im(fake_C)
    return fake_C


def get_left_edge(string, h, w):
    for j in range(w):
        for i in range(h):
            r, g, b = string.getpixel((j, i))
            L = 0.2126 * r + 0.7152 * g + 0.0722 * b
            if L < 240:
                return j


def get_right_edge(string, h, w):
    for j in range(w - 5, -1, -1):
        for i in range(h):
            r, g, b = string.getpixel((j, i))
            L = 0.2126 * r + 0.7152 * g + 0.0722 * b
            if L < 240:
                return j


def get_top_edge(string, h, w):
    for i in range(5, h):
        for j in range(w):
            r, g, b = string.getpixel((j, i))
            L = 0.2126 * r + 0.7152 * g + 0.0722 * b
            if L < 240:
                return i


def get_bottom_edge(string, h, w):
    for i in range(h - 1, -1, -1):
        for j in range(w):
            r, g, b = string.getpixel((j, i))
            L = 0.2126 * r + 0.7152 * g + 0.0722 * b
            if L < 240:
                return i


def generate_char_cropped(texture_name, char, net_G, few_size, n_encode, horizontal, device):
    char_array = generate_char(texture_name, char, net_G, few_size, n_encode, device)
    char_image = Image.fromarray(np.uint8(char_array)).convert("RGB")
    h, w = char_image.size
    if horizontal:
        lx = get_left_edge(char_image, h, w)
        rx = get_right_edge(char_image, h, w)
        char_cropped = char_image.crop((lx, 0, rx, h))
    else:
        bx = get_bottom_edge(char_image, h, w)
        tx = get_top_edge(char_image, h, w)
        char_cropped = char_image.crop((0, tx, w, bx))
    return char_cropped