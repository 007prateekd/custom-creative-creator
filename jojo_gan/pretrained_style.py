import torch
from torchvision import transforms, utils
from PIL import Image
from .util import *
from .model import *
from .e4e_projection import projection as e4e
from .restyle_projection import projection as restyle


latent_dim = 512


def align_face_helper(file_path):
    '''
    Aligns the face by using the pre-downloaded facial
    landmarks model and returns the cropped face image.
    '''
    
    name = strip_path_extension(file_path) + ".pt"
    aligned_face = align_face(file_path)
    return aligned_face, name


def load_finetuned_generator(preserve_color, style, device):
    '''
    Loads the style-specific fine-tuned generator using stored weigths. 
    Can also preserve color of the target image if that particular checkpoint is stored.
    '''
    
    ckpt = f"{style}.pt"
    ckpt = torch.load(os.path.join("JoJo_GAN/models", ckpt), map_location=lambda storage, loc: storage)
    generator = Generator(1024, latent_dim, 8, 2).to(device)
    generator.load_state_dict(ckpt, strict=False)
    return generator


def generate_sample(projection, aligned_face, name, device, n_iter, seed, generator):
    '''
    Generates an image where the reference style is applied to the target image
    by passing the latent code of the target image through the pretrained generator.
    '''
    
    my_w = projection(aligned_face, name, device, n_iter).unsqueeze(0)
    torch.manual_seed(seed)
    with torch.no_grad():
        generator.eval()
        my_sample = generator(my_w, input_is_latent=True)
    return my_sample


def get_transform(img_size, mean, std):
    '''Returns a transform to resize and normalize any image.'''

    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((mean, mean, mean), (std, std, std)),
        ]
    )
    return transform


def transform_style_images(styles, transform, device):
    '''Returns an array of style images with the given transform applied.'''
    
    style_images = []
    for style in styles:
        style = strip_path_extension(style)
        style_path = f"JoJo_GAN/style_images_aligned/{style}.png"
        style_image = transform(Image.open(style_path))
        style_images.append(style_image)
    style_images = torch.stack(style_images, 0).to(device)
    return style_images


def main(file_path, style, projection_opt, device, n_iter):
    '''
    Aligns the target image. Then loads the fine-tuned generator for 
    the given style and passes the image's code through it after which it is 
    transformed and displayed along with the reference and target image.
    '''
    
    projection = e4e if projection_opt == "e4e" else restyle
    style = [style]
    aligned_face, name = align_face_helper(file_path)
    generator = load_finetuned_generator(preserve_color=False, style=style[0], device=device)
    my_sample = generate_sample(projection, aligned_face, name, device, n_iter, 3000, generator)
    my_output = utils.make_grid(my_sample, normalize=True, value_range=(-1, 1))
    return my_output


if __name__ == "__main__":
    # options = ["art", "arcane_caitlyn", "arcane_jinx", "disney", "jojo", "jojo_yasuho", "sketch_multi"]
    device = "cpu"
    style = "art"
    file_path = f"test_input/Photo.jpeg"
    projection_opt = "e4e"
    n_iter = 1
    my_output = main(file_path, style, projection, device, n_iter)