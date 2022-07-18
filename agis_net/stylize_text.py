from .preprocess_text import *


def get_char_images(texture_name, net_G, few_size, n_encode, horizontal, string, device):
    images = []
    for char in string:
        if char != " ":
            char_array = generate_char_cropped(texture_name, char, net_G, few_size, n_encode, horizontal, device)
            images.append(Image.fromarray(np.uint8(char_array)).convert("RGB"))
    return images


def stack_images(string, images, padding, space_size, horizontal):
    if horizontal:
        ones = np.zeros((images[0].size[1], padding, 3), dtype="uint8") + 255
        spaces = np.zeros((images[0].size[1], space_size, 3), dtype="uint8") + 255
    else:
        ones = np.zeros((padding, images[0].size[0], 3), dtype="uint8") + 255
        spaces = np.zeros((space_size, images[0].size[0], 3), dtype="uint8") + 255
    stack = []
    i = 0
    for image in images:
        stack.append(np.asarray(image))
        if i + 1 < len(string) and string[i + 1] == " ":
            i += 1
            stack.append(spaces)
        else:
            stack.append(ones)
        i += 1
    stack.pop()
    return stack


def concat_chars(texture_name, net_G, few_size, n_encode,  horizontal, string, padding, space_size, device):
    images = get_char_images(texture_name, net_G, few_size, n_encode, horizontal, string, device)
    stack = stack_images(string, images, padding, space_size, horizontal)
    str_image = np.hstack(stack) if horizontal else np.vstack(stack)
    str_image = Image.fromarray(str_image)
    return str_image


def main(texture_name, few_size, n_encode, horizontal, string, padding, space_size, device):
    net_G = load_saved_generator(texture_name, device)
    str_image = concat_chars(texture_name, net_G, few_size, n_encode, horizontal, string, padding, space_size, device)
    return str_image


if __name__ == "__main__":
    texture_name = "font2"
    device = "cpu"
    few_size = 5
    n_encode = 4
    horizontal = True
    string = "SOME RANDOM TEXT HERE"
    padding = 2
    space_size = 10
    main(texture_name, few_size, n_encode, horizontal, string, padding, space_size, device)