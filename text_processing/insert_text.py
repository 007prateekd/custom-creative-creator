import time
import numpy as np
import cv2
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def display(image, title=""):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 


def remove_bg(str):
    str = cv2.cvtColor(str, cv2.COLOR_RGB2RGBA)
    str[..., 3] = np.ones(str.shape[:2], dtype=np.uint8) * 255
    thresh = 240
    for i in range(str.shape[0]):
        for j in range(str.shape[1]):
            b, g, r, _ = str[i, j]
            L = 0.2126 * r + 0.7152 * g + 0.0722 * b
            if L > thresh:
                str[i, j] = (b, g, r, 0)
    return str


def overlay_images(image1, image2, x, y):
    y1, y2 = y, y + image2.shape[0]
    x1, x2 = x, x + image2.shape[1]
    alpha = image2[:, :, 3] / 255.0
    for ch in range(0, 3):
        image1[y1:y2, x1:x2, ch] = alpha * image2[:, :, ch] + (1 - alpha) * image1[y1:y2, x1:x2, ch]
    return image1


def log_time(st, title=""):
    en = time.time()
    time_taken = f"{en - st:0.3f}s"
    print(f"{title}:\t{time_taken:>5}")
    return en


def main(creative, str_image, cx, cy, scale_factor):  
    st = time.time()
    str_image = np.array(str_image)[..., ::-1]
    str_image = cv2.resize(str_image, (int(str_image.shape[1] * scale_factor), int(str_image.shape[0] * scale_factor)))
    str_image = remove_bg(str_image)
    st = log_time(st, "bg removal")
    overlayed = overlay_images(creative, str_image, cx, cy)
    st = log_time(st, "overlaying")
    return overlayed[..., ::-1]


if __name__ == "__main__":
    import time
    st = time.time()
    image_path = "ads/ad5.jpg"
    str_path = "strings/text2.jpg"
    cx, cy = 350, 5
    scale_factor = 4
    main(image_path, str_path, cx, cy, scale_factor)
    log_time(st, "total")