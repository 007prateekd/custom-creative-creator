import time
import cv2
import numpy as np
from .parser import face_parser
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def display(image, title=""):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 


def scale_image(image, scale_factor):
    w = int(image.shape[1] * scale_factor)
    h = int(image.shape[0] * scale_factor)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    return image


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return image


def get_face_mask(image, remove_hair=False):
    parser = face_parser.FaceParser()
    out = parser.parse_face(image)
    condition = np.isin(out[0], [0, 7, 8, 14, 15, 16], invert=True)
    if remove_hair:
        condition = condition & (out[0] != 17)
    mask = np.where(condition[..., None], (255, 255, 255), (0, 0, 0)).astype(np.uint8)
    # fill mask holes (if any)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    contour, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        cv2.drawContours(mask, [cnt], 0, 255, -1)
    return mask


def edge_coords_helper(image, i_start, i_end, j_start, j_end, rev=False):
    i_step = 1 if i_start <= i_end else -1
    j_step = 1 if j_start <= j_end else -1
    for i in range(i_start, i_end, i_step):
        for j in range(j_start, j_end, j_step):
            i_, j_ = (j, i) if rev else (i, j)
            if (image[i_, j_] == (255, 255, 255)).all():
                return i_, j_


def get_edge_coords(image):
    h, w = image.shape[:2]
    tx, _ = edge_coords_helper(image, 0, h, 0, w)
    bx, _ = edge_coords_helper(image, h - 1, -1, 0, w)
    _, ly = edge_coords_helper(image, 0, w, 0, h, True)
    _, ry = edge_coords_helper(image, w - 1, -1, 0, h, True)
    return tx, bx, ly, ry


def log_time(st, en, title=""):
    time_taken = f"{en - st:0.3f}s"
    print(f"{title}:\t{time_taken:>5}")
    return en


def merge_images(image1, image2, mask, x, y):
    image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2RGBA)
    image2[..., 3] = mask
    y1, y2 = y, y + image2.shape[0]
    x1, x2 = x, x + image2.shape[1]
    alpha = image2[:, :, 3] / 255.0
    for ch in range(0, 3):
        image1[y1:y2, x1:x2, ch] = alpha * image2[:, :, ch] + (1 - alpha) * image1[y1:y2, x1:x2, ch]
    return image1


def main(src, dst_path, dst_h2c, rot_degrees, center):
    st = time.time()

    # reading + masking
    src_mask = get_face_mask(src, remove_hair=True)
    dst = cv2.imread(dst_path)
    st = log_time(st, time.time(), "masking")

    # rotation
    src = rotate_image(src, rot_degrees)
    src_mask = rotate_image(src_mask, rot_degrees)
    st = log_time(st, time.time(), "rotation")

    # cropping
    tx, bx, ly, ry = get_edge_coords(src_mask)
    src = src[tx:bx, ly:ry]
    src_mask = src_mask[tx:bx, ly:ry]
    st = log_time(st, time.time(), "cropping")

    # scaling
    src_h2c = np.sqrt(src.shape[0] ** 2 + src.shape[1] ** 2)
    scale_factor = dst_h2c / src_h2c
    src = scale_image(src, scale_factor)
    src_mask = scale_image(src_mask, scale_factor)
    st = log_time(st, time.time(), "scaling")

    # poisson editing + saving
    res = cv2.seamlessClone(src[..., ::-1], dst, src_mask, center, cv2.NORMAL_CLONE)
    st = log_time(st, time.time(), "cloning")
    return res[..., ::-1]
    # insertion
    # cx, cy = 1075, 300
    # res = merge_images(dst, src[..., ::-1], src_mask, cx, cy)
    # src_name = src_path.split("/")[1].split(".")[0]
    # dst_name = dst_path.split("/")[1].split(".")[0]
    # cv2.imwrite(f"{dst_name}-{src_name}.jpg", res)


if __name__ == "__main__":
    st = time.time()
    src_path = "images/faces/face10.png"
    dst_path = "images/creatives/creative2.png"
    # dst_h2c = 140.0     # for scaling
    dst_h2c = 440
    rot_degrees = 0.0   # for rotation
    center = (490, 190) # for positioning
    main(src_path, dst_path, dst_h2c, rot_degrees, center)
    _ = log_time(st, time.time(), "total time")