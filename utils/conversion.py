import numpy as np
from PIL import Image
from pathlib import Path
from utils.IO import error_message, normalize_space_mat, get_space_mat


def np_to_image(array, save_path:Path=None):
    img = array
    print(img.shape)
    if save_path!=None:
        im = Image.fromarray(img)
        im.save(save_path)

def digit_image_to_int(array):
    im = (array - np.min(array))/(np.max(array) - np.min(array)) * 255
    im = im.astype(int)
    return im


if __name__=="__main__":
    attr = np.load("/Users/aaaje/Documents/ETH_WORK/interpretability/data_intermediate/attribution/resUnet/2019-07-01_144/2019-07-01_berlin_9ch144-saliency-target-channel1-W8_18.npy")[0]
    s = get_space_mat(attr)
    norm_s = normalize_space_mat(s, log_norm=True)[..., 11]
    print(norm_s.shape)
    np_to_image(norm_s, save_path="./test.tiff")