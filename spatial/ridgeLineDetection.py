from ridge_detection.lineDetector import LineDetector
from ridge_detection.params import Params,load_json
from ridge_detection.basicGeometry import reset_counter
from ridge_detection.helper import displayContours,save_to_disk
from argparse import ArgumentParser
from datetime import datetime
from PIL import Image
from  mrcfile import open as mrcfile_open
import numpy as np

from utils.IO import get_space_mat, normalize_space_mat


def run():
    start=datetime.now()
    parser = ArgumentParser("ridge detection parser tool")
    parser.add_argument(dest="config_filename",type=str, nargs='?',help="name of the config_file to use. Default value is 'example_config.json'", default = "configs/ridgeDetection.json")
    args=parser.parse_args()
    config_filename = args.config_filename if args.config_filename is not None else "example_config.json"
    json_data=load_json(config_filename)
    params = Params(config_filename)

    # try:
    #     img=mrcfile_open(json_data["path_to_file"]).data
    # except ValueError:
    #     img=Image.open(json_data["path_to_file"])
    try:
        img = np.load(json_data["path_to_file"])
        space_mat = get_space_mat(img[0])
        # of shape [19,512,448]
        # 19 = 12 timeepochs + 7 static features

        space_mat_ = normalize_space_mat(space_mat, log_norm=False)
        img = space_mat_[..., 11]
        print("Read image done")
    except ValueError:
        print("Error")
        return

    detect = LineDetector(params=config_filename)
    result = detect.detectLines(img)
    resultJunction =detect.junctions
    out_img,img_only_lines = displayContours(params,result,resultJunction)
    if params.get_saveOnFile() is True:
        save_to_disk(out_img,img_only_lines)

    print(" TOTAL EXECUTION TIME: " + str(datetime.now()-start))


if __name__ == "__main__":
    run()