import argparse
import json
import logging
import os.path
import random
from copy import deepcopy
from datetime import date
from glob import glob
from typing import Dict, Union, List, Tuple, Any, Iterable, Sequence

import albumentations as A
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from albumentations import BoxType
from tqdm import tqdm

from albumentations.core.bbox_utils import (
    BboxProcessor,
    convert_bboxes_to_albumentations,
)
import albumentations.augmentations.geometric.functional as F

def _convert_to_albumentations(
        self, data: Sequence[BoxType], rows: int, cols: int
) -> List[BoxType]:
    return convert_bboxes_to_albumentations(
        data, self.params.format, rows, cols, check_validity=False
    )


BboxProcessor.convert_to_albumentations = _convert_to_albumentations


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


CHAR_FREQUENCY = {
    "0": 0.07339595259386897,
    "1": 0.06323500934502162,
    "2": 0.08707079805832416,
    "3": 0.058839911836954875,
    "4": 0.047253825584254186,
    "5": 0.07384198971721238,
    "6": 0.040704695139065626,
    "7": 0.048483147412005524,
    "8": 0.035972350049934355,
    "9": 0.052987034462351146,
    "A": 0.009536491276166594,
    "B": 0.013714009699675572,
    "C": 0.024582084973335652,
    "D": 0.009623522909989697,
    "E": 0.020415445504054563,
    "F": 0.01860953910222516,
    "G": 0.011657887350604747,
    "H": 0.012908967086811864,
    "I": 0.008263653631503701,
    "J": 0.010678781470094831,
    "K": 0.022525962624264827,
    "L": 0.012245350878910696,
    "M": 0.0337639223416731,
    "N": 0.032534600513921756,
    "O": 0.008818480297125987,
    "P": 0.02089411949008163,
    "Q": 0.007915527096211286,
    "R": 0.011255366044172892,
    "S": 0.008949027747860643,
    "T": 0.008872875068265427,
    "U": 0.008263653631503701,
    "V": 0.011440308266046986,
    "W": 0.02139455138456448,
    "X": 0.011918982252074058,
    "Y": 0.00831804840264314,
    "Z": 0.009210122649329956,
    "&": 0.007915527096211286,
    "/": 0.007915527096211286,
    "+": 0.007915527096211286,
    "-": 0.008241895723047926,
    ":": 0.007915527096211286,
}


def make_text_image(
    background_image: np.ndarray,
    text_aug: A.Compose,
    font_path: str,
    is_lower: bool = False,
) -> Dict[str, Union[str, np.ndarray]]:
    """
    Making base text image (black background) of the same size as background

    Args:
        background_image: background image
        font_path: path of font file ex. ./test.ttf
        is_lower: define whether text is lower case

    Returns:
        "image": text image / text_image.shape == background_image.shape
        "bboxes": bounding box coordination of each character [xmin, ymin, width, height]
        "word_bboxes": bounding box coordinate of word [xmin, ymin, width, height]
        "text": text in image
        "color": color of character
    """
    text = np.random.choice(
        list(CHAR_FREQUENCY.keys()),
        size=1,
        p=list(CHAR_FREQUENCY.values()),
    )
    text = "".join(text)
    if is_lower:
        text = text.lower()

    image_height, image_width, _ = background_image.shape

    font = ImageFont.truetype(
        font_path,
        size=int(max(image_height, image_width) * np.random.uniform(0.05, 0.10)),
    )

    # get size of text box
    x, _, text_width, text_height = font.getbbox(text)
    text_width -= x

    bboxes = []
    x = int((image_width - text_width) * np.random.uniform(0.05, 0.95))
    y = int((image_height - text_height) * np.random.uniform(0.05, 0.95))

    x_accum = x
    for ch in text:
        xmin, ymin, xmax, ymax = font.getbbox(ch)
        bboxes += [[x_accum + xmin, y + ymin, xmax - xmin, ymax - ymin]]
        x_accum += xmax - xmin
    bboxes = np.array(bboxes)
    text_bbox: np.ndarray = np.array([[x, y, text_width, text_height]])

    text_image: Union[Image.Image, Iterable] = Image.new(
        "RGB", (text_width, text_height), color=(255, 255, 255)
    )

    draw = ImageDraw.Draw(text_image)
    draw.text((0, 0), text, fill=(0, 0, 0), font=font)

    text_image: np.ndarray = np.asarray(text_image)
    text_image: np.ndarray = cv2.copyMakeBorder(
        text_image,
        y,
        image_height - text_height - y,
        x,
        image_width - text_width - x,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    assert text_image.shape == background_image.shape

    color: np.ndarray = np.mean(
        background_image[x : x + text_width, y : y + text_height, :], axis=(0, 1)
    )

    res = {
        "image": text_image,
        "bboxes": bboxes,
        "word_bboxes": text_bbox,
        "text": text,
        "color": color,
    }

    res["word_bboxes"] = res["word_bboxes"].astype(np.float32)
    res["bboxes"] = res["bboxes"].astype(np.float32)

    res["word_bboxes"][..., 2] += res["word_bboxes"][..., 0]
    res["word_bboxes"][..., 3] += res["word_bboxes"][..., 1]

    res["bboxes"] = res["word_bboxes"]

    xmin, ymin, xmax, ymax = res["bboxes"][0]

    angle = np.random.uniform(-90, 90)

    res["bboxes"] = np.array([
        [xmin, ymin, 1],
        [xmin, ymax, 1],
        [xmax, ymax, 1],
        [xmax, ymin, 1]
    ]).T
    res["bboxes"] = res["bboxes"].astype(np.float32)
    height, width = res["image"].shape[:2]
    rot_mat = cv2.getRotationMatrix2D((width / 2 - 0.5, height / 2 - 0.5), angle, 1.0)
    res["image"] = F.rotate(res["image"], angle, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, (255, 255, 255))
    res["bboxes"] = rot_mat @ res["bboxes"]

    res["bboxes"] = res["bboxes"].astype(np.int32).T
    # for x, y in res["bboxes"]:
    #     res["image"] = cv2.circle(res["image"], (int(x), int(y)), 10, (0, 0, 0), 5)

    if np.sum(res["bboxes"] < 0):
        raise Exception()

    return res

def make_synth_image(
    background_image: np.ndarray,
    font_path: str,
    image_aug: A.Compose,
    text_aug: A.Compose,
    color_set: List[str],
    num_words: int = 2,
    iou_thr: float = 0.1,
    lower_prob: float = 0.0,
) -> Dict[str, Any]:
    """
    Making synthesis ocr image of the same size as background

    Args:
        background_image: background image
        font_path: path of font file ex. ./test.ttf
        image_aug: augmentation of image
        color_set: set of color / ["white", "black", "background_dark", "background_white"]
        num_words: the number of words made in image
        iou_thr: maximum iou of each word
        lower_prob: the probability of making lower case ocr image
    Returns:

    Returns:
        "image": text image / text_image.shape == background_image.shape
        "bboxes": bounding box coordination of each character [xmin, ymin, width, height]
        "word_bboxes": bounding box coordinate of word [xmin, ymin, width, height]
        "text": text in image
    """
    alpha = np.random.uniform(0.8, 0.99)
    color = np.random.choice(color_set)
    if color[: len("background")] == "background":
        alpha = np.random.uniform(0.8, 1.0)

    images = {
        "image": background_image,
        "bboxes": [],
        "text": [],
    }
    num_words = 5

    is_lower = True if np.random.uniform() < lower_prob else False
    j = 0
    while j < num_words:
        try:
            result_image = make_text_image(images["image"], text_aug, font_path, is_lower)
        except:
            continue
        j += 1
        blend_background_and_text(alpha, color, images, result_image)
        images["bboxes"] += [result_image["bboxes"]]
        images["text"] += [result_image["text"]]

    images["bboxes"] = np.array(images["bboxes"])

    # images = image_aug(**images)

    return images


def iou(bbox1: np.ndarray, bbox2: np.ndarray):
    """
    Calculate IOU

    Args:
        bbox1: coordinate of bbox / Nx4: [xmin, ymin, widht, height]
        bbox2: coordinate of bbox / Mx4: [xmin, ymin, widht, height]

    Returns:
        iou_map: N x M

    """
    bbox2 = np.transpose(bbox2)

    area1 = bbox1[:, 2:3] * bbox1[:, 3:4]
    area2 = bbox2[2:3, :] * bbox2[3:4, :]

    xmin = np.maximum(bbox1[:, 0:1], bbox2[0:1, :])
    ymin = np.maximum(bbox1[:, 1:2], bbox2[1:2, :])
    xmax = np.minimum(bbox1[:, 0:1] + bbox1[:, 2:3], bbox2[0:1, :] + bbox2[2:3, :])
    ymax = np.minimum(bbox1[:, 1:2] + bbox1[:, 3:4], bbox2[1:2, :] + bbox2[3:4, :])

    intersection = np.maximum(xmax - xmin, 0) * np.maximum(ymax - ymin, 0)
    iou_map = (intersection + 1e-6) / (area1 + area2 - intersection + 1e-6)

    return iou_map


def is_text_overlap(
    images: Dict[str, Any], iou_thr: float, result_image: Dict[str, Any]
):
    """
    Args:
        images: images already made
        iou_thr: maximum iou of each word
        result_image: Images to add

    Returns: bool
    """
    iou_result: np.ndarray = iou(images["word_bboxes"], result_image["word_bboxes"])
    iou_result = iou_result > iou_thr
    return iou_result.any()


def update_bbox(image_info: Dict[str, Any], text_info: Dict[str, Any]):
    image_info["bboxes"] = np.stack(
        (image_info["bboxes"], text_info["bboxes"])
    )

    image_info["text"] += text_info["text"]


def blend_background_and_text(
    alpha: float, color: str, images: Dict[str, Any], result_image: Dict[str, Any]
):
    """
    Args:
        alpha: blend alpha of background and text image
        color: text color
        images: original image
        result_image: text image to be added
    """
    color_pixel = (255, 255, 255)
    if color == "black":
        color_pixel = (0, 0, 0)
    if color == "background_dark":
        color_pixel = np.clip(result_image["color"] * 0.3, 0, 255)
    if color == "background_white":
        color_pixel = np.clip(result_image["color"] * 2.5, 0, 255)

    result_image["image"] = result_image["image"] // 255
    result_image["image"] = (1 - result_image["image"]).astype(np.uint8)
    text_with_color = images["image"] * result_image["image"]
    result_image["image"] *= np.array([[color_pixel]], dtype=np.uint8)
    result_image["image"] = (
        (1 - alpha) * text_with_color + alpha * result_image["image"]
    ).astype(np.uint8)

    images["image"] = np.where(
        result_image["image"] != 0, result_image["image"], images["image"]
    )


def get_base_coco_label(
    index_to_char: Dict[int, str]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    today = date.today()

    train_label_json = {
        "info": {
            "description": "SynthOCR",
            "url": "https://www.neuro-cle.com/",
            "version": "1.0",
            "year": today.year,
            "date_created": today.strftime("%Y/%m/%d"),
        },
        "images": [],
        "annotations": [],
        "categories": [],
    }

    train_label_json["categories"] += [
        {"id": index, "name": character, "supercategory": "character"}
        for index, character in index_to_char.items()
    ]
    val_label_json = deepcopy(train_label_json)

    return (
        train_label_json,
        val_label_json,
    )


def get_base_neurocle_label() -> Dict[str, Any]:
    label_json = {
        "name": "synth_ocr",
        "label_type": "ocr",
        "source": "labelset",
        "time": "2023-04-21T02:38:01.162",
        "version": "3.2.0",
        "data": [],
    }

    return label_json


def get_char_info() -> Tuple[Dict[str, int], Dict[int, str]]:
    char_to_index = {}
    index_to_char = {}

    chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ&/+-:"
    for i in range(len(chars)):
        char_to_index[chars[i]] = i + 1
        index_to_char[i + 1] = chars[i]
    return char_to_index, index_to_char


def get_augmentation(is_simple: bool = False) -> Tuple[A.Compose, A.Compose]:
    def _make_darker(image: np.ndarray, **kwargs):
        image = (image * 0.8).astype(np.uint8)
        return image

    if not is_simple:
        image_aug = A.Compose(
            [
                A.HueSaturationValue(),
                A.RandomGamma(),
                A.CLAHE(),
                A.RandomBrightnessContrast(),
                A.ImageCompression(),
                A.OneOf(
                    [
                        A.MotionBlur(5, p=0.5),
                        A.GlassBlur(max_delta=1, p=0.5),
                        A.GaussianBlur((3, 5), p=0.5),
                        A.ISONoise(p=0.5),
                        A.RingingOvershoot(p=0.5),
                    ],
                    p=1,
                ),
                A.OneOf(
                    [
                        A.Perspective((0.05, 0.15), p=0.5),
                        A.Affine(shear=(-20, 20), p=0.5),
                        A.OpticalDistortion(0.1, 0.1, p=0.5),
                    ],
                    p=1,
                ),
                A.OneOf(
                    [
                        A.RandomFog(fog_coef_lower=0.2, fog_coef_upper=0.3, p=0.3),
                        A.RandomRain(-2, 2, p=0.3),
                        A.RandomShadow(p=0.3),
                    ],
                    p=1,
                ),
                A.OneOf(
                    [A.Equalize(p=0.3), A.Spatter(p=0.3), A.Lambda(image=_make_darker)],
                    p=0.5,
                ),
                A.ToGray(p=0.5),
            ],
            bbox_params=A.BboxParams(format="coco", label_fields=["text"]),
        )
    else:
        image_aug = A.Compose(
            [
                A.HueSaturationValue(),
                A.RandomGamma(),
                A.CLAHE(),
                A.ImageCompression(),
                A.OneOf(
                    [
                        A.Perspective((0.05, 0.15), p=0.5),
                        A.Affine(shear=(-20, 20), p=0.5),
                        A.OpticalDistortion(0.1, 0.1, p=0.5),
                    ],
                    p=1,
                ),
                A.ToGray(p=0.5),
            ],
            bbox_params=A.BboxParams(format="yolo", label_fields=["text"]),
        )
    text_aug = A.Compose([
    ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["text"]),
    )
    return image_aug, text_aug


def write_json_file(result_dir, num_train, num_val):
    with open(f"{result_dir}/coco_synthocr_train.json", "w") as f:
        json.dump(coco_train_label, f)
    with open(f"{result_dir}/coco_synthocr_val.json", "w") as f:
        json.dump(coco_val_label, f)
    with open(f"{result_dir}/neurocle_synthocr.json", "w") as f:
        json.dump(neurocle_label, f)
    with open(f"{result_dir}/data_info.txt", "w") as f:
        f.write(
            f"num train: {num_train}, num val: {num_val} "
            f"created at {date.today().strftime('%Y/%m/%d')}"
        )


def read_image_info(image_paths: List[str]) -> Tuple[np.ndarray, str, int]:
    image_path = np.random.choice(image_paths)
    image = cv2.imread(image_path)
    h, w, c = image.shape

    ratio = 1024 / max(h, w)
    image = cv2.resize(image, (int(w * ratio), int(h * ratio)))
    font = np.random.choice(font_paths)
    num_text = int(np.random.uniform(2, 4))

    return image, font, num_text


if __name__ == "__main__":
    config = argparse.ArgumentParser()

    config.add_argument(
        "--background_dir", type=str, help="directory of background for creating images"
    )
    config.add_argument(
        "--font_dir", type=str, help="directory of font file for creating images"
    )
    config.add_argument("--result_dir", type=str, help="directory of result images")
    config.add_argument(
        "--num_create_images",
        type=int,
        default=80_000,
        help="number of image to create",
    )
    config.add_argument(
        "--train_ratio", type=float, default=0.8, help="ratio of train image"
    )

    args = config.parse_args()

    logging.info("start creating image")
    color_set = ["white", "black", "background_dark", "background_white"]
    image_extensions = {"png", "jpg", "jpeg", "PNG", "JPG"}

    num_images = args.num_create_images
    train_ratio = args.train_ratio
    result_dir = args.result_dir

    image_paths = [
        p
        for p in glob(os.path.join(args.background_dir, "*"))
        if os.path.splitext(p)[1][1:] in image_extensions
    ]

    font_paths = glob(os.path.join(args.font_dir, "*.TTF")) + glob(
        os.path.join(args.font_dir, "*.ttf")
    )

    if not os.path.exists(f"{result_dir}/images/"):
        os.makedirs(f"{result_dir}/images/")

    # get base information
    image_aug, text_aug = get_augmentation()
    char_to_index, index_to_char = get_char_info()
    coco_train_label, coco_val_label = get_base_coco_label(index_to_char)
    neurocle_label = get_base_neurocle_label()

    # set index using in label
    image_index = 1
    annotation_index = 1
    num_train = 0
    num_val = 0

    label_list = []
    pbar = iter(tqdm(range(num_images), total=num_images))
    while image_index <= num_images:
        image, font, num_text = read_image_info(image_paths)

        # to handle with putting text outside of label

        image_data = make_synth_image(image, font, image_aug, text_aug, color_set, num_text)

        if np.random.uniform() < train_ratio:
            coco_label = coco_train_label
            set_name = "train"
            num_train += 1
        else:
            coco_label = coco_val_label
            set_name = "test"
            num_val += 1

        anno = []
        h, w, c = image.shape
        image_name = f"{image_index:08d}.jpg"

        for label, bbox in zip(image_data["text"], image_data["bboxes"]):
            anno += [{"label": label, "bbox": bbox.tolist()}]

        anno = {
            "image_name": image_name,
            "annotation": anno
        }

        cv2.imwrite(f"{result_dir}/images/{image_name}", image_data["image"])
        image_index += 1
        label_list += [anno]
        print(len(label_list))
        next(pbar)

    with open(f"{result_dir}/result.json", "w") as f:
        json.dump(label_list, f)
