import copy
import numpy as np
import torch
# import torchvision
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import math


def cutout(n_mask, size, length, p):
    """
    Args:
        n_mask: the number of mask per section, int
        size: the size of section, int
        length: length of mask. Int.
        p: probability of mask, Float.
    
    Return
        _cutout: Function for cutout
    """

    def _cutout(image):
        h, w, _ = image.shape
        mask = np.ones((h, w), np.float32)
        if np.random.random() > 0.5:
            return image
        for sh in range(h // size):
            for sw in range(w // size):
                for _ in range(n_mask):
                    if np.random.random() > p:
                        continue
                    y = np.random.randint(sh * size, (sh + 1) * size)
                    x = np.random.randint(sw * size, (sw + 1) * size)
                    y1 = np.clip(y - length // 2, 0, h)
                    y2 = np.clip(y + length // 2, 0, h)
                    x1 = np.clip(x - length // 2, 0, w)
                    x2 = np.clip(x + length // 2, 0, w)
                    mask[y1:y2, x1:x2] = 0.0
        mask = np.concatenate([np.expand_dims(mask, -1)] * 3, axis=-1)
        mask = np.int32(mask)
        return image * mask

    return _cutout


def get_aug(aug_cfg):
    seq = iaa.Sequential(
        [
            iaa.Sometimes(
                aug_cfg["Shear"]["p"],
                iaa.OneOf(
                    [
                        iaa.ShearX(aug_cfg["Shear"]["X"]),
                        iaa.ShearY(aug_cfg["Shear"]["Y"]),
                    ]
                ),
            ),
            iaa.Sometimes(
                aug_cfg["GaussianBlur"]["p"],
                iaa.GaussianBlur(sigma=aug_cfg["GaussianBlur"]["sigma"]),
            ),
            iaa.OneOf(
                [
                    iaa.LinearContrast(aug_cfg["LinearContrast"]["alpha"]),
                    iaa.AdditiveGaussianNoise(
                        loc=aug_cfg["AdditiveGaussianNoise"]["loc"],
                        scale=aug_cfg["AdditiveGaussianNoise"]["scale"],
                        per_channel=aug_cfg["AdditiveGaussianNoise"]["per_channel"],
                    ),
                    iaa.Multiply(
                        aug_cfg["Multiply"]["mul"],
                        per_channel=aug_cfg["Multiply"]["per_channel"],
                    ),
                ]
            ),
            iaa.Sometimes(
                aug_cfg["Affine"]["p"],
                iaa.Affine(
                    translate_percent=iap.Normal(
                        *aug_cfg["Affine"]["translate_percent"]
                    ),
                    rotate=iap.Normal(*aug_cfg["Affine"]["rotate"]),
                    scale=None,
                ),
            ),
        ],
        random_order=True,
    )
    return seq


def _get_cutout(aug_cfg):
    return cutout(
        n_mask=aug_cfg["cutout"]["n_mask"],
        size=aug_cfg["cutout"]["size"],
        length=aug_cfg["cutout"]["length"],
        p=aug_cfg["cutout"]["p"],
    )
