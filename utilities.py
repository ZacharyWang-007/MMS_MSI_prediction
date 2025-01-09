import torch
import pydicom

import numpy as np


def normalize_image(img, pixel_value):
    Hu = pixel_value * int(img.RescaleSlope) + int(img.RescaleIntercept)

    window_width, window_level = 300, 75
    minWindow = window_level - window_width * 0.5
    Hu = (Hu - minWindow) / window_width
    Hu[Hu > 1] = 1
    Hu[Hu < 0] = 0
    return Hu

def read_image(img_path, transformation=None):
    img = pydicom.dcmread(img_path)
    img = normalize_image(img, img.pixel_array)[None,]
    img = torch.Tensor(img)
    if transformation is not None:
        img = transformation(img)
    return img


max = np.array([1.0000e+00, 8.9000e+01, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
    1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
    1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
    1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
    1.0000e+00, 1.0000e+00, 6.6330e+01, 1.6600e+01, 2.5900e+02, 9.3400e+02,
    5.7500e+01, 2.4800e+01, 2.5000e+00, 4.7000e+01, 1.5400e+02, 8.9600e+01,
    5.4100e+01, 4.5700e+01, 2.5000e+00, 2.7200e+01, 5.8050e+01, 9.5800e+00,
    4.8300e+00, 7.0400e+00, 8.2788e+02, 3.1891e+03, 3.6835e+03])

min = np.array([0.0000e+00, 1.8000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 0.0000e+00, 6.6000e-01, 6.6000e-01, 1.1000e+01, 2.7700e+00,
    1.8000e-01, 1.0000e-01, 1.0000e-01, 3.7400e+00, 2.0000e-02, 3.4000e+00,
    7.8000e+00, 1.4900e+01, 3.0000e-01, 2.3400e+00, 2.8000e-01, 6.6000e-01,
    2.2000e-01, 8.1000e-01, 1.0000e-02, 7.0000e-02, 2.0000e-02])
