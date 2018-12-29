# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.


'''
Created on Oct 29, 2018

author: borec
'''
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import posixpath

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from .image_util import BaseDataProvider

__all__ = [
    "InnH5Asphalt",
    "InnH5PCards",
]

DF_IMAGE_ID = "image_id"
DF_IMAGE_PATH = "src_image_path"
DF_MASK_PATH = "mask_path"


def pd_abs_paths(filename, dirname):
    return posixpath.join(dirname, filename)


class InnH5(BaseDataProvider):
    """
    Innovatrics data provider

    Expects the innovatrics h5 data
    """

    n_class = 2

    def __init__(self, nx, ny, h5_img_path, img_df_name, h5_ann_path,
                 ann_df_name, channels=1, a_min=0, a_max=255, seed=5):
        super(InnH5, self).__init__(a_min, a_max, channels)
        self.nx = nx
        self.ny = ny
        self.h5_img_path = h5_img_path
        self.img_df_name = img_df_name
        self.h5_ann_path = h5_ann_path
        self.ann_df_name = ann_df_name
        self.root_img_path = posixpath.dirname(h5_img_path)
        self.root_ann_path = posixpath.dirname(h5_ann_path)

        self.img_df = None
        self.ann_df = None

        self._load_dfs()

        self.rng = np.random.RandomState(seed)
        self.indices = list(range(len(self.ann_df)))
        self.rng.shuffle(self.indices)
        self.iter = iter(self.indices)

    def _load_dfs(self):
        with pd.HDFStore(self.h5_img_path, "r") as hdf:
            self.img_df = hdf[self.img_df_name]
            self.img_df[DF_IMAGE_PATH] = self.img_df[DF_IMAGE_PATH].apply(
                pd_abs_paths, args=(self.root_img_path,))

        with pd.HDFStore(self.h5_ann_path, "r") as hdf:
            self.ann_df = hdf[self.ann_df_name]
            self.ann_df[DF_MASK_PATH] = self.ann_df[DF_MASK_PATH].apply(
                pd_abs_paths, args=(self.root_ann_path,))

    def _get_data_label(self):
        try:
            i_ann = next(self.iter)
        except StopIteration:
            self.rng.shuffle(self.indices)
            self.iter = iter(self.indices)
            i_ann = next(self.iter)

        ann_row = self.ann_df.iloc[i_ann]
        select_condition = self.img_df[DF_IMAGE_ID] == ann_row[DF_IMAGE_ID]
        img_df = self.img_df.loc[select_condition]
        ann_path = ann_row[DF_MASK_PATH]
        img_path = getattr(img_df.iloc[0], DF_IMAGE_PATH)

        data = np.array(Image.open(img_path), np.float32)
        label = np.array(Image.open(ann_path), np.bool) >= 1

        return data, label

    def _next_data(self):
        return self._get_data_label()


class InnH5Asphalt(InnH5):
    """
    Innovatrics data provider for Asphalt Images

    Expects the innovatrics h5 data
    """

    n_class = 2

    def __init__(self, nx, ny, h5_img_path, img_df_name, h5_ann_path,
                 ann_df_name, channels=1, min_ratio=0.1, a_min=0, a_max=255,
                 seed=5):
        super(InnH5Asphalt, self).__init__(
            nx, ny, h5_img_path, img_df_name, h5_ann_path, ann_df_name,
            channels, a_min, a_max, seed)
        self.min_ratio = min_ratio

    def _pad(self, img, shape):

        height, width = img.shape[:2]
        if height > shape[0] and width > shape[1]:
            return img

        if height > shape[0]:
            hu_pad = hb_pad = 0
        else:
            hu_pad = (shape[0] - height) // 2
            hb_pad = (shape[0] - height) - hu_pad

        if width > shape[1]:
            wl_pad = wr_pad = 0
        else:
            wl_pad = (shape[1] - width) // 2
            wr_pad = (shape[1] - width) - wl_pad

        if len(img.shape) == 3:
            borders = ((hu_pad, hb_pad), (wl_pad, wr_pad), (0, 0))
        else:
            borders = ((hu_pad, hb_pad), (wl_pad, wr_pad))

        pad_img = np.pad(img, borders, mode="symmetric")

        return pad_img

    def _crop_data_label(self, data, label):
        if data.shape[0] - self.nx <= 0:
            start_y = 0
            end_y = start_y + data.shape[0]
        else:
            start_y = self.rng.randint(0, data.shape[0] - self.nx)
            end_y = start_y + self.nx

        if data.shape[1] - self.nx <= 0:
            start_x = 0
            end_x = start_x + data.shape[1]
        else:
            start_x = self.rng.randint(0, data.shape[1] - self.nx)
            end_x = start_x + self.nx

        sly = slice(start_y, end_y)
        slx = slice(start_x, end_x)

        crop_data = data[sly, slx]
        crop_label = label[sly, slx]

        return crop_data, crop_label

    def _next_data(self):
        data, label = self._get_data_label()
        crop_data, crop_label = self._crop_data_label(data, label)
        crop_label_int = crop_label * 1

        while self.min_ratio > (crop_label_int.sum() / crop_label_int.size):
            data, label = self._get_data_label()
            crop_data, crop_label = self._crop_data_label(data, label)
            crop_label_int = crop_label * 1

        in_data = self._pad(crop_data, (self.nx, self.nx))
        in_label = self._pad(crop_label, (self.nx, self.nx))

        return in_data, in_label


class InnH5PCards(InnH5):
    """
    Innovatrics data provider for Plastic Cards

    Expects the innovatrics h5 data
    """

    n_class = 2

    def __init__(self, nx, ny, h5_img_path, img_df_name, h5_ann_path, ann_df_name,
                 channels=3, a_min=0, a_max=255, seed=5):
        super(InnH5PCards, self).__init__(
            nx, ny, h5_img_path, img_df_name, h5_ann_path, ann_df_name,
            channels, a_min, a_max, seed)

    def _post_process(self, data, labels):
        data, labels = super()._post_process(data, labels)
        if data.shape[:2] != (self.ny, self.nx):
            data = cv2.resize(data, (self.nx, self.ny), interpolation=cv2.INTER_LINEAR)
            labels = cv2.resize(
                labels, (self.nx, self.ny), interpolation=cv2.INTER_NEAREST)
        return data, labels
