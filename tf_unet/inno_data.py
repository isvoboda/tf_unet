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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from . image_util import BaseDataProvider

class InnoH5(BaseDataProvider):
    """
    Innovatrics data provider

    Expects the innovatrics h5 data
    """

    channels = 1
    n_class = 2

    def __init__(self, nx, h5_img_path, img_df_name, h5_an_path, an_df_name,
                 min_ratio=0.1, a_min=0, a_max=255, seed=5):
        super(InnoH5, self).__init__(a_min, a_max)
        self.nx = nx
        self.h5_img_path = h5_img_path
        self.img_df_name = img_df_name
        self.h5_an_path = h5_an_path
        self.an_df_name = an_df_name
        self.min_ratio = min_ratio
        self.base_path = posixpath.dirname(h5_img_path)

        self.img_df = None
        self.an_df = None

        self._load_df()

        self.rng = np.random.RandomState(seed)
        self.indices = list(range(len(self.an_df)))
        self.rng.shuffle(self.indices)
        self.iter = iter(self.indices)

    def _load_df(self):
        with pd.HDFStore(self.h5_img_path, "r") as hdf:
            self.img_df = hdf[self.img_df_name]

        with pd.HDFStore(self.h5_an_path, "r") as hdf:
            self.an_df = hdf[self.an_df_name]

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

    def _get_data_label(self):
        try:
            i_an = next(self.iter)
        except StopIteration:
            self.rng.shuffle(self.indices)
            self.iter = iter(self.indices)
            i_an = next(self.iter)

        an_row = self.an_df.iloc[i_an]
        img_df = self.img_df.loc[self.img_df["image_id"] == an_row["image_id"]]
        an_path = posixpath.join(self.base_path, an_row["mask_path"])
        img_path = posixpath.join(self.base_path, getattr(
            img_df.iloc[0], "src_image_path"))

        data = np.array(Image.open(img_path), np.float32)
        label = np.array(Image.open(an_path), np.bool) >= 1

        return data, label

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

        while self.min_ratio > (crop_label.sum() / crop_label.size):
            data, label = self._get_data_label()
            crop_data, crop_label = self._crop_data_label(data, label)

        in_data = self._pad(crop_data, (self.nx, self.nx))
        in_label = self._pad(crop_label, (self.nx, self.nx))

        return in_data, in_label
