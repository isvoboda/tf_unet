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
Created on Dec 10, 2018

author: borec
'''
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import os
import posixpath

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter

from tf_unet import unet
from tf_unet.inno_data import InnH5PCards

INPUT_SIZE = 572


def parse_args():
    description = ("Train UNet on Plastic Cards data")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--img-h5", required=True, type=str)
    parser.add_argument("--img-df", required=False, default="images", type=str)
    parser.add_argument("--ann-h5", required=True, type=str)
    parser.add_argument("--ann-df", required=False, default="annotations",
                        type=str)
    parser.add_argument("--batch-size", required=False, type=int, default=1)
    parser.add_argument("--gpu-id", required=False, type=int, default=1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    flags = parse_args()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(flags.gpu_id)

    generator = InnH5PCards(
        INPUT_SIZE, flags.img_h5, flags.img_df, flags.ann_h5, flags.ann_df,
        channels=3)

#    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
#    for _ in range(100):
#        img, label = generator(1)
#        ax[0].imshow(img[0, ..., 0], aspect="auto", cmap=plt.cm.gray)
#        ax[1].imshow(label[0, ..., 1], aspect="auto", cmap=plt.cm.gray)
#        plt.pause(3)
#        plt.draw()

    net = unet.Unet(
        channels=generator.channels, n_class=generator.n_class, layers=5,
        features_root=16, summaries=True,
        cost_kwargs={"class_weights": [0.5, 0.5]})

    trainer = unet.Trainer(
        net, optimizer="momentum", batch_size=flags.batch_size,
        opt_kwargs=dict(learning_rate=0.001))

    path = trainer.train(
        generator, "./unet_trained-2", training_iters=10000, epochs=10,
        dropout=0.85, display_step=250)
