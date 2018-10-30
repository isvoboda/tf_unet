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
from scipy.ndimage import gaussian_filter

from tf_unet.inno_data import InnoH5
from tf_unet import unet

INPUT_SIZE = 572
# IMG_H5 = "/mnt/data/Tasks/XXXX_Asphalt_train_data/dataset/images.h5"
# AN_H5 = "/mnt/data/Tasks/XXXX_Asphalt_train_data/dataset/annotations.h5"
IMG_H5 = "../data/images.h5"
AN_H5 = "../data/annotations.h5"

if __name__ == '__main__':
    generator = InnoH5(INPUT_SIZE, IMG_H5, "df", AN_H5, "df")
    
    # fig, ax = plt.subplots(1,2, figsize=(12,4))
    # for _ in range(100):
    #     img, label = generator(1)
    #     ax[0].imshow(img[0, ..., 0], aspect="auto", cmap=plt.cm.gray)
    #     ax[1].imshow(label[0, ..., 1], aspect="auto", cmap=plt.cm.gray)
    #     plt.pause(3)
    #     plt.draw()

    net = unet.Unet(channels=generator.channels, 
                    n_class=generator.n_class, 
                    layers=3,
                    features_root=16,
                    summaries=True,
                    cost_kwargs={"class_weights": [0.33, 0.66]})

    trainer = unet.Trainer(net,
                           optimizer="momentum",
                           batch_size=1,
                           opt_kwargs=dict(learning_rate=0.001))
    
    path = trainer.train(generator, "./unet_trained",
                         training_iters=10000,
                         epochs=10,
                         dropout=0.75,  # probability to keep units
                         display_step=10)
  