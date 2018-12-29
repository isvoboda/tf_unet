#%%
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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(5)
ROOT_PATH = "/srv/tasks/IMAGE-1474-Plastic-Cards-Semantic-Segmentation/data-mounts/pcards-alpha-0.1-0.4"
h5_img = posixpath.join(ROOT_PATH, "images-train.h5")
h5_ann = posixpath.join(ROOT_PATH, "annotations-train.h5")
IN_SHAPE = (2040, 1278)

#%%
generator = InnH5PCards(
    *IN_SHAPE, h5_img, "images", h5_ann, "annotations", channels=3)

#%%
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
img, label = generator(1)
ax[0].imshow(img[0, ..., 0], aspect="auto", cmap=plt.cm.gray)
ax[1].imshow(label[0, ..., 1], aspect="auto", cmap=plt.cm.gray)
plt.draw()
print("{} and {}".format(img.shape, label.shape))

#%%
net = unet.Unet(
    channels=generator.channels, n_class=generator.n_class, layers=5,
    features_root=16, summaries=True,
    cost_kwargs={"class_weights": [0.15, 0.85]})

trainer = unet.Trainer(
    net, optimizer="momentum", batch_size=1,
    opt_kwargs=dict(learning_rate=0.001))

#%%
path = trainer.train(
    generator, "./unet_trained-2", training_iters=10, epochs=1,
    dropout=0.85, display_step=1)
