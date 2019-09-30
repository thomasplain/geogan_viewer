import glob
import matplotlib.cm as cm
import os
import tkinter

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import numpy as np

plt.ion()

root = tkinter.Tk()
root.wm_title("Embedding in Tk")

fig = Figure(figsize=(5, 4), dpi=100)
# t = np.arange(0, 3, .01)
# fig.add_subplot(111).plot(t, 2 * np.sin(2 * np.pi * t))
im_ax = plt.imshow(np.zeros((64, 64)))

canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
canvas.draw()
canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

toolbar = NavigationToolbar2Tk(canvas, root)
toolbar.update()
canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)


def on_key_press(event):
    print("you pressed {}".format(event.key))
    key_press_handler(event, canvas, toolbar)


canvas.mpl_connect("key_press_event", on_key_press)

def _quit():
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate


button = tkinter.Button(master=root, text="Quit", command=_quit)
button.pack(side=tkinter.BOTTOM)

base_img_dir = os.path.join('saved_images', 'orig_geo_self_attn', 'epoch_latest/')

def get_images(series_no):
    image_filenames = glob.glob(os.path.join(base_img_dir, series_no + '_output_one_hot_*.png'))
    image_filenames.sort()

    images = [plt.imread(filename) for filename in image_filenames]
    gt_image = plt.imread(os.path.join(base_img_dir, series_no + '_ground_truth_one_hot.png'))

    mask_loc = np.where(images[0][:, :, :-1].sum(axis=-1) == 0)

    mask_min_x = min(mask_loc[1])
    mask_max_x = max(mask_loc[1])
    mask_min_y = min(mask_loc[0])
    mask_max_y = max(mask_loc[0])
    mask_size = mask_max_x-mask_min_x

    masked_regions = [image[mask_min_y:mask_max_y, mask_min_x:mask_max_x, :] for image in images]
    # for mask_region in masked_regions:
    #     plate_area = mask_region.sum(axis=-1) == 4
    #     mask_region[:, :, :-1][np.stack([plate_area]*3, axis=-1)] = 0.9
    masked_gt_region = gt_image[mask_min_y:mask_max_y + 1, mask_min_x:mask_max_x + 1, :]
    # plate_area = masked_gt_region.sum(axis=-1) == 4
    # masked_gt_region[:, :, :-1][np.stack([plate_area]*3, axis=-1)] = 0.9

    masked_box = np.vstack([np.hstack(masked_regions[:3]), np.hstack(masked_regions[3:6]), np.hstack(masked_regions[6:9])])
    side_bar = np.zeros((masked_box.shape[0], 1, 4))
    side_bar[:, :, -1] = 1
    masked_box = np.hstack([masked_box, side_bar])
    bottom_bar = np.zeros((1, masked_box.shape[1], 4))
    bottom_bar[:, :, -1] = 1
    masked_box = np.vstack([masked_box, bottom_bar])
    masked_box[mask_size:mask_size*2+1, mask_size:mask_size*2+1] = masked_gt_region

    plt.imsave(os.path.join(base_img_dir, series_no + '_inpaint_grid.png'), masked_box)

    plt.figure(); plt.imshow(masked_box)

    # rows = int(np.sqrt(len(images)))
    # cols = int(np.ceil(len(images) / rows))
    # fig, ax = plt.subplots(rows, cols)
    # ax = ax.ravel()
    # for i, image in enumerate(images):
    #     ax[i].imshow(image)

    ridge_images = np.sum(np.stack([(1-image[:, :, 0])*p*0.1 for (image, p) in zip(images, range(1, 11))]), axis=0)
    ridge_image_mask = np.max([(1-image[:, :, 0]) for image in images], axis=0)
    ridge_image = cm.autumn(ridge_images/np.max(ridge_images))*ridge_image_mask[:, :, None]
    ridge_image[:, :, -1] = 1
    sub_images = np.sum(np.stack([(1-image[:, :, 2])*p*0.1 for (image, p) in zip(images, range(1, 11))]), axis=0)
    sub_image_mask = np.max([(1-image[:, :, 2]) for image in images], axis=0)
    sub_image = cm.winter(sub_images / np.max(sub_images))*sub_image_mask[:, :, None]
    sub_image[:, :, -1] = 1

    plate = np.ones((256, 512, 4))
    plate[:, :, :-1][np.where(np.stack([ridge_image_mask]*3, axis=-1))] = 0
    plate[:, :, :-1][np.where(np.stack([sub_image_mask]*3, axis=-1))] = 0
    # plate[np.where(np.logical_or(ridge_images.sum(axis=-1) == 0, sub_images.sum(axis=-1) == 0))] = 1
    # ridge_images[np.where(ridge_images.sum(axis=-1) == 0)] = 1
    # sub_images[np.where(sub_images.sum(axis=-1) == 0)] = 1
    #

    full_im = ridge_image + sub_image + plate
    full_im[:, :, 0][mask_loc] = 0.498
    full_im[:, :, 1][mask_loc] = 0.
    full_im[:, :, 2][mask_loc] = 0.

    merged_masked = full_im[mask_min_y:mask_max_y+1, mask_min_x:mask_max_x+1]
    plt.imsave(os.path.join(base_img_dir, series_no + '_combined_inpaint.png'), merged_masked)
    im_ax.set_data(merged_masked)
    canvas.draw()


available_series = [filename[len(base_img_dir):len('series_xxxxx')+len(base_img_dir)] for filename in
                    glob.glob(os.path.join(base_img_dir, 'series_*_gt_divergence.png'))]
series = tkinter.StringVar(root)
series.set(available_series[0])

dropdown = tkinter.OptionMenu(root, series, *available_series, command=get_images)
dropdown.pack(side=tkinter.BOTTOM)

tkinter.mainloop()
# If you put root.destroy() here, it will cause an error if the window is
# closed with the window manager.