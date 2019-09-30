from tkinter import filedialog, messagebox
from tkinter import *
from model import Model
import matplotlib.pyplot as plt
from collections import OrderedDict
from matplotlib import cm
from skimage.morphology import skeletonize

import base64
from PIL import Image, ImageTk
import glob
import numpy as np
import os
import random
import torch

torch.set_default_tensor_type('torch.FloatTensor')

class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.init_window()

        self.arch_file_name = None
        self.arch = Model().float()
        self.dataroot = '/home/tom/data/ellipses3/test'
        self.files = glob.glob(os.path.join(self.dataroot, '*.pkl'))

        self.start_dir = os.getcwd()

        self.mask_size = 64
        self.mask_pos = (0, 0)

    def init_window(self):
        self.master.title('Geogan viewer')
        self.pack(fill=BOTH, expand=1)

        menu = Menu(self.master)
        self.master.config(menu=menu)
        file = Menu(menu)
        file.add_command(label='Exit', command=exit)

        menu.add_cascade(label='File', menu=file)
        edit = Menu(menu)
        edit.add_command(label='Undo')
        menu.add_cascade(label='Edit', menu=edit)

        archButton = Button(self, text='Choose arch file', command=self.build_arch)
        archButton.place(x=0, y=0)

        chooseDatarootButton = Button(self, text='Choose dataroot', command=self.choose_dataroot)
        chooseDatarootButton.place(x=150, y=0)

        load_model_button = Button(self, text='Load model', command=self.load_model)
        load_model_button.place(x=300, y=0)

        load_model_button = Button(self, text='Ellipse preset', command=self.load_ellipse_preset)
        load_model_button.place(x=450, y=0)

        load_model_button = Button(self, text='Earth preset', command=self.load_earth_preset)
        load_model_button.place(x=600, y=0)

        load_model_button = Button(self, text='Synthetic preset', command=self.load_synthetic_preset)
        load_model_button.place(x=750, y=0)

        random_image_button = Button(self, text='Get random image', command=self.set_random_image)
        random_image_button.place(x=100, y=300)

        refresh_button = Button(self, text='Refresh image', command=self.refresh)
        refresh_button.place(x=250, y=300)

        save_images_button = Button(self, text='Save images', command=self.save_current_images)
        save_images_button.place(x=100, y=360)

        blank_image = Image.fromarray(np.ones((256, 512, 3), dtype=np.uint8) * 127)
        blank_image_tk = ImageTk.PhotoImage(image=blank_image)

        self.input_image_pane = Label(self.master, image=blank_image_tk, height=255, width=512)
        self.input_image_pane.image = blank_image_tk
        self.input_image_pane.pack(fill=BOTH, expand=1)
        self.input_image_pane.place(x=640/2-512/2, y=40)
        self.input_image_pane.bind('<B1-Motion>', self.update_mask_pos)

        self.output_div_image_pane = Label(self.master, image=blank_image_tk, height=255, width=512)
        self.output_div_image_pane.image = blank_image_tk
        self.output_div_image_pane.pack(fill=BOTH, expand=1)
        self.output_div_image_pane.place(x=3*640/2-512/2, y=40)

        self.output_disc_image_pane = Label(self.master, image=blank_image_tk, height=255, width=512)
        self.output_disc_image_pane.image = blank_image_tk
        self.output_disc_image_pane.pack(fill=BOTH, expand=1)
        self.output_disc_image_pane.place(x=640-512/2, y=340)

        self.slider = Scale(self.master, from_=0, to=100, orient=HORIZONTAL, command=self.display_discrete_output)
        self.slider.configure(length=300, sliderlength=30, resolution=0.1)
        self.slider.place(x=640-512/2, y=630)
        self.slider.pack()

        self.flip_channels = IntVar()
        self.invert_checkbox = Checkbutton(self.master, text='Flip channels', variable=self.flip_channels,
                                           command=self.display_discrete_output)
        self.invert_checkbox.place(x=800, y=610)
        self.skeletonise = IntVar()
        self.skeletonise_checkbox = Checkbutton(self.master, text='Skeletonise', variable=self.skeletonise,
                                           command=self.display_discrete_output)
        self.skeletonise_checkbox.place(x=950, y=610)
        # self.invert_checkbox.place(x=self.slider.winfo_rootx()+self.slider.winfo_width(), y=self.slider.winfo_rooty())
        # self.invert_checkbox.pack()


    def build_arch(self):
        dir = filedialog.askopenfilename(initialdir='/home/tom/data/work/geology/geogan_checkpoints', title='Select architecture description file')
        print(dir)
        if dir == '':
            return

        self.arch_file_name = dir

        self.arch.arch_from_slurm((self.arch_file_name))
        print(self.arch)

    def collect_images(self):
        self.files = glob.glob(os.path.join(self.dataroot, '*.pkl'))
        if len(self.files) == 0:
            messagebox.showerror('Error', "No pickle files in this location")
            self.dataroot = None

    def choose_dataroot(self):
        dir = filedialog.askdirectory(initialdir='/home/tom/data/', title='Choose dataroot')
        print(dir)
        if dir == '':
            return

        self.dataroot = dir
        self.collect_images()

    def load_weights(self):
        weights = torch.load(self.weights_filename)

        self.arch.model.load_state_dict(weights)
        self.arch.model = self.arch.model.float()

    def load_model(self):
        if self.arch == None:
            messagebox.showerror('Error', 'Define an architecture first')

        dir = filedialog.askopenfilename(initialdir=os.path.dirname(self.arch_file_name),
                                                         title='Select model weights file')

        print(dir)

        if dir =='':
            return

        self.weights_filename = dir
        self.load_weights()

        self.display_div_output()


    def update_mask_pos(self, event):
        x, y = event.x, event.y

        x -= 32
        y -= 32

        x = max(0, x)
        y = max(0, y)
        x = min(self.input_im.shape[1] - self.mask_size, x)
        y = min(self.input_im.shape[0] - self.mask_size, y)

        self.mask_pos = (x, y)
        self.display_discrete_input(x, y)
        self.masked_input = self.input_im.copy()

        self.masked_input[y:y + self.mask_size, x:x + self.mask_size, :] = 0
        self.display_div_output()
        self.display_discrete_output(self.slider.get())

    def draw_mask(self, im):
        x, y = self.mask_pos
        # Left
        im[y:y+self.mask_size, x, 0] = 0.498
        im[y:y+self.mask_size, x, 1] = 0
        im[y:y+self.mask_size, x, 2] = 0

        # Right
        im[y:y+self.mask_size, x+self.mask_size, 0] = 0.498
        im[y:y+self.mask_size, x+self.mask_size, 1] = 0
        im[y:y+self.mask_size, x+self.mask_size, 2] = 0

        # Top
        im[y, x:x+self.mask_size, 0] = 0.498
        im[y, x:x+self.mask_size, 1] = 0
        im[y, x:x+self.mask_size, 2] = 0

        # Bottom
        im[y+self.mask_size, x:x+self.mask_size, 0] = 0.498
        im[y+self.mask_size, x:x+self.mask_size, 1] = 0
        im[y+self.mask_size, x:x+self.mask_size, 2] = 0

        return im

    def display_discrete_input(self, *args):
        input_display_im = self.input_im.copy()

        input_display_im[:, :, 0][np.where(input_display_im[:, :, 1])] = 1
        input_display_im[:, :, 2][np.where(input_display_im[:, :, 1])] = 1

        if len(args) > 0:
            input_display_im = self.draw_mask(input_display_im)

        self.input_display_im = input_display_im

        image = Image.fromarray(input_display_im.astype(np.uint8) * 255)
        in_image_tk = ImageTk.PhotoImage(image=image)
        self.input_image_pane.place(x=256-image.width/2)
        self.input_image_pane.configure(height=image.height, width=image.width, image=in_image_tk)
        self.input_image_pane.image = in_image_tk


    def display_div_output(self):

        self.div_im = self.arch.model(torch.from_numpy(self.masked_input.transpose(2, 0, 1)).unsqueeze(0).float().cuda())
        self.div_im = self.div_im.detach().cpu().data.squeeze(0).numpy().transpose(1, 2, 0)[:, :, 0]
        self.div_im = np.interp(self.div_im, [self.div_im.min(), 0, self.div_im.max()], [-1, 0, 1])
        image = Image.fromarray(((self.div_im + 1) / 2 * 255).astype(np.uint8))
        out_image_tk = ImageTk.PhotoImage(image=image)
        self.output_div_image_pane.configure(image=out_image_tk)
        self.output_div_image_pane.image = out_image_tk


    def display_discrete_output(self, *args):
        self.thresh = np.interp(self.slider.get(), [0, 100], [0, 1.0])

        ridge_layer = np.ones(self.div_im.shape, dtype=bool)
        sub_layer = np.ones(self.div_im.shape, dtype=bool)
        ridge_layer[np.where(self.div_im < -self.thresh)] = False
        sub_layer[np.where(self.div_im > self.thresh)] = False
        if self.skeletonise.get() > 0:
            ridge_layer = 1 - skeletonize(1 - ridge_layer)
            sub_layer = 1 - skeletonize(1 - sub_layer)
        plate_layer = np.logical_and(ridge_layer, sub_layer)

        if self.flip_channels.get() > 0:
            tmp = sub_layer.copy()
            sub_layer = ridge_layer.copy()
            ridge_layer = tmp

        self.out_disc = np.dstack((ridge_layer, plate_layer, sub_layer)).astype(np.uint8) * 255
        image = Image.fromarray(self.out_disc)
        out_disc_image_tk = ImageTk.PhotoImage(image=image)

        self.output_disc_image_pane.configure(image=out_disc_image_tk)
        self.output_disc_image_pane.image = out_disc_image_tk

    def refresh(self, *args):
        self.display_div_output()
        self.display_discrete_output(self.slider.get())

    def set_random_image(self):
        if self.dataroot == None or len(self.files) == 0:
            messagebox.showinfo('Error', 'Set dataroot first')
            return

        self.pkl_file = random.sample(self.files, 1)[0]
        data = torch.load(self.pkl_file)

        self.input_im = data['A']
        self.gt_div = data['A_DIV']
        self.masked_input = self.input_im.copy()
        self.display_discrete_input()
        self.display_div_output()
        self.display_discrete_output(self.slider.get())

    def save_current_images(self):
        model_name, checkpoint = self.weights_filename.split('/')[-2:]
        epoch = checkpoint.split('_')[0]

        series_no, _ = os.path.basename(self.pkl_file).split('.')

        save_dir = os.path.join(self.start_dir, 'saved_images', model_name, 'epoch_{}'.format(epoch))
        try:
            os.makedirs(save_dir)
        except:
            pass

        out_div_file = 'series_{}_output_divergence_{}.png'.format(series_no, '{}')
        out_div_count = len(glob.glob(os.path.join(save_dir, out_div_file.format('*'))))
        out_div_file = out_div_file.format('{:03}')
        out_div_img = self.draw_mask(cm.seismic(((self.div_im + 1) / 2 * 255).astype(np.uint8)))

        gt_div_file = 'series_{}_gt_divergence.png'.format(series_no)
        gt_div_img = self.draw_mask(cm.seismic(((self.gt_div + 1) / 2 * 255).astype(np.uint8)))

        out_disc_file = 'series_{}_output_one_hot_thresh_{:.4}_{}.png'.format(series_no, self.thresh, '{}')
        out_disc_count = len(glob.glob(os.path.join(save_dir, out_disc_file.format('*'))))
        out_disc_file = out_disc_file.format('{:03}')
        out_disc_img = self.draw_mask(self.out_disc)

        plt.imsave(os.path.join(save_dir, 'series_{}_ground_truth_one_hot.png'.format(series_no)), self.input_display_im)
        plt.imsave(os.path.join(save_dir, gt_div_file.format(out_div_count)), gt_div_img)
        plt.imsave(os.path.join(save_dir, out_div_file.format(out_div_count)), out_div_img)
        plt.imsave(os.path.join(save_dir, out_disc_file.format(out_disc_count)), out_disc_img)

        print('Saved 3 images to {}'.format(save_dir))

    def load_ellipse_preset(self):
        self.arch_file_name = "/media/data/work/geology/geogan_checkpoints/test_ellipse_no_weighting/slurm-18933.out"
        self.arch.arch_from_slurm((self.arch_file_name))

        self.dataroot = "/home/tom/data/ellipses3/validation"
        self.collect_images()

        self.weights_filename = "/media/data/work/geology/geogan_checkpoints/test_ellipse_no_weighting/40_net_G.pth"
        self.load_weights()

        self.set_random_image()

    def load_earth_preset(self):
        self.arch_file_name = "/media/data/work/geology/geogan_checkpoints/test_no_weighting_ex_2/slurm-18072.out"
        self.arch.arch_from_slurm((self.arch_file_name))

        self.dataroot = "/home/tom/data/old_pytorch_records/validation"
        self.collect_images()

        self.weights_filename = "/media/data/work/geology/geogan_checkpoints/test_no_weighting_ex_2/2000_net_G.pth"
        self.load_weights()

        self.set_random_image()

    def load_synthetic_preset(self):
        self.arch_file_name = "/media/data/work/geology/geogan_checkpoints/test_no_weighting_ex_2/slurm-18072.out"
        self.arch.arch_from_slurm((self.arch_file_name))

        self.dataroot = "/home/tom/data/old_pytorch_records/test"
        self.collect_images()

        self.weights_filename = "/media/data/work/geology/geogan_checkpoints/orig_geo_self_attn/latest_net_G.pth"
        self.load_weights()

        self.set_random_image()

root = Tk()
root.geometry("1260x640")
app = Window(root)
root.mainloop()