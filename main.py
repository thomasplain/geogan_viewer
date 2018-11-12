from tkinter import filedialog
from tkinter import *
from model import Model

import os

class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.init_window()

        self.arch_file_name = None

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
        chooseDatarootButton.place(x=100, y=0)


    def build_arch(self):
        self.arch_file_name = filedialog.askopenfilename(initialdir=os.getcwd(), title='Select architecture description file')

        self.arch = Model()
        self.arch.arch_from_file((self.arch_file_name))
        print(self.arch)


    def choose_dataroot(self):
        self.dataroot = filedialog.askdirectory(initialdir=os.getcwd(), title='Choose dataroot')


root = Tk()
root.geometry("640x480")
app = Window(root)
root.mainloop()