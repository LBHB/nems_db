from tkinter import filedialog, simpledialog, messagebox
import tkinter as tk
import image_stitching_tools
from multiprocessing import Process
from tkinter import *

def image_pipe(e):
    # make instance of stitcher class
    ents = e[0]
    stitcher = e[1]
    stitcher.set_images()

    # update default output name and path
    update_defaults = ("Output name", "Output directory")
    nm1 = stitcher.image1_metadata['name'].split(".")[0]
    nm2 = stitcher.image2_metadata['name'].split(".")[0]
    default_name = "_".join([nm1, nm2])+'.jpg'
    default_vals = {"Output name": default_name, "Output directory": stitcher.image_dir}
    for field in update_defaults:
        print(field)
        ents[field].delete(0, END)
        ents[field].insert(0, default_vals[field])

    # ask about image rotation
    # ROOT = tk.Tk()
    # ROOT.withdraw()
    # the input dialog
    # cor_input1 = False
    # cor_input2 = False
    #
    # while cor_input1 == False:
    #     im1_rot = simpledialog.askstring(title="Rotation 1",
    #                                       prompt="Rotate image 1 clockwise x degrees:", initialvalue="0")
    #     stitcher.rotation(0, int(im1_rot))
    #     res = messagebox.askquestion('Correct Rotaion', 'Correct rotation?')
    #     if res == 'yes':
    #         cor_input1 = True
    #
    # while cor_input2 == False:
    #     im2_rot = simpledialog.askstring(title="Rotation 2",
    #                                      prompt="Rotate image 2 clockwise x degrees:", initialvalue="0")
    #     stitcher.rotation(1, int(im2_rot))
    #     res = messagebox.askquestion('Correct Rotaion', 'Correct rotation?')
    #     if res == 'yes':
    #         cor_input2 = True

    return stitcher

fields = ('Sigma', "Edge Threshold", "Contrast Threshold", "Output name", "Output directory")
field_opts = {'Sigma': 1.9, 'Edge Threshold': 18, 'Contrast Threshold': 0.01, 'Output name': '', 'Output directory': ''}
def run_sift(e):
    ent = e[0]
    sift_opts = {"Sigma": float(ent['Sigma'].get()), "Edge Threshold": float(ent['Edge Threshold'].get()), "Contrast Threshold": float(ent['Contrast Threshold'].get())}
    stitcher = e[1]
    stitcher.run_sift(sift_opts = sift_opts, debug = True)

def run_align(e):
    stitcher = e[1]
    stitcher.run_align()

def add_masks(e):

    stitcher = e[1]
    stitcher.mask_select()

def set_masks(e):
    stitcher = e[1]
    stitcher.mask_set()

def clear_features(e):
    stitcher = e[1]
    stitcher.clear_features()

def move_features(e):
    stitcher = e[1]
    stitcher.move_features()

def add_point(e):
    stitcher = e[1]
    stitcher.add_point()

def remove_point(e):
    stitcher = e[1]
    stitcher.rm_point()

def save_stitched(e):
    ents = e[0]
    stitcher = e[1]
    stitcher.save_stitched(ents['Output directory'].get(), ents['Output name'].get())

def makeform(root, fields):
    entries = {}
    for field in fields:
        print(field)
        row = tk.Frame(root)
        lab = tk.Label(row, width=22, text=field + ": ", anchor='w')
        ent = tk.Entry(row)
        ent.insert(0, field_opts[field])
        row.pack(side=tk.TOP,
                 fill=tk.X,
                 padx=5,
                 pady=5)
        lab.pack(side=tk.LEFT)
        ent.pack(side=tk.RIGHT,
                 expand=tk.YES,
                 fill=tk.X)
        entries[field] = ent
    return entries

if __name__ == '__main__':
    root = tk.Tk()
    # setup view
    ents = makeform(root, fields)

    # setup model
    stitcher = image_stitching_tools.image_stitcher()

    # bind view and model
    b = [ents, stitcher]

    # define interactions
    b0 = tk.Button(root, text='Select Image',
                   command=(lambda e=b: image_pipe(e)))
    b0.pack(side=tk.LEFT, padx=50, pady=5)
    b0_1 = tk.Button(root, text='Add Masks',
                   command=(lambda e=b: add_masks(e)))
    b0_1.pack(side=tk.LEFT, padx=5, pady=5)
    b0_2 = tk.Button(root, text='Set Masks',
                   command=(lambda e=b: set_masks(e)))
    b0_2.pack(side=tk.LEFT, padx=5, pady=5)
    b1 = tk.Button(root, text='Run Sift',
                   command=(lambda e=b: run_sift(e)))
    b1.pack(side=tk.LEFT, padx=50, pady=5)
    b2 = tk.Button(root, text='Clear',
                   command=(lambda e=b: clear_features(e)))
    b2.pack(side=tk.LEFT, padx=5, pady=5)
    b3 = tk.Button(root, text='+', command=(lambda e=b: add_point(e)))
    b3.pack(side=tk.LEFT, padx=5, pady=5)
    b4 = tk.Button(root, text='-', command=(lambda e=b: remove_point(e)))
    b4.pack(side=tk.LEFT, padx=5, pady=5)
    b4_1 = tk.Button(root, text='Move', command=(lambda e=b: move_features(e)))
    b4_1.pack(side=tk.LEFT, padx=5, pady=5)
    b5 = tk.Button(root, text='stich', command=(lambda e=b: run_align(e)))
    b5.pack(side=tk.LEFT, padx=50, pady=5)
    b6 = tk.Button(root, text='save', command=(lambda e=b: save_stitched(e)))
    b6.pack(side=tk.LEFT, padx=10, pady=5)
    root.mainloop()