'''
GUI for loading photo data and caching the mean over user defined ROI
'''
import pickle
try:
    import tkinter as tk
except:
    import Tkinter as tk
import matplotlib.backends.tkagg as tkagg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib as mpl
import matplotlib.image as mpimg
from tkinter import filedialog
import os

tmp_frame_folder = '/auto/data/nems_db/pup_py/tmp/'


class PhotoBrowser:

    def __init__(self, master):
        self.master = master
        master.title("Photo Browser")
        master.geometry("950x600")

        # create plot attributes
        self.video_plot = None

        # create canvas for photo video
        self.video_canvas = tk.Canvas(master, width=400, height=300)
        self.video_canvas.grid(row=0, column=3, rowspan=6, columnspan=5)

        # create drawing area for analyzed trace
        fig = mpl.figure.Figure(figsize=(9.5, 3), dpi=100)
        self.ax = fig.add_subplot(1,1,1)
        self.roi_trace = FigureCanvasTkAgg(fig, master=root)
        self.roi_trace.draw()
        self.roi_trace.get_tk_widget().grid(row=10, column=0, 
                    rowspan=5, columnspan=8, sticky='nwes')
        self.h_line = None
        
        # create button to load data
        self.load_button = tk.Button(master, text='Load video', 
                                        command=self.load_video_file)
        self.load_button.grid(row=1, column=2)

        # text boxes to hold information about the animal / filename
        self.animal_n = tk.Label(master, text="Animal")
        self.animal_n.grid(row=0, column=0, columnspan=1)
        self.animal_name = tk.Entry(master)
        self.animal_name.grid(row=1, column=0)
        self.animal_name.focus_set()

        self.video_n = tk.Label(master, text="Video file name")
        self.video_n.grid(row=0, column=1, columnspan=1)
        self.video_name = tk.Entry(master)
        self.video_name.grid(row=1, column=1)
        self.video_name.focus_set()

        # Jump to frame number
        self.frame_n = tk.Label(master, text="Frame number: ")
        self.frame_n.grid(row=2, column=0, columnspan=1)
        self.frame_n_value = tk.Entry(master)
        self.frame_n_value.grid(row=2, column=1)
        self.frame_n_value.focus_set()

        self.next_frame = tk.Button(master, text="Next frame", command=self.get_next_frame)
        self.next_frame.grid(row=3, column=1)

        self.previous_frame = tk.Button(master, text="Previous frame", command=self.get_prev_frame)
        self.previous_frame.grid(row=3, column=0)

        self.frame_update = tk.Button(master, text="Jump to frame", command=self.get_frame)
        self.frame_update.grid(row=2, column=2)


    def load_video_file(self,):
        self.raw_video = filedialog.askopenfilename(initialdir = "/auto/data/daq/",
                            title = "Select raw video file", 
                            filetypes = (("avi files","*.avi*"), ("mj2 files","*.mj2")))
        params_file = os.path.split(self.raw_video)[-1].split('.')[0]
        animal = os.path.split(self.raw_video)[0].split(os.path.sep)[4]
        
        self.video_name.delete(0, 'end')
        self.animal_name.delete(0, 'end')
        self.video_name.insert(0, params_file)
        self.animal_name.insert(0, animal)

        self.frame_n_value.delete(0, 'end')
        self.frame_n_value.insert(0, str(0))

        os.system("ffmpeg -ss 00:00:00 -i {0} -vframes 1 {1}frame%d.jpg".format(self.raw_video, tmp_frame_folder))
        frame_file = tmp_frame_folder + 'frame1.jpg'

        self.video_canvas.delete(self.video_plot)
        self.video_plot = self.plot_frame(frame_file)
        self.master.mainloop()

    def plot_frame(self, frame_file):
        frame = mpimg.imread(frame_file)
        canvas = self.video_canvas
        canvas.delete('all')
        loc = (0, 0)

        figure = mpl.figure.Figure(figsize=(4, 3))
        ax = figure.add_axes([0, 0, 1, 1])
        ax.imshow(frame)

        figure_canvas_agg = FigureCanvasTkAgg(figure, master=self.master)
        figure_canvas_agg.draw()

        figure_x, figure_y, figure_w, figure_h = figure.bbox.bounds

        figure_w = self.video_canvas.winfo_width()
        figure_h = self.video_canvas.winfo_height()
        photo = tk.PhotoImage(master=canvas, width=figure_w, height=figure_h)

        # Position: convert from top-left anchor to center anchor
        canvas.create_image(loc[0] + figure_w/2, loc[1] + figure_h/2, image=photo)

        # Unfortunately, there's no accessor for the pointer to the native renderer
        tkagg.blit(photo, figure_canvas_agg.get_renderer()._renderer, colormode=2)

        return photo

    def get_next_frame(self):
        '''
        display the next frame in the training set
        '''
        frame = int(self.frame_n_value.get())
        self.frame_n_value.delete(0, 'end')
        self.frame_n_value.insert(0, string=str(frame+1))

        self.get_frame()

    def get_prev_frame(self):
        '''
        display the previous frame in the training set
        '''
        frame = int(self.frame_n_value.get())
        self.frame_n_value.delete(0, 'end')
        self.frame_n_value.insert(0, string=str(frame-1))

        self.get_frame()
    
    def get_frame(self):
        video = self.raw_video

        frame = int(self.frame_n_value.get())
        t = frame * (1 / 30)

        # save new frames
        os.system("ffmpeg -ss {0} -i {1} -vframes 1 {2}frame%d.jpg".format(t, video, tmp_frame_folder))
        #import pdb; pdb.set_trace()
        frame_file = tmp_frame_folder + 'frame1.jpg'

        self.video_plot = self.plot_frame(frame_file)

def analyze_video(videofile, roi):
    """
    Function to load and analyze the video, save the prediction in 
    the sorted file under video_name.photo.pickle
    """

root = tk.Tk()
my_gui = PhotoBrowser(root)
root.mainloop()
