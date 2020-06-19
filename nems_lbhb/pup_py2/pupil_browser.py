'''
GUI for loading pupil data fit with CNN, evaluating the fit, saving the fit (or retraining the model).
'''
import pickle
try:
    import tkinter as tk
except:
    import Tkinter as tk
import matplotlib.backends.tkagg as tkagg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import sys
import matplotlib as mpl
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.image as mpimg
import getpass
import nems.db as nd
import scipy.io
import sys
from tkinter import filedialog
import nems_db

import nems_lbhb
nems_lbhb_path = nems_lbhb.__path__[0]
sys.path.append(os.path.join(nems_lbhb_path, 'pup_py2/'))
import pupil_settings as ps

executable_path = sys.executable
script_path = os.path.split(os.path.split(nems_db.__file__)[0])[0]
training_browser_path = os.path.join(script_path, 'nems_lbhb', 'pup_py2', 'browse_training_data.py')

tmp_frame_folder = ps.TMP_SAVE  #'/auto/data/nems_db/pup_py/tmp/'
video_folder = ps.ROOT_VIDEO_DIRECTORY  #'/auto/data/daq/'

class PupilBrowser:

    def __init__(self, master):
        self.master = master
        master.title("Pupil browser")
        master.geometry('1050x600')

        # create a plot attributemod
        self.pupil_plot = None
        self.pupil_trace_plot = None

        self.pupil_canvas = tk.Canvas(master, width=400, height=300)
        self.pupil_canvas.grid(row=0, column=4, rowspan=6, columnspan=5)

        fig = mpl.figure.Figure(figsize=(10.5, 3), dpi=100)
        self.ax = fig.add_subplot(1,1,1)
        self.pupil_trace = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
        self.pupil_trace.draw()
        self.pupil_trace.get_tk_widget().grid(row=10, column=0, rowspan=5, columnspan=8, sticky='nwes')
        self.hline = None
        self.a_plot = None
        self.b_plot = None

        master.grid_columnconfigure(3, weight=1)
        master.grid_rowconfigure(10, weight=1)

        self.file_browse = tk.Button(master, text="Browse", command=self.browse_file)
        self.file_browse.grid(row=1, column=2)

        self.load_button = tk.Button(master, text="Load recording", command=self.load_file)
        self.load_button.grid(row=1, column=3)

        self.animal_n = tk.Label(master, text="animal")
        self.animal_n.grid(row=0, column=0, columnspan=1)
        self.animal_name = tk.Entry(master)
        self.animal_name.grid(row=1, column=0)
        self.animal_name.focus_set()

        self.video_n = tk.Label(master, text="video name")
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

        self.save_final = tk.Button(master, text='Save analysis', command=self.save_analysis)
        self.save_final.grid(row=4, column=0)

        self.label_data = tk.Button(master, text="Label more training data", command=self.open_training_browser)
        self.label_data.grid(row=4, column=1)

        self.face = tk.IntVar()
        self.full_face = tk.Checkbutton(master, text='Full face video', onvalue=1, offvalue=0, variable=self.face)
        self.full_face.grid(row=4, column=3)

        self.retrain = tk.Button(master, text="Re-train network", command=self.retrain)
        self.retrain.grid(row=4, column=2)

        self.shift_is_held = False
        self.exclude_starts = []
        self.exclude_ends = []

    def get_frame(self):
        animal = self.animal_name.get()
        video_n = self.video_name.get()
        video = self.raw_video

        frame = int(self.frame_n_value.get())
        t = frame * (1 / 30)

        # save new frames
        os.system("ffmpeg -ss {0} -i {1} -vframes 1 {2}frame%d.jpg".format(t, video, tmp_frame_folder))

        frame_file = tmp_frame_folder + 'frame1.jpg'

        self.pupil_plot = self.plot_frame(frame_file)


    def plot_frame(self, frame_file):

        frame = mpimg.imread(frame_file)
        canvas = self.pupil_canvas
        canvas.delete('all')  # prevent memory leak
        loc = (0, 0)

        figure = mpl.figure.Figure(figsize=(4, 3))
        ax = figure.add_axes([0, 0, 1, 1])
        ax.imshow(frame)

        # get frame number
        fn = int(self.frame_n_value.get())

        # get predictions
        filename = self.video_name.get()
        predictions_folder = (os.path.sep).join(self.processed_video.split(os.path.sep)[:-1])

        with open(os.path.join(predictions_folder, filename + '_pred.pickle'), 'rb') as fp:
            self.parms = pickle.load(fp)

        a = self.parms['cnn']['a'][fn]
        b = self.parms['cnn']['b'][fn]
        x = self.parms['cnn']['x'][fn]
        y = self.parms['cnn']['y'][fn]
        phi = self.parms['cnn']['phi'][fn]

        ellipse = Ellipse((y, x), b * 2, - a * 2, 180 * phi / np.pi, fill=False, color='red')
        ax.add_patch(ellipse)

        figure_canvas_agg = FigureCanvasTkAgg(figure, master=self.master)
        figure_canvas_agg.draw()

        figure_x, figure_y, figure_w, figure_h = figure.bbox.bounds

        figure_w = self.pupil_canvas.winfo_width()
        figure_h = self.pupil_canvas.winfo_height()
        photo = tk.PhotoImage(master=canvas, width=figure_w, height=figure_h)

        # Position: convert from top-left anchor to center anchor
        canvas.create_image(loc[0] + figure_w/2, loc[1] + figure_h/2, image=photo)

        # Unfortunately, there's no accessor for the pointer to the native renderer
        tkagg.blit(photo, figure_canvas_agg.get_renderer()._renderer, colormode=2)

        return photo

    def plot_trace(self, params_file, exclude=False):

        predictions_folder = (os.path.sep).join(self.processed_video.split(os.path.sep)[:-1])

        params_file = os.path.join(predictions_folder, params_file + '_pred.pickle')
        with open(params_file, 'rb') as fp:
            ellipse_preds = pickle.load(fp)

        a = ellipse_preds['cnn']['a']
        b = ellipse_preds['cnn']['b']

        if exclude:
            # exclude frames marked as bad
            for s, e in zip(self.exclude_starts, self.exclude_ends):
                a[s:e] = np.nan
                b[s:e] = np.nan

        self.max_frame = len(a)

        canvas = self.pupil_trace

        if hasattr(self, 'a_plot'):
            try:
                self.a_plot.pop(0).remove()
                self.b_plot.pop(0).remove()
            except:
                pass

        self.a_plot = self.ax.plot(a, 'r')
        self.b_plot = self.ax.plot(b, color='b', picker=5)
        self.ax.set_ylim((np.nanmin([np.nanmin(a), np.nanmin(b)]),
                         np.nanmax([np.nanmax(a), np.nanmax(b)])))
        self.ax.set_xlim((0, len(a)))

        self.ax.legend(['minor axis', 'major axis'])

        canvas.get_tk_widget().focus_force()
        canvas.mpl_connect('key_press_event', self.on_key)
        canvas.mpl_connect('pick_event', self.get_coords)
        canvas.mpl_connect('key_release_event', self.off_key)
        canvas.draw()

    def get_coords(self, event):
        self.frame_n_value.delete(0, 'end')
        self.frame_n_value.insert(0, str(int(event.mouseevent.xdata)))

        if self.shift_is_held == False:
            if hasattr(self, 'hline'):
                try:
                    self.hline.remove()
                except:
                    pass
            self.hline = self.ax.axvline(event.ind[0], color='k')
            self.pupil_trace.draw()

            self.get_frame()

        elif self.shift_is_held == True:
            if event.mouseevent.button == 1:
                if hasattr(self, 'hline_start'):
                    self.hline_start.remove()
                    del self.start_val

                self.hline_start = self.ax.axvline(int(event.mouseevent.xdata),
                                             color='red')
                self.start_val = int(event.mouseevent.xdata)
                self.pupil_trace.draw()

            elif event.mouseevent.button == 3:
                if hasattr(self, 'hline_end'):
                    self.hline_end.remove()
                    self.hline_fill.remove()
                    del self.end_val

                if hasattr(self, 'hline_start') == False:
                    print("First specify start using shift+left-click!")
                else:
                    self.hline_end = self.ax.axvline(int(event.mouseevent.xdata),
                                                 color='red')
                    mi, ma = self.ax.get_ylim()
                    mi = int(mi)
                    ma = int(ma)+1
                    self.end_val = int(event.mouseevent.xdata)
                    self.hline_fill = self.ax.fill_betweenx(range(mi, ma),
                                                self.end_val,
                                                self.start_val, color='grey',
                                                alpha=0.5)
                    self.pupil_trace.draw()

    def on_key(self, event):
        if event.key=='shift':
            self.shift_is_held=True
        elif event.key=='enter':
            # check if exclusion thing exists and delete it on the plot
            if hasattr(self, 'hline_start') & hasattr(self, 'hline_end'):
                self.hline_start.remove()
                del self.hline_start
                self.hline_end.remove()
                del self.hline_end
                self.hline_fill.remove()
                del self.hline_fill
                self.pupil_trace.draw()
                # save the currently stored start/end values to self.exclude_starts
                # and self.exclude_ends
                self.exclude_starts.append(self.start_val)
                self.exclude_ends.append(self.end_val)

                self.plot_trace(self.video_name.get(), exclude=True)
            else:
                pass

        else:
            pass

    def off_key(self, event):
        if event.key=='shift':
            self.shift_is_held=False
        else:
            pass

    def browse_file(self):
        # get the pupil video file
        self.raw_video = filedialog.askopenfilename(initialdir = ps.ROOT_VIDEO_DIRECTORY,
                            title = "Select raw video file",
                            filetypes = (("mj2 files","*.mj2*"), ("avi files","*.avi")))

        params_file = os.path.split(self.raw_video)[-1].split('.')[0]
        animal = os.path.split(self.raw_video)[0].split(os.path.sep)[4]

        self.video_name.delete(0, 'end')
        self.animal_name.delete(0, 'end')
        self.video_name.insert(0, params_file)
        self.animal_name.insert(0, animal)

        self.load_file()

    def load_file(self):
        """
        Load the overall predictions and plot the trace on the trace canvas.
        Display the first frame of the video on the pupil canvas.
        """

        params_file = self.video_name.get()
        # get raw video -- try to use the exisiting path from raw video
        fp = os.path.split(self.raw_video)[0]
        
        self.processed_video = os.path.join(fp, 'sorted', params_file)

        # reset raw video attribute
        ext = self.raw_video.split('.')[-1]
        if len(self.raw_video.split('.')) > 2:
            ext2 = self.raw_video.split('.')[-2]
            self.raw_video = os.path.join(fp, params_file)+'.'+ext2+'.'+ext
        else:
            self.raw_video = os.path.join(fp, params_file)+'.'+ext
        print(self.raw_video)

        self.plot_trace(params_file)

        self.frame_n_value.insert(0, str(0))

        # reset exclusion frames
        self.exclude_starts = []
        self.exclude_ends = []

        # save first ten frames and display the first
        video = self.raw_video

        os.system("ffmpeg -ss 00:00:00 -i {0} -vframes 1 {1}frame%d.jpg".format(video, tmp_frame_folder))

        frame_file = tmp_frame_folder + 'frame1.jpg'

        self.pupil_canvas.delete(self.pupil_plot)
        self.pupil_plot = self.plot_frame(frame_file)
        self.master.mainloop()

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

    def save_analysis(self):
        video_name = self.video_name.get()

        fn = video_name + '.pickle'
        fn_mat = video_name + '.mat'
        fp = os.path.split(self.processed_video)[0]
        save_path = os.path.join(fp, fn)
        # for matlab loading
        mat_fn = os.path.join(fp, fn_mat)

        sorted_dir = os.path.split(save_path)[0]

        if os.path.isdir(sorted_dir) != True:
            # create sorted directory and force to be world writeable
            os.system("mkdir {}".format(sorted_dir))
            os.system("chmod a+w {}".format(sorted_dir))
            print("created new directory {0}".format(sorted_dir))
        else:
            pass

        save_dict = self.parms

        # add excluded frames to the save dictionary
        excluded_frames = np.concatenate((np.array(self.exclude_starts)[np.newaxis, :],
                                        np.array(self.exclude_ends)[np.newaxis, :]),
                                        axis=0)
        save_dict['cnn']['excluded_frames'] = excluded_frames.T


        print("computing eyespeed")
        x_diff = np.diff(save_dict['cnn']['x'])
        y_diff = np.diff(save_dict['cnn']['y'])
        d = np.sqrt((x_diff ** 2) + (y_diff ** 2))
        d[-1] = 0
        d = np.concatenate((d, np.zeros(1)))
        save_dict['cnn']['eyespeed'] = d

        with open(save_path, 'wb') as fp:
                pickle.dump(save_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

        scipy.io.savemat(mat_fn, save_dict)

        print("saved analysis successfully")

    def open_training_browser(self):
        os.system("{0} \
                {1} {2} {3} {4} {5} {6} {7}".format(executable_path, training_browser_path,
            self.animal_name.get(), self.video_name.get(), self.raw_video, 0, self.max_frame, self.face.get()))

    def retrain(self):
        # retrain the model. This will happen on the queue (needs to be fit on gpu). Therefore, we'll start the queue
        # job and automatically exit the window
        py_path = sys.executable
        script_path = os.path.split(os.path.split(nems_db.__file__)[0])[0]
        script_path = os.path.join(script_path, 'nems_lbhb', 'pup_py2', 'training_script.py')
        username = getpass.getuser()

        # add job to queue
        nd.add_job_to_queue([], note="Pupil Job: Training CNN", executable_path=py_path,
                            user=username, force_rerun=True, script_path=script_path, GPU_job=1)
        print("Queueing new model training. Check status on queue. When finished, re-fit the pupil for this recording")
        # self.master.destroy

root = tk.Tk()
my_gui = PupilBrowser(root)
root.mainloop()
