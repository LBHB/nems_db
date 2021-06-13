'''
Script meant to be used for browsing saved training data and manually manipulating the saved values. For instance,
useful when you notice there is frame the must be manually annotated.
'''
import tkinter as tk
from matplotlib.backends import _backend_tk as tkagg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from matplotlib.figure import Figure
from PIL import Image
import pickle
import sys
import nems_db
nems_db_path = nems_db.__path__[0]
sys.path.append(os.path.join(nems_db_path, 'nems_lbhb/pup_py/'))
import pupil_settings as ps
train_data_path = ps.TRAIN_DATA_PATH  #'/auto/data/nems_db/pup_py/training_data/'
tmp_save = ps.TMP_SAVE                #'/auto/data/nems_db/pup_py/tmp/'

class TrainingDataBrowser:

    def __init__(self, master, animal=None, video_name=None, raw_video=None, min_frame=0, max_frame=5000, train_frame_range=None, n_frames=None):

        # figure out which frames to display. If animal and video_name are none, display first frame from training
        # directory. If animal and video_name are specified, add 50 random frames from this video to the end of
        # the training directory and then display the first of these frames. if train_frame_range is not None, choose 
        # these 50 frames from this range. train frame range is tuple (start_frame, end_frame)

        self.plot_calls = 0
        self.edgepoints = []        
        if (animal is None) and (video_name is None):
            self.from_browser = False
            default_frame = os.listdir(train_data_path)[0].split('.')[0]

        else:
            self.from_browser = True
            self.video_name = video_name
            # where the prediction will be stored if this vid has already been
            # fit
 
            #predictions_folder = '/auto/data/daq/{0}/{1}/sorted/'.format(animal, video_name[:6])
            predictions_folder = os.path.join(os.path.split(raw_video)[0], 'sorted/')
            # save videos
            video = raw_video

            # current parameters (the current best fit based on the cnn fit)
            params_file = predictions_folder + video_name + '_pred.pickle'
            with open(params_file, 'rb') as fp:
                parms = pickle.load(fp)
            
            # delete all videos in the training folder that have this video name (Do this inside the loop instead
            # e.g. just check for duplicate frames)
            # os.system("rm {}*".format(train_data_path + video_name))
            if (n_frames is not None) & (n_frames != 'None'):
                n_frames = int(n_frames)
            fps = 30
            t0 = int(min_frame) * (1 / fps)
            tend = int(max_frame) * (1 / fps)
            if (train_frame_range is not None) & (train_frame_range != 'None'):
                start, end = int(train_frame_range.split('_')[0]), int(train_frame_range.split('_')[1])
                if (n_frames is not None) & (n_frames != 'None') & (n_frames < (end - start)):
                    nframes = n_frames
                else:
                    if n_frames < (end - start):
                        nframes = n_frames
                    elif 50 < (end - start):
                        nframes = 50
                    else:
                        nframes = end - start
                frames = np.sort(np.random.choice(np.arange(start * (1 / fps), 
                                        end * (1 / fps), 1/fps), nframes, replace=False))
            else:
                frames = np.sort(np.random.choice(np.arange(t0, tend, 1/fps), n_frames, replace=False))
            output_dict = {}
            n_frames_added = 0
            for i, t in enumerate(frames):
                f = int(t * fps)
                # save temporarily
                os.system("ffmpeg -ss {0} -i {1} -vframes 1 {2}{3}%d.jpg".format(t, video, tmp_save, video_name))

                # load, convert to grayscale, cut off artifact, extact/save first channel only
                img = Image.open(tmp_save + video_name + '1' + '.jpg') #.convert('LA')
                frame = np.asarray(img)[:, :-10, 0]
                output_dict['frame'] = frame
                img2 = Image.open(tmp_save + video_name + '1' + '.jpg') #.convert('LA')
                frame2 = np.asarray(img2)[:, :-10, 0]
                output_dict['frame2'] = frame2  # for display purposes
                try:
                    output_dict['ellipse_zack'] = {
                                    'a': parms['cnn']['a'][f],
                                    'b': parms['cnn']['b'][f],  
                                    'X0_in': parms['cnn']['x'][f],
                                    'Y0_in': parms['cnn']['y'][f],
                                    'phi': parms['cnn']['phi'][f],
                                    'eyelid_left_x': parms['cnn']['eyelid_left_x'][f],
                                    'eyelid_left_y': parms['cnn']['eyelid_left_y'][f],
                                    'eyelid_top_x': parms['cnn']['eyelid_top_x'][f],
                                    'eyelid_top_y': parms['cnn']['eyelid_top_y'][f],
                                    'eyelid_right_x': parms['cnn']['eyelid_right_x'][f],
                                    'eyelid_right_y': parms['cnn']['eyelid_right_y'][f],
                                    'eyelid_bottom_x': parms['cnn']['eyelid_bottom_x'][f],
                                    'eyelid_bottom_y': parms['cnn']['eyelid_bottom_y'][f]

                    }
                except:
                    output_dict['ellipse_zack'] = {
                                    'a': parms['cnn']['a'][f],
                                    'b': parms['cnn']['b'][f],  
                                    'X0_in': parms['cnn']['x'][f],
                                    'Y0_in': parms['cnn']['y'][f],
                                    'phi': parms['cnn']['phi'][f],

                    }
                name = train_data_path + video_name + str(f) + '.pickle'

                # if this frame isn't already in the training set, add it
                if os.path.isfile(name):
                    pass
                else:
                    with open(name, 'wb') as fp:
                        pickle.dump(output_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
                    if n_frames_added == 0:
                        default_frame = video_name + str(f)
                    n_frames_added += 1

            print("Added {0} video frames to training set".format(n_frames_added))

        self.master = master
        master.title("Training data browser")
        master.geometry('950x400')
        #master.geometry('1200x500')

        # create a plot attributemod
        self.pupil_plot = None

        #self.pupil_canvas = tk.Canvas(master, width=300, height=300)
        #self.pupil_canvas.grid(row=0, column=4, rowspan=11)

        # just build the figure here to try to prevent mem leak
        self.figure_handle = mpl.figure.Figure(figsize=(4, 4))
        self.ax_handle = self.figure_handle.add_subplot(111)
        self.figure_canvas_agg = FigureCanvasTkAgg(self.figure_handle, master=self.master)
        self.figure_canvas_agg.draw()
        self.figure_canvas_agg.mpl_connect(
                "button_press_event", self.button_press_callback)
        self.figure_canvas_agg.mpl_connect(
                "key_press_event", self.key_press_callback)
        self.figure_canvas_agg.get_tk_widget().grid(row=0, column=4, rowspan=11, ipadx=0, ipady=0, pady=0, padx=0)


        self.load_button = tk.Button(master, text="Display current frame", command=self.browse_files)
        self.load_button.grid(row=0, column=1)

        if self.from_browser:
            self.frame_count = tk.Text(master, height=1, width=12)
            all_frames = os.listdir(train_data_path)
            all_frames = [f for f in all_frames if self.video_name in f]
            self.frame_count.grid(row=0, column=2)
            self.frame_count.insert(tk.END, "1/{0}".format(len(all_frames)))
        else:
            self.frame_count = tk.Text(master, height=1, width=12)
            self.frame_count.grid(row=0, column=2)
            all_frames = os.listdir(train_data_path)
            self.frame_count.insert(tk.END, "1/{0}".format(len(all_frames)))

        self.frame_name = tk.Entry(master)
        self.frame_name.grid(row=0, column=0)
        self.frame_name.insert(0, string=default_frame)
        self.frame_name.focus_set()

        self.next_frame = tk.Button(master, text="Next frame", command=self.display_next_frame)
        self.next_frame.grid(row=1, column=1)

        self.previous_frame = tk.Button(master, text="Previous frame", command=self.display_prev_frame)
        self.previous_frame.grid(row=1, column=0)

        # ellipse parameters
        self.params_title = tk.Label(master, text="Ellipse parameters")
        self.params_title.grid(row=2, column=0, columnspan=2)

        self.long_axis = tk.Label(master, text="long axis: ")
        self.long_axis.grid(row=3, column=0, columnspan=1)
        self.long_axis_value = tk.Entry(master)
        self.long_axis_value.grid(row=3, column=1)
        self.long_axis_value.focus_set()
        self.long_axis_ua = tk.Button(master, text=u"\u25B2", command=self.increase_la)
        self.long_axis_ua.grid(row=3, column=2)
        self.long_axis_da = tk.Button(master, text=u"\u25BC", command=self.decrease_la)
        self.long_axis_da.grid(row=3, column=3)

        self.short_axis = tk.Label(master, text="short axis: ")
        self.short_axis.grid(row=4, column=0, columnspan=1)
        self.short_axis_value = tk.Entry(master)
        self.short_axis_value.grid(row=4, column=1)
        self.short_axis_value.focus_set()
        self.short_axis_ua = tk.Button(master, text=u"\u25B2", command=self.increase_sa)
        self.short_axis_ua.grid(row=4, column=2)
        self.short_axis_da = tk.Button(master, text=u"\u25BC", command=self.decrease_sa)
        self.short_axis_da.grid(row=4, column=3)

        self.x_pos = tk.Label(master, text="x-postition")
        self.x_pos.grid(row=5, column=0, columnspan=1)
        self.x_pos_value = tk.Entry(master)
        self.x_pos_value.grid(row=5, column=1)
        self.x_pos_value.focus_set()
        self.x_pos_ua = tk.Button(master, text=u"\u25B2", command=self.increase_x)
        self.x_pos_ua.grid(row=5, column=2)
        self.x_pos_da = tk.Button(master, text=u"\u25BC", command=self.decrease_x)
        self.x_pos_da.grid(row=5, column=3)

        self.y_pos = tk.Label(master, text="y-position")
        self.y_pos.grid(row=6, column=0, columnspan=1)
        self.y_pos_value = tk.Entry(master)
        self.y_pos_value.grid(row=6, column=1)
        self.y_pos_value.focus_set()
        self.y_pos_ua = tk.Button(master, text=u"\u25B2", command=self.increase_y)
        self.y_pos_ua.grid(row=6, column=2)
        self.y_pos_da = tk.Button(master, text=u"\u25BC", command=self.decrease_y)
        self.y_pos_da.grid(row=6, column=3)

        self.phi = tk.Label(master, text="Rotation (degrees): ")
        self.phi.grid(row=7, column=0, columnspan=1)
        self.phi_value = tk.Entry(master)
        self.phi_value.grid(row=7, column=1)
        self.phi_value.focus_set()
        self.phi_ua = tk.Button(master, text=u"\u25B2", command=self.increase_phi)
        self.phi_ua.grid(row=7, column=2)
        self.phi_da = tk.Button(master, text=u"\u25BC", command=self.decrease_phi)
        self.phi_da.grid(row=7, column=3)

        # ellipse update
        self.update_ellipse = tk.Button(master, text="Update ellipse display", command=self.update_ellipse_plot)
        self.update_ellipse.grid(row=9, column=0)

        self.set_ellipse_prev = tk.Button(master, text="Set ellipse params to previous", command=self.set_to_previous)
        self.set_ellipse_prev.grid(row=9, column=1)

        # save new ellipse params for this frame
        self.save_ellipse = tk.Button(master, text="Save new parameters", command=self.save_ellipse_params)
        self.save_ellipse.grid(row=10, column=1)

        self.close_button = tk.Button(master, text="Close", command=master.destroy)
        self.close_button.grid(row=10, column=0)

    def increase_la(self):
        val = np.float(self.long_axis_value.get())
        val += 0.5
        self.long_axis_value.delete(0, 'end')
        self.long_axis_value.insert(0, string=str(round(val, 2)))

        #self.pupil_canvas.delete(self.pupil_plot)
        self.pupil_plot = self.update_ellipse_plot()

    def decrease_la(self):
        val = np.float(self.long_axis_value.get())
        val -= 0.5
        self.long_axis_value.delete(0, 'end')
        self.long_axis_value.insert(0, string=str(round(val, 2)))

        #self.pupil_canvas.delete(self.pupil_plot)
        self.pupil_plot = self.update_ellipse_plot()

    def increase_sa(self):
        val = np.float(self.short_axis_value.get())
        val += 0.5
        self.short_axis_value.delete(0, 'end')
        self.short_axis_value.insert(0, string=str(round(val, 2)))

        #self.pupil_canvas.delete(self.pupil_plot)
        self.pupil_plot = self.update_ellipse_plot()

    def decrease_sa(self):
        val = np.float(self.short_axis_value.get())
        val -= 0.5
        self.short_axis_value.delete(0, 'end')
        self.short_axis_value.insert(0, string=str(round(val, 2)))

        #self.pupil_canvas.delete(self.pupil_plot)
        self.pupil_plot = self.update_ellipse_plot()

    def increase_x(self):
        val = np.float(self.x_pos_value.get())
        val += 0.5
        self.x_pos_value.delete(0, 'end')
        self.x_pos_value.insert(0, string=str(round(val, 2)))

        #self.pupil_canvas.delete(self.pupil_plot)
        self.pupil_plot = self.update_ellipse_plot()

    def decrease_x(self):
        val = np.float(self.x_pos_value.get())
        val -= 0.5
        self.x_pos_value.delete(0, 'end')
        self.x_pos_value.insert(0, string=str(round(val, 2)))

        #self.pupil_canvas.delete(self.pupil_plot)
        self.pupil_plot = self.update_ellipse_plot()

    def increase_y(self):
        val = np.float(self.y_pos_value.get())
        val += 0.5
        self.y_pos_value.delete(0, 'end')
        self.y_pos_value.insert(0, string=str(round(val, 2)))

        #self.pupil_canvas.delete(self.pupil_plot)
        self.pupil_plot = self.update_ellipse_plot()

    def decrease_y(self):
        val = np.float(self.y_pos_value.get())
        val -= 0.5
        self.y_pos_value.delete(0, 'end')
        self.y_pos_value.insert(0, string=str(round(val, 2)))

        #self.pupil_canvas.delete(self.pupil_plot)
        self.pupil_plot = self.update_ellipse_plot()

    def increase_phi(self):
        val = np.float(self.phi_value.get())
        val += 0.5
        self.phi_value.delete(0, 'end')
        self.phi_value.insert(0, string=str(round(val, 2)))

        #self.pupil_canvas.delete(self.pupil_plot)
        self.pupil_plot = self.update_ellipse_plot()

    def decrease_phi(self):
        val = np.float(self.phi_value.get())
        val -= 0.5
        self.phi_value.delete(0, 'end')
        self.phi_value.insert(0, string=str(round(val, 2)))

        #self.pupil_canvas.delete(self.pupil_plot)
        self.pupil_plot = self.update_ellipse_plot()


    def update_ellipse_params(self, params):
        la = params['b'] * 2
        sa = params['a'] * 2
        x = params['Y0_in']
        y = params['X0_in']
        phi = params['phi'] * 180 / np.pi

        self.long_axis_value.delete(0, 'end')
        self.long_axis_value.insert(0, string=str(round(la, 2)))

        self.short_axis_value.delete(0, 'end')
        self.short_axis_value.insert(0, string=str(round(sa, 2)))

        self.x_pos_value.delete(0, 'end')
        self.x_pos_value.insert(0, string=str(round(x, 2)))

        self.y_pos_value.delete(0, 'end')
        self.y_pos_value.insert(0, string=str(round(y, 2)))

        self.phi_value.delete(0, 'end')
        self.phi_value.insert(0, string=str(round(phi, 2)))

    def set_to_previous(self):

        current_frame = self.frame_name.get()+'.pickle'
        all_frames = os.listdir(train_data_path)
        if self.from_browser:
            all_frames = [f for f in all_frames if self.video_name in f]
            inds = np.argsort(np.array(all_frames))
        else:
            inds = np.argsort(np.array(all_frames))

        all_frames = np.array(all_frames)[inds]

        cur_index = np.argwhere(all_frames == current_frame)[0][0]
        prev_index = cur_index - 1
        prev_frame = all_frames[prev_index]

        with open('{0}/{1}'.format(train_data_path, prev_frame), 'rb') as fp:
            prev_params = pickle.load(fp)
        
        self.update_ellipse_params(prev_params['ellipse_zack'])

        ##self.pupil_canvas.delete(self.pupil_plot)
        self.pupil_plot = self.update_ellipse_plot()

    def save_ellipse_params(self):

        X0_in = np.float(self.y_pos_value.get())  # these are flipped on purpose
        Y0_in = np.float(self.x_pos_value.get())
        long_axis = np.float(self.long_axis_value.get())
        short_axis = np.float(self.short_axis_value.get())
        phi = np.float(self.phi_value.get()) / 180 * np.pi
        # save eyelid key points
        self.sort_edgepoints()
        left = self.edgepoints[0]
        top = self.edgepoints[1]
        right = self.edgepoints[2]
        bottom = self.edgepoints[3]
        print(self.edgepoints)
        e_params = {
            'X0_in': X0_in,
            'Y0_in': Y0_in,
            'b': (long_axis / 2),
            'a': (short_axis / 2),
            'phi': phi,
            'eyelid_left_x': left[0],
            'eyelid_left_y': left[1],
            'eyelid_top_x': top[0],
            'eyelid_top_y': top[1],
            'eyelid_right_x': right[0],
            'eyelid_right_y': right[1],
            'eyelid_bottom_x': bottom[0],
            'eyelid_bottom_y': bottom[1],
        }

        frame_file = self.frame_name.get()

        with open('{0}/{1}.pickle'.format(train_data_path, frame_file), 'rb') as fp:
            current_params = pickle.load(fp)

        new_params = current_params.copy()
        # old naming convention leftover from Leah's MATLAB pupil analysis 
        # which served as first training data for this.
        new_params['ellipse_zack'] = e_params

        with open(train_data_path+'{0}.pickle'.format(frame_file), 'wb') as fp:
            pickle.dump(new_params, fp, protocol=pickle.HIGHEST_PROTOCOL)

        print("saved new ellipse parameters for {0}".format(frame_file))

        self.display_next_frame()


    def update_ellipse_plot(self):
        '''
        update the plot of the ellipse, don't save parameters
        '''

        frame_file = self.frame_name.get()

        with open('{0}/{1}.pickle'.format(train_data_path, frame_file), 'rb') as fp:
            frame_data = pickle.load(fp)

        loc = (0, 0)

        X0_in = np.float(self.y_pos_value.get())  # these are flipped on purpose
        Y0_in = np.float(self.x_pos_value.get())
        long_axis = np.float(self.long_axis_value.get())
        short_axis = np.float(self.short_axis_value.get())
        phi = np.float(self.phi_value.get()) / 180 * np.pi

        if self.edgepoints == []:
            self.edgepoints = get_eyelid_keypoints(frame_data)

        self.plot_calls += 1
        # clear items on the canvas / axes
        try:
            self.ax_handle.clear()
        except:
            pass

        self.ax_handle.imshow(frame_data['frame'])
        ellipse = Ellipse((Y0_in, X0_in), long_axis, - short_axis, 180 * phi / np.pi, fill=False, color='red')
        self.ax_handle.add_patch(ellipse)
        self.ax_handle.plot(np.array(self.edgepoints)[:, 0],
                        np.array(self.edgepoints)[:, 1], lw=0, marker='o', markersize=5, color='red')
        self.figure_canvas_agg.draw()

    def plot_frame(self, frame_file):
        '''
        with open(data_dict, 'rb') as fp:
            data = pickle.load(fp)
        '''
        with open('{0}/{1}.pickle'.format(train_data_path, frame_file), 'rb') as fp:
            frame_data = pickle.load(fp)
            fp.close()

        Y0_in = frame_data['ellipse_zack']['Y0_in']
        X0_in = frame_data['ellipse_zack']['X0_in']
        long_axis = frame_data['ellipse_zack']['b'] * 2
        short_axis = frame_data['ellipse_zack']['a'] * 2
        phi = frame_data['ellipse_zack']['phi']

        # load eyelid keypoints and pack into list of x, y coordinates
        self.edgepoints = get_eyelid_keypoints(frame_data)

        # update the ellipse params display to correspond to the current frame
        self.update_ellipse_params(frame_data['ellipse_zack'])

        loc = (0, 0)

        self.plot_calls += 1
        # clear items on the canvas / axes
        try:
            self.ax_handle.clear()
        except:
            pass
        try:
            self.ax_handle.imshow(frame_data['frame2'])
        except:
            self.ax_handle.imshow(frame_data['frame'])
        #self.ax_handle.axis('off')
        ellipse = Ellipse((Y0_in, X0_in), long_axis, - short_axis, 180 * phi / np.pi, fill=False, color='red')
        self.ax_handle.add_patch(ellipse)
        self.ax_handle.plot(np.array(self.edgepoints)[:, 0],
                                np.array(self.edgepoints)[:, 1], lw=0, marker='o', markersize=5, color='red')
        self.figure_canvas_agg.draw()

    # button press callbacks
    def button_press_callback(self, event):
        # append location to polygon
        try:
            self.ax_handle.lines.pop(0)
        except:
            pass

        if len(self.edgepoints)>=4:
            self.edgepoints = []
            
        self.edgepoints.append((event.xdata, event.ydata))
        ep = np.array(self.edgepoints)
        self.edgeline = self.ax_handle.plot(ep[:, 0], ep[:, 1], lw=0, marker='o', markersize=5, color='red')
        
        self.figure_canvas_agg.draw()

    def key_press_callback(self, event):
        # remove polygon
        if event.key == 'escape':
            # empty the polygon array
            self.edgepoints = []
            try:
                self.ax_handle.lines.pop(0)
            except:
                pass
            self.figure_canvas_agg.draw()


    def sort_edgepoints(self):
        # helper function for sorting edgepoints so that the order is:
        # 1 - left, 2 - top, 3 - right and 4 - bottom
        # that way they're always in the same position for the fitter
        # also, make sure they're not Nan (placeholders)
        if (len(self.edgepoints)!=4) | np.any(np.isnan(np.array(self.edgepoints))):
            raise ValueError("Must label all for edgepoints of the eyelid (left corner, middle-top, right corner, middle-bottom")
        xlocs = np.array(self.edgepoints)[:,0]
        ylocs = np.array(self.edgepoints)[:,1]
        left = np.argmin(xlocs)
        right = np.argmax(xlocs)
        top = np.argmax(ylocs)
        bottom = np.argmin(ylocs)
        edgepoints = np.array(self.edgepoints)
        self.edgepoints = [edgepoints[left], edgepoints[top], edgepoints[right], edgepoints[bottom]]


    def browse_files(self):
        '''
        open a dialog box to choose which frame to load
        '''

        frame_file = self.frame_name.get()

        #self.pupil_canvas.delete(self.pupil_plot)
        self.pupil_plot = self.plot_frame(frame_file)
        self.master.mainloop()


    def display_next_frame(self):
        '''
        display the next frame in the training set
        '''

        current_frame = self.frame_name.get()
        all_frames = os.listdir(train_data_path)
        if self.from_browser:
            all_frames = [f for f in all_frames if self.video_name in f]
            #frame_numbers = [int(f[15:].split('.')[0]) for f in all_frames]
            #inds = np.argsort(np.array(frame_numbers))
            inds = np.argsort(np.array(all_frames))
        else:
            inds = np.argsort(np.array(all_frames))
        all_frames = np.array(all_frames)[inds]

        cur_index = np.argwhere(all_frames == current_frame + '.pickle')[0][0]
        new_index = cur_index + 1
        self.frame_count.delete('1.0', tk.END)
        self.frame_count.insert(tk.END, "{0}/{1}".format(new_index+1, len(all_frames)))
        if new_index == len(all_frames):
            new_index = 0
        new_frame = all_frames[new_index]
        self.frame_name.delete(0, 'end')

        # if the current/new frame aren't from the same video, reset the plot
        if new_frame[:15] != current_frame[:15]:
            self.plot_calls = 0

        new_frame_root = new_frame.split('.')[0]
        self.frame_name.insert(0, new_frame_root)

        #self.pupil_canvas.delete(self.pupil_plot)
        self.pupil_plot = self.plot_frame(new_frame_root)

    def display_prev_frame(self):
        '''
        display the previous frame in the training set
        '''

        current_frame = self.frame_name.get()
        all_frames = os.listdir(train_data_path)
        if self.from_browser:
            all_frames = [f for f in all_frames if self.video_name in f]
            #frame_numbers = [int(f[15:].split('.')[0]) for f in all_frames]
            #inds = np.argsort(np.array(frame_numbers))
            inds = np.argsort(np.array(all_frames))
        else:
            inds = np.argsort(np.array(all_frames))
        all_frames = np.array(all_frames)[inds]

        cur_index = np.argwhere(all_frames == current_frame + '.pickle')[0][0]
        new_index = cur_index - 1
        self.frame_count.delete('1.0', tk.END)
        self.frame_count.insert(tk.END, "{0}/{1}".format(new_index+1, len(all_frames)))
        new_frame = all_frames[new_index]
        self.frame_name.delete(0, 'end')

        if new_frame[:15] != current_frame[:15]:
            self.plot_calls = 0

        new_frame_root = new_frame.split('.')[0]
        self.frame_name.insert(0, new_frame_root)

        ##self.pupil_canvas.delete(self.pupil_plot)
        #self.pupil_canvas.delete("all")
        self.pupil_plot = self.plot_frame(new_frame_root)


def get_eyelid_keypoints(frame):
    try:
        lx = frame['ellipse_zack']['eyelid_left_x']
        ly = frame['ellipse_zack']['eyelid_left_y']
        tx = frame['ellipse_zack']['eyelid_top_x']
        ty = frame['ellipse_zack']['eyelid_top_y']
        rx = frame['ellipse_zack']['eyelid_right_x']
        ry = frame['ellipse_zack']['eyelid_right_y']
        bx = frame['ellipse_zack']['eyelid_bottom_x']
        by = frame['ellipse_zack']['eyelid_bottom_y']
        return [[lx, ly], [tx, ty], [rx, ry], [bx, by]]
    except:
        print("WARNING: Can't find labeled keypoints for eyelid. Make sure to label keypoints by clicking on corners of eyelids")
        return [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]]

root = tk.Tk()

if len(sys.argv) > 1:
    my_gui = TrainingDataBrowser(root, sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])
    root.mainloop()
else:
    my_gui = TrainingDataBrowser(root)
    root.mainloop()

