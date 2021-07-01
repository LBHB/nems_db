# this script will attempt to use the default trained CNN to predict pupil for all times in a video. Results from the
# fit will be saved in a pickled python dictionary to be used by pupil_browser (to check the fit) and by nems (for
# data analysis). The job will be added to the queue and will run on the next available GPU machine.
try:
    import tkinter as tk
except:
    import Tkinter as tk
import nems.db as nd
import getpass
from tkinter import filedialog, messagebox

import nems_db
import sys
import os
executable_path = sys.executable
script_path = os.path.split(os.path.split(nems_db.__file__)[0])[0]
script_path = os.path.join(script_path, 'nems_lbhb', 'pup_py', 'pupil_fit_script.py')
nems_db_path = nems_db.__path__[0]
sys.path.append(os.path.join(nems_db_path, 'nems_lbhb/pup_py/'))
import pupil_settings as ps

class queue_pupil_job:

    def __init__(self, master):
        self.master = master
        master.title("Queue pupil job")
        master.geometry('700x175')

        self.filename = tk.Label(master, text="Filename: ")
        self.filename.grid(row=1, column=0)
        self.filename_value = tk.Entry(master)
        self.filename_value.grid(row=1, column=1, sticky='ew')
        self.filename_value.focus_set()
        
        self.file_browse = tk.Button(master, text="Select pupil file", command=self.get_fn)
        self.file_browse.grid(row=0, column=0)


        self.fitDate = tk.Label(master, text="CNN train date: ")
        self.fitDate.grid(row=2, column=0)
        self.fitDate_value = tk.Entry(master)
        self.fitDate_value.grid(row=2, column=1, sticky='ew')
        self.fitDate_value.focus_set()
        self.fitDate_value.insert(0, "Current")

        self.species_label = tk.Label(master, text="Species: ")
        self.species_label.grid(row=3, column=0)
        self.species = tk.StringVar(master)
        self.species.set('ferret') # default value
        options = os.listdir(ps.ROOT_DIRECTORY)
        self.species_dd = tk.OptionMenu(master, self.species, *options)
        self.species_dd.grid(row=3, column=1, sticky='ew')
        self.species_dd.focus_set()


        self.animal_fit = tk.Label(master, text="Animal: ")
        self.animal_fit.grid(row=4, column=0)
        self.animal_fit_value = tk.Entry(master)
        self.animal_fit_value.grid(row=4, column=1, sticky='ew')
        self.animal_fit_value.focus_set()
        self.animal_fit_value.insert(0, "All")

        self.executable_path = tk.Label(master, text="Python path: ")
        self.executable_path.grid(row=5, column=0)
        self.executable_path_value = tk.Entry(master)
        self.executable_path_value.grid(row=5, column=1, sticky='ew')
        self.executable_path_value.focus_set()
        self.executable_path_value.insert(0, executable_path)

        self.script_path = tk.Label(master, text="Fit script: ")
        self.script_path.grid(row=6, column=0)
        self.script_path_value = tk.Entry(master)
        self.script_path_value.grid(row=6, column=1, sticky='ew')
        self.script_path_value.focus_set()
        self.script_path_value.insert(0, script_path)

        self.fit_button = tk.Button(master, text="Start fit", command=self.start_fit)
        self.fit_button.grid(row=7, column=0)

        master.grid_columnconfigure(1, weight=1)


    def start_fit(self):
        # get video name from dialog box (in case didn't import with button)
        self.video_file = self.filename_value.get()
        fn = self.video_file
        modeldate = self.fitDate_value.get()
        animal = self.animal_fit_value.get()
        species = self.species.get()
        answer = True
        if (animal != '') & (animal != 'All') & (animal != 'all'):
            answer = messagebox.askokcancel("Question", "You've asked to fit this video using the model architecture that was overfit"
                            " on {0}. Are you sure you want to proceed?".format(animal))

        py_path = self.executable_path_value.get()
        script_path = self.script_path_value.get()
        #py_path = '/auto/users/hellerc/anaconda3/envs/pupil_processing/bin/python3.6'
        #script_path = '/auto/users/hellerc/code/nems/nems_db/nems_lbhb/pup_py/pupil_fit_script.py'

        username = getpass.getuser()

        # add job to queue
        if answer:
            nd.add_job_to_queue([fn, modeldate, f"{species}_{animal}"], note="Pupil Job: {}".format(fn),
                                executable_path=py_path, user=username,
                                force_rerun=True, script_path=script_path, GPU_job=1)
            print("added job to queue")
        else:
            return None

    def get_fn(self):
        self.video_file = filedialog.askopenfilename(initialdir = "/auto/data/daq/",
                            title = "Select raw video file", 
                            filetypes = (("mj2 files","*.mj2*"), ("avi files","*.avi")))
        self.filename_value.delete(0, 'end')
        self.filename_value.insert(0, self.video_file)

root = tk.Tk()
my_gui = queue_pupil_job(root)
root.mainloop()
