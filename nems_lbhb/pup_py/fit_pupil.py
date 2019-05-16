# this script will attempt to use the default trained CNN to predict pupil for all times in a video. Results from the
# fit will be saved in a pickled python dictionary to be used by pupil_browser (to check the fit) and by nems (for
# data analysis). The job will be added to the queue and will run on the next available GPU machine.
try:
    import tkinter as tk
except:
    import Tkinter as tk
import nems.db as nd
import getpass

import sys
import os
import nems_db
executable_path = sys.executable
script_path = os.path.split(os.path.split(nems_db.__file__)[0])[0]
script_path = os.path.join(script_path, 'nems_lbhb', 'pup_py', 'pupil_fit_script.py')

class queue_pupil_job:

    def __init__(self, master):
        self.master = master
        master.title("Queue pupil job")
        master.geometry('300x100')

        self.animal = tk.Label(master, text="Animal: ")
        self.animal.grid(row=0, column=0)
        self.animal_value = tk.Entry(master)
        self.animal_value.grid(row=0, column=1, sticky='ew')
        self.animal_value.focus_set()

        self.filename = tk.Label(master, text="Filename: ")
        self.filename.grid(row=1, column=0)
        self.filename_value = tk.Entry(master)
        self.filename_value.grid(row=1, column=1, sticky='ew')
        self.filename_value.focus_set()

        self.executable_path = tk.Label(master, text="Python path: ")
        self.executable_path.grid(row=2, column=0)
        self.executable_path_value = tk.Entry(master)
        self.executable_path_value.grid(row=2, column=1, sticky='ew')
        self.executable_path_value.focus_set()
        self.executable_path_value.insert(0, executable_path)

        self.script_path = tk.Label(master, text="Fit script: ")
        self.script_path.grid(row=3, column=0)
        self.script_path_value = tk.Entry(master)
        self.script_path_value.grid(row=3, column=1, sticky='ew')
        self.script_path_value.focus_set()
        self.script_path_value.insert(0, script_path)

        self.fit_button = tk.Button(master, text="Start fit", command=self.start_fit)
        self.fit_button.grid(row=4, column=0)

        master.grid_columnconfigure(1, weight=1)


    def start_fit(self):
        animal = self.animal_value.get()
        filename = self.filename_value.get()
        py_path = self.executable_path_value.get()
        script_path = self.script_path_value.get()
        #py_path = '/auto/users/hellerc/anaconda3/envs/pupil_processing/bin/python3.6'
        #script_path = '/auto/users/hellerc/code/nems/nems_db/nems_lbhb/pup_py/pupil_fit_script.py'

        username = getpass.getuser()

        # add job to queue
        nd.add_job_to_queue([animal, filename], note="Pupil Job: {}".format(filename), 			executable_path=py_path,
                            user=username, force_rerun=True, script_path=script_path, GPU_job=1)
        print("added job to queue")

root = tk.Tk()
my_gui = queue_pupil_job(root)
root.mainloop()
