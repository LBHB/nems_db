## Offline pupil analysis

The code in this directory uses a deep learning approach to estimate the ellipse parameters of the pupil in each frame of a given 
`.mj2` file. The convolutional deep neural network used for this approach was fit on a set of labeled training collected in lbhb.
This training data, along with the current CNN model, is stored in `/auto/data/nems_db/pup_py/`. The most current model is stored
under `default_trained_model`. Old model fits are stored in `old_model_fits` along with their respective training data. As more data
is collected, one might wish to retrain the model to incorporate new training data. This structure makes this possible and the 
procedure for doing this is discussed below.

## Analyzing pupil

### Start a pupil analysis
  * activate conda environment (make sure necessary packages are installed - see `nems_db/README.md`)
  * navigate into `nems_db/nems_lbhb/pup_py`
  * from bash prompt: `python fit_pupil.py`, or from Ipython console: `run fit_pupil.py`
  * Specify the Animal name and pupil video filename:
    * ex: Animal: Amanita, Filename: AMT031a06_p_PTD
  * Specify optional options
    * CNN train date: default is set to current, which will run the most up-to-date version of the pupil analysis model
    * Python path: Path to the python that will be used for the fit 
    * Fit script: Script that will be run to perform the anaysis of the video
  * Start fit - this will queue a job on the next available cluster GPU. You can check the status of your job at 
    `http://hyrax.ohsu.edu` 
  
### Browse analysis results and save
  Once your fit is finished, again navigate to `nems_db/nems_lbhb/pup_py` in either a bash prompt or an Ipython console. Make sure
  you have activate the correct conda environment (see above and `nems_db/README.md`)
  
  * In bash prompt: `python pupil_browser.py`, or in Ipython: `run pupil_browser.py`
  * Specify the Animal and Video name, then select load recording
  * The first frame, along with the predicted ellipse (in red) should be displayed in the upper right corner of the gui window
  * Both the major and minor axis of the ellipse over the course of the recording will be displayed in the lower half of the window
  * In order to inspect a particular part of the trace, click on the plot at that location. This will update the displayed frame
  allowing you to evaluate the quality of the model prediction.
  * If unsatistified with some portion of your fit:
    * Exclude frames from analysis if there are small portions of the trace you wish to mark for exclusion, such as blinks
      * hold shift and left click at the start of the segment you wish to exclude. A vertical red bar should appear.
      * hold shift and right click and the end of the segment. You shoud see another red bar appear, and the area between shaded
      grey
      * Once you are pleased with the selection press enter. The selection will disappear and the start/end values will be stored      
    
    * Retrain network (see instructions below)
      
  * Finally, once you're pleased with the analysis and have all bad segments marked (if desired), press save analysis. At this 
  point you can load the pupil trace in either `baphy` (`loadpupiltrace`) or `nems_db` (also `loadpupiltrace`) 

## Re-training CNN
