## Offline pupil analysis

The code in this directory uses a deep learning approach to estimate the ellipse parameters of the pupil in each frame of a given 
`.mj2` file. The convolutional deep neural network used for this approach was fit on a set of labeled training collected in lbhb.
This training data, along with the current CNN model, is stored in `/auto/data/nems_db/pup_py/`. The most current model is stored
under `default_trained_model`. Old model fits are stored in `old_model_fits` along with their respective training data. As more data
is collected, one might wish to retrain the model to incorporate new training data. This structure makes this possible and the 
procedure for doing this is discussed below.

## Requirements

  * tensorflowg (or tensorflow)
  * ffmpeg

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

    * **Option 1**: Retrain network (see instructions below)

    * **Option 2**: Exclude frames from analysis if there are small portions of the trace you wish to mark for exclusion that you don't think can be fixed, such as blinks
      * hold shift and left click at the start of the segment you wish to exclude. A vertical red bar should appear.
      * hold shift and right click and the end of the segment. You shoud see another red bar appear, and the area between shaded
      grey
      * Once you are pleased with the selection press enter. The selection will disappear and the start/end values will be stored      
      * You may exclude multiple segments
      
  * Finally, once you're pleased with the analysis and have all bad segments marked (if desired), press save analysis. At this 
  point you can load the pupil trace in either `baphy`, using `MATLAB`, (`loadpupiltrace` function) or `nems_db`, using `Python`, (`nems_lbhb.io.loadpupiltrace`) 

## Re-training CNN
If you are not pleased with the fit for your pupil video, the model can probably do better! In order to re-train the model, and reanalyze your pupil video, follow these steps:
  * *Is there a section of time in which the model did particularly bad job?*
    * Use the method outline above of "shift+left click, shift+right click, press enter" to select this section of time
    * Press "Label more training data"
    * You will be asked if these range of selected frames is appropriate, press okay
    * You will be prompted to choose how many frames you'd like to re-label and add to the training data. Default is 50, but you can most likley get away with as few as 10, depending on how big your frame range is. Press okay.

  * *Is the whole video garbage?*
    * In this case, you may not want to bother with selecting specific frames. Simply press "Label more training data", press okay at the first dialog, and enter the desired frame count at the second dialog. 50 frames seems to be a reasonable number.

  * At this point, the training browser GUI should open up, ready to scroll through all video frames in the training data corresponding to this particular pupil video
  * Click through the frames one by one, updating the ellipse to fit the pupil. Make sure to click "save new ellipse params" for each frame!
  * Close the training browser GUI once all frames are labeled correctly
  * Finally, back in the standard pupil browser window, press "Re-train network" to re-train the pupil fitting algorithm with this new data
  * Check the status of the job on celldb. When finished, re-run your pupil video through the updated model. 

