"""

  conda create -n dlc5 -c conda-forge python=3.8 pip ipython ffmpeg 
  conda activate dlc5
  conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
  pip install tensorflow==2.10.0  # 2.10.* installed 2.10.1 which then got replaced by 2.10.0 during DLC install
  pip install pyqt6
  pip install -e ./DeepLabCutNew[tf,gui]
  mkdir -p $CONDA_PREFIX/etc/conda/activate.d
  echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
  conda deactivate
  conda activate dlc5


"""


import deeplabcut

path='/auto/data/dlc/free_train-svd-2023-09-26/'

name="free_train"
experimenter="svd"
videos = [path+'videos/SlipperyJack_2023_08_02_NTD_1.avi',
          path+'videos/LemonDisco_2023_09_07_NFB_2.avi']


path_config_file=deeplabcut.create_new_project(
   name, experimenter,
   videos,
   working_directory='/auto/data/dlc/',
   copy_videos=True,
   multianimal=False,
)

path_config_file = path + 'config.yaml'

# various functions pulled from https://github.com/DeepLabCut/DeepLabCut/blob/main/examples/JUPYTER/Demo_yourowndata.ipynb

deeplabcut.extract_frames(path_config_file)

# need gui:
deeplabcut.label_frames(path_config_file)

deeplabcut.create_training_dataset(path_config_file, augmenter_type='imgaug')
deeplabcut.train_network(path_config_file, shuffle=1, max_snapshots_to_keep=5, 
                         autotune=False, displayiters=100, saveiters=5000, maxiters=30000, allow_growth=True)

deeplabcut.evaluate_network(path_config_file, plotting=True)

deeplabcut.analyze_videos(path_config_file, videos, videotype='.avi')

deeplabcut.create_labeled_video(path_config_file, videos)

deeplabcut.extract_outlier_frames(path_config_file, videos) #pass a specific video

deeplabcut.refine_labels(path_config_file)





