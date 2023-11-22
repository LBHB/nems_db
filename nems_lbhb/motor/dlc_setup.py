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

experimenter = "svd"

#name = 'free_training'  # implant but no chimneys
name = 'two_chimney'  # dual recording setup
name = 'two_chimney2'  # dual recording setup - take 2
#name = 'two_chimney_LMD'  # dual recording setup
name = 'right_chimney_PRN'  # right recording setup

create_new = False

if create_new:
    # create a new project
    # for original project creation
    path = '/auto/data/dlc/two_chimney-svd-2023-09-28/'
    videos = [path + 'videos/SLJ010a04_a_NTD.avi',
              path + 'videos/SLJ032a10_a_NTD.avi',
              path + 'videos/LMD004a00_a_NFB.avi',
              path + 'videos/LMD005a10_a_NFB.avi']
    # redoing PRN RH
    vpath = '/auto/data/dlc/free_top_RH-jereme-2023-02-17/videos'
    videos = [vpath + '/PRN044a02_a_NTD.avi',
              vpath + '/PRN046a06_a_NTD.avi',
              vpath + '/PRN047a06_a_NTD.avi'
              ]
    path_config_file = deeplabcut.create_new_project(
       name, experimenter,
       videos,
       working_directory='/auto/data/dlc/',
       copy_videos=True,
       multianimal=False,
    )

elif name=='free_training':
    # set up for fitting/refining

    # implant but no chimneys
    path='/auto/data/dlc/free_train-svd-2023-09-26/'
    videos = [path+'videos/SlipperyJack_2023_08_02_NTD_1.avi',
              path+'videos/LemonDisco_2023_09_07_NFB_2.avi']
elif name == 'two_chimney':
    path = '/auto/data/dlc/two_chimney-svd-2023-09-28/'
    videos = [path + 'videos/SLJ010a04_a_NTD.avi',
              path + 'videos/SLJ032a10_a_NTD.avi',
              path + 'videos/LMD004a00_a_NFB.avi',
              path + 'videos/LMD005a10_a_NFB.avi']
elif name == 'two_chimney2':
    path = '/auto/data/dlc/two_chimney2-svd-2023-11-03/'
    videos = [path + 'videos/SLJ010a04_a_NTD.avi',
              path + 'videos/SLJ032a10_a_NTD.avi',
              path + 'videos/LMD004a00_a_NFB.avi',
              path + 'videos/LMD005a10_a_NFB.avi']
elif name == 'right_chimney_PRN':
    path = '/auto/data/dlc/right_chimney_PRN-svd-2023-11-21/'
    videos = [path + 'videos/PRN044a02_a_NTD.avi',
              path + 'videos/PRN046a06_a_NTD.avi',
              path + 'videos/PRN047a06_a_NTD.avi']
else:
    path = '/auto/data/dlc/two_chimney-LMD-2023-11-01/'
    videos = [path+'videos/LMD004a00_a_NFB.avi',
              path+'videos/LMD005a10_a_NFB.avi']
    videos = [path+'videos/LMD004a00_a_NFB.avi']

path_config_file = path + 'config.yaml'

# various functions pulled from
# https://github.com/DeepLabCut/DeepLabCut/blob/main/examples/JUPYTER/Demo_yourowndata.ipynb

deeplabcut.extract_frames(path_config_file)

# need gui:
deeplabcut.label_frames(path_config_file)

deeplabcut.create_training_dataset(path_config_file, augmenter_type='imgaug')
deeplabcut.train_network(path_config_file, shuffle=1, max_snapshots_to_keep=5, 
                         autotune=False, displayiters=100, saveiters=10000, maxiters=60000, allow_growth=True)

deeplabcut.evaluate_network(path_config_file, plotting=True)

deeplabcut.analyze_videos(path_config_file, videos, videotype='.avi')

deeplabcut.create_labeled_video(path_config_file, videos)

deeplabcut.extract_outlier_frames(path_config_file, videos)  #pass a specific video

deeplabcut.refine_labels(path_config_file)
deeplabcut.merge_datasets(path_config_file)
deeplabcut.create_training_dataset(path_config_file, augmenter_type='imgaug')
# edit init_weights in pose_cfg.yaml to point to checkpoint from previous fit
deeplabcut.train_network(path_config_file, shuffle=1, max_snapshots_to_keep=5,
                         autotune=False, displayiters=100, saveiters=5000, maxiters=50000, allow_growth=True)





