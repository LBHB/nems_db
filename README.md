# NEMS DB API

This repository contains code for saving modelspecs, logfiles, images, and recordings to a database accessible via HTTP.

## Installation

SSH into the server that will be running this API. Then install the following required packages:

```
pip3 install flask --user
pip3 install flask_restful --user
```

Get a copy of this repository and `cd` into it.

Copy the configuration in "default_config.sh" to a new file and edit the new copy. Then run:

```
source configs/my_config.sh   # Sets environment variables
python3 -m nems_db
```

Note that `NEMS_DB_API_HOST` should be localhost for testing, but be sure to make it an externally-facing address if you want anybody else to be able to use it.


## Testing the API

In a new terminal on that same machine, you should be able to test that the API is working using:

```
curl TODO
```

## Setting up your own personal Python environment

On one of the workstations connected to the NFS server, CD to your personal
home folder on /auto (e.g., `/auto/users/bburan`). Create a `bin` folder.
Download the miniconda installer. When prompted, point to ththe folder you just
created (e.g., you want it installed in `/auto/users/bburan/bin/miniconda3`). 

Now, open your shell configuration file and create some aliases to the key
commands you will need. I use zsh, so you will need to figure out how to set up
similar aliases for your preferred shell:

```
alias cconda="/auto/users/bburan/bin/miniconda3/bin/conda"
alias cactivate="source /auto/users/bburan/bin/miniconda3/bin/activate"
```

These aliases will allow you to keep your own local copy of Miniconda on the
computer while being able to quickly access and configure the copy of Miniconda
that you inted to use for cluster jobs.

Now, open a new shell (or source the configuration file). The aliases should
now be active. To install your new environment:

	cconda create -n nems-intel -c intel python=3

To activate the environment:

	cactivate nems-intel

Note that once you've activated the environment, your path has been updated so that you no longer need to use the cconda and ccactivate aliases. Once the environment is deactivated, you will have to use the cconda and ccactivate aliases again.

## Offline pupil analysis
In order to use the offline pupil analysis tools located in `nems_lbhb/pup_py/` you must make sure you have the following dependencies installed in your environment. In order to install these dependencies (and any additional dependencies you might need), we recommend using the conda package manager.

For example, this protocol for creating an environment seems stable (for LBHB workstation):

```
conda create --name my_pupil_env python=3.7 tensorflow-gpu keras-gpu
conda activate my_pupil_env
conda install cudatoolkit==9.0
pip install -e <path/to/nems_db>
pip install -e <path/to/NEMS>
conda install -c conda-forge av scikit-image
```
