# Working with the job queue

Queue commands are in the nems_db.db library

## Example:

This is how you can run a single instance of a script:

```
import nems_db.db as nd

# python environment where you want to run the job
executable_path='/auto/users/luke/miniconda3/envs/nemsenv/bin/python'

# name of script that you'd like to run
script_path='/auto/users/svd/python/nems_db/nems_queue_demo.py'

# parameters that will be passed to script as argv[1], argv[2], argv[3]:
parm1='Parameter34'   # for nems_fit_single, this is cellid
parm2='Parameter2'    # for nems_fit_single, this is the batch #
parm3='Parameter332'  # for nems_fit_single, this is the modelname


force_rerun = False   # true if job already has been run and you want to rerun
user = 'joe'            # will be used for load balancing across users

nd.enqueue_single_model(parm1, parm2, parm3, force_rerun=force_rerun,
                        executable_path=executable_path,
                        script_path=script_path, user=user)
```

## OOPS! I made a mistake.

Need to cancel all running jobs? You can kill queue jobs by sending the SQL command:

    update tQueue set killnow = 2 where user = 'bburan' and note like "%||modelname||%"

Kill unstarted jobs as well:

    update tQueue set complete=2 where user='bburan' and note like "%||modelname||%" and complete=0
