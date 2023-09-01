""" test code for demo_script.py

should return an html string that, if copy and pasted into an html doc with
bokeh's dependencies linked, will display a bokeh plot.

the plot should also open on its own in a web browser when the function is run.

"""

from nems0.utilities.output import web_print

import nems_scripts.demo_script as demo
# This information would normally be passed by the web interface,
# but it can also be specified manually.
argsdict = {
        'batch' : 291,
        'models' : ['fb18ch100_wc01_fir15_dexp_fititer00',
                    'fb18ch100_wc01_stp1pc_fir15_dexp_fititer00'
                    ],
        'cells' : ['bbl031f-a1', 'bbl034e-a1', 'bbl034e-a2', 'bbl036e-a1',
                   'bbl036e-a2', 'bbl038f-a1', 'bbl038f-a2', 'bbl041e-a1',
                   'bbl041e-a2',
                   ],
        # currently not working with onlyFair turned on.
        'onlyFair' : False,
        'includeOutliers' : False,
        'snr' : 0.0,
        'iso' : 70.0,
        'snri' : 0.1,
        'measure' : 'r_test',
        }

output = demo.run_script(argsdict)
log.info('html string: ')
log.info(output['html'])
log.info('final dataframe used for plot: ')
log.info(output['data'])
# the cellid column is just a duplicate of the index, but necessary
# because bokeh's hovertool will only display the index as an integer

# TODO: make a public s3 bucket that stores script outputs for download by user?

# Note that print statements will NOT show up in your local output unless you
# are running a local copy of the web interface. If interacting with
# neuralprediction.org, you would instead want to either return some
# html-friendly string to display on the webpage or save the output to a file
# that you can retrieve. 