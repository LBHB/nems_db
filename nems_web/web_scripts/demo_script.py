"Demo for using custom script interface to generate a scatter plot to compare
modelwise performance.

NOTE: This module and doc need to be updated - in the interim,
      custom scripts are better run on a local machine using
      the main nems distribution rather than through the web UI.
      --jacob 3/11/18

"""

# module might be named: jacob_scatter.py

import itertools

from bokeh.io import gridplot
from bokeh.plotting import figure, show
from bokeh.embed import components
from bokeh.models import (
        ColumnDataSource, HoverTool, ResizeTool ,SaveTool, WheelZoomTool,
        PanTool, ResetTool,
        )
from bokeh.models.glyphs import Circle
import pandas as pd

# import sessionmaker and connection engine from db.py
# both can be used for interacting with the mysql database, but an open session
# is a required argument for the script_utils functions
from nems0.db import Session, engine
# import helper functions for custom scripts
from nems0.web.run_custom.script_utils import filter_cells, form_data_array


# visual attributes for the circle glyphs in the scatter plot
CIRCLE_SIZE = 5
CIRCLE_ALPHA = 0.5
CIRCLE_FILL = 'navy'

def run_script(argsdict):
    """The function that will be run by nems.web.run_custom.views.py if called
    through the web interface. To ensure that your script is useable both
    in isolation and through the web interface, this function must be included
    and should take a dictionary containing batch, cell and model information
    as an argument.
    
    Arguments:
    ----------
    Argsdict : Dict object
        Defines the batch, cellids, and modelnames to analyze as well as cell
        filtering options (onlyFair, includeOutliers, iso, snr and snri). The
        filtering options should always be included if the script makes use of
        script_utils functions.
        
    Returns:
    --------
    script_output: Dict object (optional)
        If a dict is returned with the key 'html' assigned, the web interface
        will display that output (ex: the script and div components for a bokeh
        plot could be concatenated under the 'html' key).
        
    """
    
    # start a database connection
    session = Session()
    
    # re-assign values from argsdict for more readable variable names
    
    # batch numer
    batch = argsdict['batch']
    # list of cellids
    cells = argsdict['cells']
    # minimum signal to noise ratio desired
    snr = argsdict['snr']
    # minimum isolation value
    iso = argsdict['iso']
    # minimum SNR index
    snri = argsdict['snri']
    
    # get a list of the cells that don't meet the snr/iso/snri criteria
    bad_cells = filter_cells(session, batch, cells, snr, iso, snri)
    # remove the bad cells from the cellid list
    cells = [c for c in cells if c not in bad_cells]
    
    # list of model names
    models = argsdict['models']
    # whether or not to exclude cells that were only fit for some of the models
    # True by default
    only_fair = argsdict['onlyFair']
    # whether or not to include cells that fall outside of the 'normal' range
    # False by default
    include_outliers = argsdict['includeOutliers']
    
    data = form_data_array(
            session, batch, cells, models, columns=None, only_fair=only_fair, 
            include_outliers=include_outliers,
            )
        
    
    ### Code above sets up the data -- code below makes the plot with bokeh
    ### Pandas also has its own plotting methods for dataframes, and they
    ### work well with matplotlib also.
    
    # The performance metric to compare by
    measure = argsdict['measure']
    plots = []
    
    # Iterate over a list of tuples representing all unique pairs of models.
    for pair in list(itertools.combinations(models,2)):
        # create a hover tool that shows the x value, y value and cellid for
        # a given point on the plot
        hover_html = """
            <div>
                <span class="hover-tooltip">{0} x: @x_values</span>
            </div>
            <div>
                <span class="hover-tooltip">{0} y: @y_values</span>
            </div>
            <div>
                <span class="hover-tooltip">cell: @cellid</span>
            </div>
            """.format(measure)
        hover = HoverTool(tooltips=hover_html)
        tools = [
                PanTool(), ResizeTool(), SaveTool(), WheelZoomTool(),
                ResetTool(), hover,
                ]
        
        modelX = pair[0]
        modelY = pair[1]
        
        # pull out two dataframes representing the data for for the two
        # individual models
        dataX = data.loc[modelX]
        dataY = data.loc[modelY]
            
        # Only necessary b/c bokeh's $index identifier for HoverTool()
        # is pulling an integer index despite data being indexed by cellid.
        # If cellid strings can be pulled from index instead,
        # this code will no longer be needed.
        cells = []
        cellsX = list(set(dataX.index.values.tolist()))
        cellsY = list(set(dataY.index.values.tolist()))
        if not only_fair:
            # If fair wasn't checked, use the longer list to avoid errors.
            if len(cellsX) >= len(cellsY):
                cells = cellsX
            else:
                cells = cellsY
        
        plot_data = pd.DataFrame({
                'x_values':dataX[measure],
                'y_values':dataY[measure],
                'cellid':cells,
                })
        # create a bokeh data source from the dataframe
        dat_source = ColumnDataSource(plot_data)
    
        p = figure(
                x_range=[0,1], y_range=[0,1],
                x_axis_label=modelX, y_axis_label=modelY,
                title=measure, tools=tools, responsive=True,
                )
        glyph = Circle(
                x='x_values', y='y_values', size=CIRCLE_SIZE,
                fill_color=CIRCLE_FILL, fill_alpha=CIRCLE_ALPHA,
                )
        p.add_glyph(dat_source, glyph)
        p.line([0,1], [0,1], line_width=1, color='black')
        plots.append(p)
    
        # If more than one plot was made (i.e. 2 or more models were selected),
        # put them in a grid.

    script = ''
    div = ''
    if len(plots) == 1:
        singleplot = plots[0]
        script, div = components(singleplot)
        show(singleplot)      
    elif len(plots) > 1:            
        grid = gridplot(
                plots, responsive=True,
                )
        script, div = components(grid)
        show(grid)
    else:
        script, div = (
                'Error, no plots to display.',
                'Make sure you selected two models.'
                )
        
    script_output = {'html' : (script + div),
                     'data' : plot_data,
                     }
    return script_output
    
    
    
    
    
