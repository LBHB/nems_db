import logging
import io
import math
import itertools

from bokeh.layouts import gridplot
from bokeh.plotting import figure, show
from bokeh.embed import components
from bokeh.models import (
        ColumnDataSource, HoverTool, SaveTool, WheelZoomTool,
        PanTool, ResetTool, Range1d, FactorRange,
        )
from bokeh.models.glyphs import VBar, Circle
import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch

import nems_web.utilities.pruffix as prx

log = logging.getLogger(__name__)
#NOTE: All subclasses of PlotGenerator should be added to the PLOT_TYPES
#      list for use with web interface
PLOT_TYPES = [
        'Scatter', 'Bar', 'Pareto', 'Tabular', 'Significance'
        ]

# Setting default tools as global variable was causing issues with scatter
# plot. They're included here for copy-paste as needed instead.
#tools = [
#    PanTool(), SaveTool(), WheelZoomTool(),
#    ResetTool(), self.create_hover()
#    ]


# Specify the number of columns to use for gridplots
GRID_COLS = 1
# Appearance options for circle glyphs (ex. scatter plot)
CIRCLE_FILL = 'navy'
CIRCLE_SIZE = 5
CIRCLE_ALPHA = 0.5
# Appearance options for virtical bar glyphs (ex. bar plot)
VBAR_FILL = '#FF5740'
VBAR_WIDTH = 0.5
# Position of toolbar around plot area
# (above, below, left, right)
TOOL_LOC = 'above'
# Should the toolbar be inside the axis?
# TODO: check back on bokeh issue. currently cannot combine tool_loc above
#       with tool_stick false due to a conflict with responsive mode.
#       -jacob 7/12/17
TOOL_STICK = True
OUTPUT_FORMAT = 'svg'

class PlotGenerator():
    """Base class for plot generators."""

    def __init__(
            self, data, measure, models, fair=True, outliers=False,
            extra_cols=[], display=False,
            ):
        # Force measure to be interpreted as a list for
        # forward-compatibility with specifying multiple measures.
        if isinstance(measure, list):
            self.measure = measure + extra_cols
        else:
            self.measure = [measure] + extra_cols
        self.fair = fair
        self.outliers = outliers
        self.display = display
        log.debug("Checking for common prefixes in modelnames...")
        self.abbr, self.pre, self.suf = prx.find_common(models)
        log.debug("Got back:\nabbr:{}\npre:{}\nsuff:{}\n"
                  .format(self.abbr, self.pre, self.suf))
        data.replace(models, self.abbr, inplace=True)
        log.debug("Replaced modelnames with abbreviations.")
        self.models = self.abbr
        log.info("Forming data array inside PlotGenerator...")
        self.data = self.form_data_array(data)

        # Use this inside views function to check whether generate_plot
        # should be invoked.
        if (self.data.size == 0):
            log.info("All data filterd out when forming data array,"
                     "setting self.emptycheck to True")
            self.emptycheck = True
        else:
            self.emptycheck = False
        log.info("Building plot...")


    def generate_plot(self):
        """Assigns script and div attributes to the plot generator object.

        All plot classes should implement some form of this function to ensure
        that script and div are returned to the relevant view function.

        Assigns:
        --------
        self.script : JSON data (or a string default/error)
            Bokeh javascript needed by whatever HTML document embeds self.div
        self.div : JSON data (or a string default/error)
            Bokeh javascript used to render a plot from the attached
            datasource.

        """

        self.script, self.div = ('Plot type is','not yet implemented')

    def create_hover(self):
        """Returns a Bokeh HoverTool() object, possibly with generated
        toolip HTML.

        $ and @ tell the HoverTool() where to pull data values from.
        $ denotes a special column (ex: $index or $x),
            refer to Bokeh docs for a full list.
        @ denotes a field in the data source (ex: @cellid)

        """

        return HoverTool()

    def form_data_array(self, data):
        """Formats data into a multi-indexed DataFrame for plotting.

        Takes a DataFrame (built from a full Results query) and converts it
        to a new multi-indexed DataFrame (cellid level 0, modelname level 1)
        with a column for each performance measure, plus any other columns
        needed for specific plots.
        If 'outliers' isn't checked, cellid rows containing values that don't
            meet outlier criteria will be removed.
        If 'fair' is checked, cellid rows containing a NaN value in any
            column will be removed.

        Returns:
        --------
        newData : Pandas DataFrame w/ multi-index
            Multi-indexed DataFrame containing a series of values for each
            cellid + modelname combination.

        See Also:
        ---------
        _Analysis : compute_data_matrix

        """

        # TODO: figure out a good way to form this from the existing
        #       dataframe instead of making a new one then copying over.
        #       Should be able to just re-index then apply some
        #       lambda function over vectorized dataframe for filtering?

        celllist = [
                cell for cell in
                list(set(data['cellid'].values.tolist()))
                ]
        modellist = self.models
        # Use lists of unique cell and model names to form a multiindex.
        multiIndex = pd.MultiIndex.from_product(
                [celllist,modellist], names=['cellid','modelname'],
                )
        newData = pd.DataFrame(
                index = multiIndex, columns = self.measure,
                )

        newData.sort_index()

        for c in celllist:
            for m in modellist:
                dataRow = data.loc[(data.cellid == c) & (data.modelname == m)]

                # Add column values for any additional columns specificed
                # in plot class (ex: n_parms for pareto plot)
                #if self.extra_cols:
                #    for col in self.extra_cols:
                #        try:
                #            colval = dataRow[col].values.tolist()[0]
                #        except Exception as e:
                #            # TODO: Is this a good way to do this?
                #            #       This requires that all extra columns also
                #            #       have values for every cell/model
                #            #       if fair is checked.
                #            colval = math.nan
                #            log.info(e)
                #        finally:
                #            newData[col].loc[c,m] = colval

                for meas in self.measure:
                    value = np.nan
                    newData[meas].loc[c,m] = value
                    # If loop hits a continue, value will be left as NaN.
                    # Otherwise, will be assigned a value from data
                    # after passing all checks.
                    try:
                        value = dataRow[meas].values.tolist()[0]
                    except Exception as e:
                        # Error should mean no value was recorded,
                        # so leave as NaN.
                        # No need to run outlier checks if value is missing.
                        log.info("No %s recorded for %s,%s"%(meas,c,m))
                        continue

                    if not self.outliers:
                        # If outliers is false, run a bunch of checks based on
                        # measure and if a check fails, step out of the loop.

                        # Comments for each check are copied from
                        # from _Analysis : compute_data_matrix

                        # "Drop r_test values below threshold"
                        a1 = (meas == 'r_test')
                        b1 = (value < dataRow['r_floor'].values.tolist()[0])
                        a2 = (meas == 'r_ceiling')
                        b2 = (
                            dataRow['r_test'].values.tolist()[0]
                            < dataRow['r_floor'].values.tolist()[0]
                            )
                        a3 = (meas == 'r_floor')
                        b3 = b1
                        if (a1 and b1) or (a2 and b2) or (a3 and b3):
                            continue

                        # "Drop MI values greater than 1"
                        a1 = (meas == 'mi_test')
                        b1 = (value > 1)
                        a2 = (meas == 'mi_fit')
                        b2 = (0 <= value <= 1)
                        if (a1 and b1) or (a2 and not b2):
                            continue

                        # "Drop MSE values greater than 1.1"
                        a1 = (meas == 'mse_test')
                        b1 = (value > 1.1)
                        a2 = (meas == 'mse_fit')
                        b2 = b1
                        if (a1 and b1) or (a2 and b2):
                            continue

                        # "Drop NLOGL outside normalized region"
                        a1 = (meas == 'nlogl_test')
                        b1 = (-1 <= value <= 0)
                        a2 = (meas == 'nlogl_fit')
                        b2 = b1
                        if (a1 and b1) or (a2 and b2):
                            continue

                        # TODO: is this still used? not listed in NarfResults
                        # "Drop gamma values that are too low"
                        a1 = (meas == 'gamma_test')
                        b1 = (value < 0.15)
                        a2 = (meas == 'gamma_fit')
                        b2 = b1
                        if (a1 and b1) or (a2 and b2):
                            continue

                        # TODO: is an outlier check needed for cohere_test
                        #       and/or cohere_fit?

                    # If value existed and passed outlier checks,
                    # re-assign it to the proper DataFrame position
                    # to overwrite the NaN value.
                    newData[meas].loc[c,m] = value

        if self.fair:
            # If fair is checked, drop all rows that contain a NaN value for
            # any column.
            for c in celllist:
                for m in modellist:
                    if newData.loc[c,m].isnull().values.any():
                        newData.drop(c, level='cellid', inplace=True)
                        log.debug("Dropping cellid {}, not fit by model {}"
                                  .format(c, m))
                        break


        # Swap the 0th and 1st levels so that modelname is the primary index,
        # since most plots group by model.
        newData = newData.swaplevel(i=0, j=1, axis=0)

        # Leave these in for testing to make sure dropping NaN values
        # is working correctly
        #log.info("was fair checked?")
        #log.info(self.fair)
        #log.info("does the data look different or contain nans?")
        #log.info(newData[self.measure[0]].values)

        return newData


class Scatter(PlotGenerator):
    """Defines the class used to generate a model-comparison scatter plot."""

    def create_hover(self):
        hover_html = """
            <div>
                <span class="hover-tooltip">%s x: @x_values</span>
            </div>
            <div>
                <span class="hover-tooltip">%s y: @y_values</span>
            </div>
            <div>
                <span class="hover-tooltip">cell: @cellid</span>
            </div>
            """%(self.measure[0],self.measure[0])

        return HoverTool(tooltips=hover_html)

    def generate_plot(self):
        """Iteratively reformats and plots self.data for each cell+model combo.

        TODO: Finish this doc

        """

        plots = []
        modelnames = self.data.index.levels[0].tolist()

        # Iterate over a list of tuples representing all unique pairs of models.
        for pair in list(itertools.combinations(modelnames,2)):
            tools = [
                PanTool(), SaveTool(), WheelZoomTool(),
                ResetTool(), self.create_hover(),
                ]

            modelX = pair[0]
            modelY = pair[1]

            dataX = self.data.loc[modelX]
            dataY = self.data.loc[modelY]

            # Only necessary b/c bokeh's $index identifier for HoverTool()
            # is pulling an integer index despite data being indexed by cellid.
            # If cellid strings can be pulled from index instead,
            # this code will no longer be needed.
            cells = []
            cellsX = list(set(dataX.index.values.tolist()))
            cellsY = list(set(dataY.index.values.tolist()))
            if self.fair:
                # cellsX and cellsY should be the same if fair was checked
                cells = cellsX
                if cells != cellsY:
                    self.script = 'Problem with form_data_array:'
                    self.div = 'Model x: ' + modelX + 'and Model y: ' + modelY\
                            + ' applied to different cells despite fair check.'
                    return
            else:
                # If fair wasn't checked, use the longer list to avoid errors.
                if len(cellsX) >= len(cellsY):
                    cells = cellsX
                else:
                    cells = cellsY

            x_mean = np.mean(dataX[self.measure[0]])
            x_median = np.median(dataX[self.measure[0]])
            y_mean = np.mean(dataY[self.measure[0]])
            y_median = np.median(dataY[self.measure[0]])

            x_label = (
                    "{0}, mean: {1:5.4f}, median: {2:5.4f}"
                    .format(modelX, x_mean, x_median)
                    )
            y_label = (
                    "{0}, mean: {1:5.4f}, median: {2:5.4f}"
                    .format(modelY, y_mean, y_median)
                    )

            data = {
                    'x_values':dataX[self.measure[0]],
                    'y_values':dataY[self.measure[0]],
                    'cellid':cells,
                    }
            dat_source = ColumnDataSource(data)

            p = figure(
                    x_range=[0,1], y_range=[0,1],
                    x_axis_label=x_label, y_axis_label=y_label,
                    title=("{0}, prefix: {1}, suffix: {2}"
                           .format(self.measure[0], self.pre, self.suf)),
                    tools=tools,
                    toolbar_location=TOOL_LOC, toolbar_sticky=TOOL_STICK,
                    output_backend="svg", sizing_mode='scale_width',
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

        #if len(plots) == 1:
        #    singleplot = plots[0]
        #    self.script,self.div = components(singleplot)
        #    return
        #elif len(plots) > 1:
        grid = gridplot(
                plots, ncols=GRID_COLS,
                sizing_mode='scale_width',
                )
        self.script,self.div = components(grid)
        if not plots:
            self.script, self.div = (
                    'Error, no plots to display.',
                    'Make sure you selected two models.'
                    )
        else:
            self.plot = grid
            if self.display:
                show(grid)


class Bar(PlotGenerator):
    """Defines the class used to generate a mean-performance bar plot for
    a model-by-model comparison.

    """

    def create_hover(self):
        hover_html = """
            <div>
                <span class="hover-tooltip">model: $x</span>
            </div>
            <div>
                <span class="hover-tooltip">mean: @mean</span>
            </div>
            <div>
                <span class="hover-tooltip">median: @median</span>
            </div>
            <div>
                <span class="hover-tooltip">n cells: @n_cells</span>
            </div>
            <div>
                <span class="hover-tooltip">stdev: @stdev</span>
            </div>
            """

        return HoverTool(tooltips=hover_html)

    def generate_plot(self):
        """Calculates mean and standard deviation for measure(s) by model,
        then generates a bar plot of model vs mean performance.

        TODO: Finish this doc.

        """

        # Use this for a built-in bar plot instead,
        # but doesn't work with custom hover tool
        #p = Bar(self.data,label='modelname',values=self.measure,agg='mean',\
        #        title='Mean %s Performance By Model'%self.measure,legend=None,\
        #        tools=tools, color='modelname')
        #self.script,self.div = components(p)
        #return

        # TODO: hardcoded self.measure[0] for now, but should incorporate
        #       a for loop somewhere to subplot for each selected measure

        # TODO: add significance information (see plot_bar_pretty
        #       and randttest in narf_analysis)

        # build new pandas series of stdev values to be added to dataframe
        # if want to show more info on tooltip in the future, just need
        # to build an appropriate series to add and then build its tooltip
        #in the create_hover function

        modelnames = self.data.index.levels[0].tolist()

        # Bokeh gets unhappy if modelnames are too long for this one.
        # Not an ideal solution but avoids errors for now.
        length_flag = False
        for m in modelnames:
            if len(m) >= 70:
                length_flag = True
        if length_flag:
            shortened_modelnames = [m[:67]+'...' for m in modelnames]
            self.data = self.data.rename(index=dict(zip(modelnames,
                                                        shortened_modelnames)))
            modelnames = shortened_modelnames

        stdev_col = pd.Series(index=modelnames)
        mean_col = pd.Series(index=modelnames)
        median_col = pd.Series(index=modelnames)
        n_cells_col = pd.Series(index=modelnames)
        #for each model, find the stdev and mean over the measure values, then
        #assign those values to new Series objects to use for the plot
        for model in modelnames:
            values = self.data[self.measure[0]].loc[model]
            stdev = values.std(skipna=True)
            mean = values.mean(skipna=True)
            median = values.median(skipna=True)
            if (math.isnan(stdev)) or (math.isnan(mean)) or (math.isnan(median)):
                # If either statistic comes out as NaN, entire column was NaN,
                # so model doesn't have the necessary data.
                continue
            stdev_col.at[model] = stdev
            mean_col.at[model] = mean
            median_col.at[model] = median
            n_cells_col.at[model] = values.count()

        newData = pd.DataFrame.from_dict({
                'stdev':stdev_col, 'mean':mean_col, 'median':median_col,
                'n_cells':n_cells_col,
                })
        # Drop any models with NaN values, since that means they had no
        # performance data for one or more columns.
        newData.dropna(axis=0, how='any', inplace=True)
        if newData.size == 0:
            self.script,self.div = (
                    "Error, no plot to display.",
                    "None of the models contained valid performance data."
                    )
            return
        dat_source = ColumnDataSource(newData)

        tools = [
                PanTool(), SaveTool(), WheelZoomTool(),
                ResetTool(), self.create_hover()
                ]

        xrange = FactorRange(factors=modelnames)
        yrange = Range1d(
                start=0,
                end=(max(newData['mean'])*1.5)
                )
        p = figure(
                x_range=xrange, x_axis_label=(
                        "Modelname, prefix: {0}, suffix: {1}"
                        .format(self.pre, self.suf)
                        ),
                y_range=yrange, y_axis_label='Mean %s'%self.measure[0],
                title="Mean %s Performance By Model"%self.measure[0],
                tools=tools, toolbar_location=TOOL_LOC,
                toolbar_sticky=TOOL_STICK, output_backend=OUTPUT_FORMAT,
                sizing_mode='stretch_both',
                )
        p.xaxis.major_label_orientation=-(np.pi/4)
        glyph = VBar(
                x='index', top='mean', bottom=0, width=VBAR_WIDTH,
                fill_color=VBAR_FILL, line_color='black',
                )
        p.add_glyph(dat_source,glyph)

        # workaround to prevent title and toolbar from overlapping
        grid = gridplot(
            [p], ncols=GRID_COLS, sizing_mode='stretch_both',
            )
        self.script, self.div = components(grid)
        self.plot = grid
        if self.display:
            show(grid)


class Pareto(PlotGenerator):
    """Defines the class used to generate a Bokeh box-plot for mean performance
    versus model complexity.

    """

    # Always include 'n_parms' as an extra column, since it's required
    # for this plot type.
    def __init__(
            self, data, measure, models, fair=True, outliers=False,
            extra_cols=['n_parms'], display=False,
            ):
        PlotGenerator.__init__(
                self, data, measure, models, fair, outliers, extra_cols,
                display,
                )

    def create_hover(self):
        hover_html = """
        <div>
            <span class="hover-tooltip">Model: @modelname</span>
        </div>
        <div>
            <span class="hover-tooltip">N Parms: @n_parms</span>
        </div>
        <div>
            <span class="hover-tooltip">mean {0}: @mean</span>
        </div>
        <div>
            <span class="hover-tooltip">std err: @stderr</span>
        </div>
        """.format(self.measure[0])

        return HoverTool(tooltips=hover_html)

    def generate_plot(self):
        """TODO: write this doc."""

        tools = [
                PanTool(), SaveTool(), WheelZoomTool(),
                ResetTool(),
                ]

        x_values = []
        y_values = []
        std_errors = []
        models = self.data.index.levels[0].tolist()
        for model in models:
            values = self.data[self.measure[0]].loc[model].values
            mean = np.mean(values)
            stderr = np.around(st.sem(values, nan_policy='omit'), 5)
            n_parms = self.data['n_parms'].loc[model].values[0]
            x_values.append(n_parms)
            y_values.append(mean)
            std_errors.append(stderr)

        newData = pd.DataFrame.from_dict({
                'stderr':std_errors, 'mean':y_values, 'n_parms':x_values,
                'modelname':models,
                })
        # Drop any models with NaN values, since that means they had no
        # performance data for one or more columns.
        newData.dropna(axis=0, how='any', inplace=True)
        if newData.size == 0:
            self.script,self.div = (
                    "Error, no plot to display.",
                    "None of the models contained valid performance data."
                    )
            return
        dat_source = ColumnDataSource(newData)

        p = figure(
                tools=tools,
                x_axis_label=("N Parms, model prefix: {0}, "
                              "suffix: {1}".format(self.pre, self.suf)),
                y_axis_label=("Mean {0}, +/- Standard Error"
                              .format(self.measure[0])),
                title="Mean {0} per Model vs Complexity".format(self.measure[0]),
                output_backend=OUTPUT_FORMAT, sizing_mode='stretch_both',
                )

        circles = Circle(
                x='n_parms', y='mean', size=6, fill_color="navy",
                fill_alpha=0.7,
                )
        circle_renderer = p.add_glyph(dat_source, circles)
        hover = self.create_hover()
        hover.renderers = [circle_renderer]
        p.add_tools(hover)
        #p.circle(x_values, y_values, size=6, color="navy", alpha=0.7)
        error_bars_x = []
        error_bars_y = []
        for i, std in enumerate(std_errors):
            error_bars_x.append([x_values[i], x_values[i]])
            error_bars_y.append([y_values[i] - std, y_values[i] + std])
        p.multi_line(
                error_bars_x, error_bars_y, color="firebrick",
                alpha=0.4, line_width=2,
                )

        # workaround to prevent title and toolbar from overlapping
        grid = gridplot(
            [p], ncols=GRID_COLS, sizing_mode='stretch_both'
            )
        self.script, self.div = components(grid)
        self.plot = grid
        if self.display:
            show(grid)


class Tabular(PlotGenerator):
    # TODO: implement this from NARF
    def __init__(
            self, data, measure, models, fair=True, outliers=False,
            extra_cols=[], display=False,
                 ):
        # Use blank measure since tabular has a fixed set of columns
        _measure=[]
        _extra_cols=[
                'n_parms', 'r_test', 'mse_test', 'nlogl_test',
                'mi_test', 'cohere_test',
                ]
        PlotGenerator.__init__(
                self, data=data, measure=_measure, models=models, fair=fair,
                outliers=outliers, extra_cols=_extra_cols, display=display,
                )

    def generate_plot(self):
        # After __init__ self.measure should contain everything that
        # was passed in extra_cols

        if self.fair:
            self.script, self.div = (
                    "Uncheck 'only fair' ",
                    "to use this plot"
                    )
            return

        columns = []
        for m in self.measure:
            if m == 'n_parms':
                columns.append('n_parms')
                continue
            m = m.replace('_test', '')
            if m == 'r':
                mean = 'r(mean)'
                median = 'r(median)'
            else:
                mean = '%s(mn)'%m
                median = '%s(md)'%m
            columns.append(mean)
            columns.append(median)
        index = self.data.index.levels[0].tolist()
        table = pd.DataFrame(
                    # index = list of model names
                    index=index,
                    # columns = list of measures, both mean and median
                    columns=columns
                    )
        self.data.replace(0, np.nan, inplace=True)

        for i, model in enumerate(table.index.tolist()):
            for j, meas in enumerate(self.measure):
                series = self.data[meas]
                if meas == 'n_parms':
                    table.at[model, 'n_parms'] = (
                            series.loc[model].values.tolist()[0]
                            )
                    continue
                else:
                    m = meas.replace('_test', '')
                    if m == 'r':
                        mn = 'r(mean)'
                        md = 'r(median)'
                    else:
                        mn = '%s(mn)'%m
                        md = '%s(md)'%m
                    table.at[model, mn] = np.nanmean(series.loc[model])
                    table.at[model, md] = np.nanmedian(series.loc[model])

                    # num valid cells is the minimum of either the current
                    # value or the number of cells in the series minus number
                    # of NaN values in series.
                    #table.at['valid_cells', mn] = min(
                    #        table.at['valid_cells', mn],
                    #        (series.size - table[mn].isnull().sum())
                    #        )
                    #table.at['valid_cells', md] = min(
                    #        table.at['valid_cells', md],
                    #        (series.size - table[md].isnull().sum())
                    #        )

        table.sort_values('n_parms', axis=0, ascending=False, inplace=True)

        # see pandas style attribute for more options
        positives = [
                'r(mean)', 'r(median)', 'mi(mn)', 'mi(md)',
                'cohere(mn)', 'cohere(md)',
                ]
        negatives = [
                'mse(mn)', 'mse(md)', 'nlogl(mn)', 'nlogl(md)',
                ]
        t = table.style.highlight_max(
                        subset=positives, axis=0, color='darkorange'
                        )\
                   .highlight_min(
                        subset=negatives, axis=0, color='darkorange'
                        )\
                   .set_caption('Prefix: {0}, Suffix: {1}'
                                .format(self.pre, self.suf))\
                   .format("{: 3.4f}")
        self.html = t.render()
        self.plot = t
        if self.display:
            print(t.render())


class Significance(PlotGenerator):

    def extents(self, f):
        # reference:
        # https://bl.ocks.org/fasiha/eff0763ca25777ec849ffead370dc907
        # (calculates the data coordinates of the corners for array chunks)
        if len(f) == 1:
            delta = 1
        else:
            delta = f[1] - f[0]
        return [f[0] - delta/2, f[-1] + delta/2]

    def wilcoxon(self, first, second):
        # Currently using scipy.stats version instead, but
        # kept this here for reference incase we want to customize later.

        # make a list of pairs of items from each list
        # TODO: should lists be randomized first? as-is would compare models
        #       on matching cells

        # make list of absolute differences between pairs
        abs_diff = [abs(f-second[i]) for i, f in enumerate(first)]
        # and list of signs
        signs = [np.sign(f-second[i]) for i, f in enumerate(first)]
        # multiply signs by abs_diffs to get signed ranks
        signed_ranks = [i*signs[i] for i, n in enumerate(abs_diff) if n != 0]
        # calculate W from sum of the signed ranks, and standard deviation
        # of W's sample distribution using number of entries
        w = sum(signed_ranks)
        n = len(signed_ranks)
        stdev_w = math.sqrt((n*(n+1)*((2*n)+1))/6)
        # compute z-score
        z = (w-0.5)/stdev_w
        # return p-value that corresponds to z-score (two-tailed)
        p = 2*(1 - st.norm.cdf(z))

        return p


    def generate_plot(self):
        modelnames = self.data.index.levels[0].tolist()

        array = np.ndarray(
                shape=(len(modelnames), len(modelnames)),
                dtype=float,
                )

        for i, m_one in enumerate(modelnames):
            for j, m_two in enumerate(modelnames):
                # get series of values corresponding to selected measure
                # for each model
                series_one = self.data.loc[m_one][self.measure[0]]
                series_two = self.data.loc[m_two][self.measure[0]]
                if j == i:
                    # if indices equal, on diagonal so no comparison
                    array[i][j] = 0.00
                elif j > i:
                    # if j is larger, below diagonal so get mean difference
                    mean_one = np.mean(series_one)
                    mean_two = np.mean(series_two)
                    array[i][j] = abs(mean_one - mean_two)
                else:
                    # if j is smaller, above diagonal so run t-test and
                    # get p-value
                    # array[i][j] = st.ttest_rel(series_one, series_two)[1]
                    first = series_one.tolist()
                    second = series_two.tolist()
                    # use custom-defined wilcoxon in this class
                    # (wasn't getting the same values, but code
                    #  may be useful later if customization is desired)
                    #array[i][j] = self.wilcoxon(first, second)
                    # use scipy version
                    array[i][j] = st.wilcoxon(first, second)[1]

        xticks = range(len(modelnames))
        yticks = xticks
        minor_xticks = np.arange(-0.5, len(modelnames), 1)
        minor_yticks = np.arange(-0.5, len(modelnames), 1)

        p = plt.figure(figsize=(len(modelnames),len(modelnames)))

        #extent = self.extents(xticks) + self.extents(yticks)
        #img = plt.imshow(
        #        array, aspect='auto', origin='lower',
        #        cmap=plt.get_cmap('RdBu'), interpolation='none',
        #        extent=extent,
        #        )

        ax = plt.gca()

        # ripped from stackoverflow. adds text labels to the grid
        # at positions i,j (model x model)  with text z (value of array at i, j)
        for (i, j), z in np.ndenumerate(array):
            if j == i:
                color="#EBEBEB"
            elif j > i:
                color="#368DFF"
            else:
                if array[i][j] < 0.001:
                    color="#00FF36"
                elif array[i][j] < 0.01:
                    color="#00CC2B"
                elif array[i][j] < 0.05:
                    color="#00A21B"
                else:
                    color="#ABABAB"
            ax.add_patch(mpatch.Rectangle(
                    xy=(j-0.5, i-0.5), width=1.0, height=1.0, angle=0.0,
                    facecolor=color, edgecolor='black',
                    ))
            if j == i:
                # don't draw text for diagonal
                continue
            formatting = '{:.04f}'
            if z <= 0.0001:
                formatting = '{:.2E}'
            ax.text(
                    j, i, formatting.format(z), ha='center', va='center',
                    )

        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_yticks(yticks)
        ax.set_yticklabels(modelnames, fontsize=10)
        ax.set_xticks(xticks)
        ax.set_xticklabels(modelnames, fontsize=10, rotation="vertical")
        ax.set_yticks(minor_yticks, minor=True)
        ax.set_xticks(minor_xticks, minor=True)
        ax.grid(b=False)
        ax.grid(which='minor', color='b', linestyle='-', linewidth=0.75)
        ax.set_title(
                ("Wilcoxon Signed Test on {0}\nprefix: {1}, suffix: {2}"
                 .format(self.measure[0], self.pre, self.suf)),
                ha='center', fontsize = 14,
                )
        blue_patch = mpatch.Patch(
                color='#368DFF', label='Mean Difference', edgecolor='black'
                )
        p001_patch = mpatch.Patch(
                color='#00FF36', label='P < 0.001', edgecolor='black'
                )
        p01_patch = mpatch.Patch(
                color='#00CC2B', label='P < 0.01', edgecolor='black'
                )
        p05_patch = mpatch.Patch(
                color='#00A21B', label='P < 0.05', edgecolor='black'
                )
        nonsig_patch = mpatch.Patch(
                color='#ABABAB', label='Not Significant', edgecolor='black',
                )

        plt.legend(
                #bbox_to_anchor=(0., 1.02, 1., .102), ncol=2,
                bbox_to_anchor=(1.05, 1), ncol=1,
                loc=2, handles=[
                        p05_patch, p01_patch, p001_patch,
                        nonsig_patch, blue_patch,
                        ]
                )
        if self.display:
            plt.show()
        self.plot = p
        img = io.BytesIO()
        plt.savefig(img, bbox_inches='tight')
        #plt.close(p)
        img.seek(0)
        self.img_str = img.read()
