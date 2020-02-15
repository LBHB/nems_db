"""Reports which cells have been fitted by which models for a given batch.

Code similar to PlotGenerator class, but different enough that it's
been separated here.

"""

import io
import logging

import matplotlib.pyplot as plt
import numpy as np

from bokeh.embed import components
from bokeh.models import (
        HoverTool, SaveTool, WheelZoomTool, PanTool, ResetTool,
        LinearColorMapper, ColumnDataSource, ColorBar, BasicTicker,
        NumeralTickFormatter,
        )
from bokeh.plotting import figure
from bokeh.layouts import gridplot

import nems_web.utilities.pruffix as prx

log = logging.getLogger(__name__)
plt.switch_backend('agg')


class Performance_Report():
    def __init__(self, data, batch, models):
        self.batch = batch
        self.abbr, self.pre, self.suf = prx.find_common(models)
        data.replace(models, self.abbr, inplace=True)
        self.data = data

    def generate_plot(self):
        tools = [
                PanTool(), SaveTool(), WheelZoomTool(),
                ResetTool(), HoverTool()
                ]
        colors = [
                '#084594', '#2171b5', '#4292c6', '#6baed6', '#9ecae1',
                '#c6dbef', '#deebf7', '#f7fbff'
                ]
        # reversed puts darker colors for higher values
        colors = [c for c in reversed(colors)]
        mapper = LinearColorMapper(
                palette=colors, low=self.data.r_test.min(),
                high=self.data.r_test.max()
                )
        source = ColumnDataSource(self.data)
        p = figure(
                title=("batch {0}, model prefix: {1}, suffix: {2}"
                       .format(self.batch, self.pre, self.suf)),
                x_range=list(set(self.data['cellid'].values.tolist())),
                y_range=list(set(self.data['modelname'].values.tolist())),
                tools=tools, toolbar_location='above',
                )

        p.grid.grid_line_color = None
        p.axis.axis_line_color = None
        p.axis.major_tick_line_color = None
        p.axis.major_label_text_font_size = '10pt'
        p.axis.major_label_standoff = 0
        p.xaxis.visible = False

        p.rect(
                x='cellid', y='modelname', width=1, height=1, source=source,
                fill_color={'field': 'r_test', 'transform': mapper},
                line_color=None,
                )

        color_bar = ColorBar(
                color_mapper=mapper, major_label_text_font_size="10pt",
                ticker=BasicTicker(desired_num_ticks=len(colors)),
                formatter=NumeralTickFormatter(format="0.00"),
                label_standoff=10, border_line_color=None, location=(0, 0),
                )

        p.add_layout(color_bar, 'right')
        p.select_one(HoverTool).tooltips = [
                ('r_test: ', '@r_test'),
                ('model: ', '@modelname'),
                ('cellid: ', '@cellid'),
                ]

        grid = gridplot(p, ncols=1, responsive=True)
        self.script, self.div = components(grid)


class Fit_Report():
    def __init__(self, data):
        self.data = data

    def generate_plot(self):
        array = self.data.values
        cols = self.data.columns.tolist()
        rows = self.data.index.tolist()

        xticks = range(len(cols))
        yticks = range(len(rows))
        minor_xticks = np.arange(-0.5, len(cols), 1)
        minor_yticks = np.arange(-0.5, len(rows), 1)
        extent = self.extents(xticks) + self.extents(yticks)

        p = plt.figure(figsize=(len(cols), len(rows)/4))
        img = plt.imshow(
                array, aspect='auto', origin='lower',
                cmap=plt.get_cmap('RdBu', 7), interpolation='none',
                extent=extent,
                )
        img.set_clim(0, 0.6)
        ax = plt.gca()

        abbr, pre, suf = prx.find_common(cols)

        ax.set_ylabel('')
        ax.set_xlabel('Model, prefix: {0}, suffix: {1}'.format(pre, suf))
        ax.set_yticks(yticks)
        ax.set_yticklabels(rows, fontsize=8)
        ax.set_xticks(xticks)
        ax.set_xticklabels(abbr, fontsize=8, rotation='vertical')
        ax.set_yticks(minor_yticks, minor=True)
        ax.set_xticks(minor_xticks, minor=True)
        ax.grid(b=False)
        ax.grid(which='minor', color='w', linestyle='-', linewidth=0.75)
        cbar = plt.colorbar()
        tick_locs = (np.arange(0, 0.7, 0.1) + 0.05) * (7 - 1) / 7
        cbar.set_ticks(tick_locs)
        cbar.set_ticklabels([
                'Dead', '', '', 'Missing', 'In Progress', 'Not Started',
                'Complete',
                ])
        # Should set the colorbar to sit at the top of the figure (y=1),
        # which it does, but also forces plot way down, which is bad.
        #cax = cbar.ax
        #current_pos = cax.get_position()
        #cax.set_position(
        #        [current_pos.x0, 1, current_pos.width, current_pos.height]
        #        )

        img = io.BytesIO()
        plt.savefig(img, bbox_inches='tight')
        plt.close(p)
        img.seek(0)
        self.img_str = img.read()

    def extents(self, f):
        # reference:
        # https://bl.ocks.org/fasiha/eff0763ca25777ec849ffead370dc907
        # (calculates the data coordinates of the corners for array chunks)
        if len(f) == 1:
            delta = 1
        else:
            delta = f[1] - f[0]
        return [f[0] - delta/2, f[-1] + delta/2]
