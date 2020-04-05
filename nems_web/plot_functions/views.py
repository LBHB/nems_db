"""View functions for handling Submit Plot button.

Query database with batch, cell and model selection.
Filter out cells that don't meet minimum SNR/Iso/SNRi criteria.
Pass the returned Results dataframe - along with measure, onlyFair and
includeOutliers criterio - to a PlotGenerator subclass based on the selected
plot type.
Use that PlotGenerator's generate_plot() method to generate <script> and <div>
components to pass back to the JS ajax function, which will insert them into
the display area in the browser.

"""

import logging
from base64 import b64encode

from flask import render_template, jsonify, request

from nems_web.nems_analysis import app
import nems_db.plot_helpers as dbp

log = logging.getLogger(__name__)


@app.route('/generate_plot_html')
def generate_plot_html():

    plot_type = request.args.get('plotType')
    batch = request.args.get('bSelected')[:3]
    models = request.args.getlist('mSelected[]')
    cells = request.args.getlist('cSelected[]')
    measure = request.args.get('measure')
    only_fair = request.args.get('onlyFair')
    if int(only_fair):
        only_fair = True
    else:
        only_fair = False
    include_outliers = request.args.get('includeOutliers')
    if int(include_outliers):
        include_outliers = True
    else:
        include_outliers = False
    snr = float(request.args.get('snr'))
    iso = float(request.args.get('iso'))
    snr_idx = float(request.args.get('snri'))

    cells = dbp.get_filtered_cells(cells, snr, iso, snr_idx)
    plot = dbp.get_plot(cells, models, batch, measure, plot_type, only_fair,
                        include_outliers, display=False)
    log.debug("Plot successfully initialized")
    if plot.emptycheck:
        log.info('Plot checked empty after forming data array')
        return jsonify(script='Empty', div='Plot')
    else:
        plot.generate_plot()

    if hasattr(plot, 'script') and hasattr(plot, 'div'):
        return jsonify(script=plot.script, div=plot.div)
    elif hasattr(plot, 'html'):
        return jsonify(html=plot.html)
    elif hasattr(plot, 'img_str'):
        image = str(b64encode(plot.img_str))[2:-1]
        return jsonify(image=image)
    else:
        return jsonify(script="Couldn't find anything ", div="to return")


@app.route('/plot_window')
def plot_window():
    return render_template('/plot/plot.html')
