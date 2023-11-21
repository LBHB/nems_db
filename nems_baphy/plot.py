import re
import os
import io
from pathlib import Path
import mimetypes

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
from flask import abort, Response, request
from flask_restful import Resource

from nems_lbhb.projects.behav_analysis_scripts import behav

import nems_lbhb.baphy as baphy
from nems0 import get_setting


class Plot(Resource):
    '''
    An interface to BAPHY that returns NEMS-compatable signal objects
    '''
    def __init__(self, **kwargs):
        pass

    def get(self, plottype):
        '''
        parameters to pass to plottype stored in flask.request
        '''
        animal = request.args.get('animal','LMD')
        list = [y for y in request.args.get('list','').split(',')]
        print(f"Passed list: {list}")

        if plottype=='behaviorbar':
            if animal=="LMD":
                fig = behav.lemon_space_snr_bar()
            elif animal=="SLJ":
                fig = behav.slippy_space_snr_bar()
        elif plottype=='behaviortime':
            b = behav.behav(animal, 'NFB', days='all', migrate_only=True, non_migrate_blocks=True)
            XVARIABLE = [x for x in request.args.get('X','snr').split(',')]
            YVARIABLE = request.args.get('Y', 'correct')

            if len(list)>0:
                day_list = list
            elif animal=="SLJ":
                day_list = ['SLJ063Ta', 'SLJ064Ta', 'SLJ065Ta', 'SLJ066Ta', 'SLJ067Ta', 'SLJ068Ta',
                             'SLJ069Ta', 'SLJ070Ta', 'SLJ071Ta', 'SLJ072Ta']
            elif animal=="LMD":
                day_list = ['LMD074Ta', 'LMD075Ta', 'LMD076Ta', 'LMD077Ta', 'LMD078Ta',
                            'LMD079Ta', 'LMD080Ta', 'LMD081Ta', 'LMD082Ta']
            fig = b.perform_over_time(XVARIABLE, YVARIABLE, day_list)

        output = io.BytesIO()
        FigureCanvas(fig).print_png(output)
        return Response(output.getvalue(), mimetype='image/png')

    def put(self, batch, file):
        abort(400, 'Not yet implemented')

    def delete(self, batch, file):
        abort(400, 'Not yet Implemented')

