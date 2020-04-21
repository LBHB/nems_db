import helpers as helper
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from nems import get_setting
dump_path = get_setting('NEMS_RESULTS_DIR')

r0_threshold = 0
octave_cutoff = 0.5
group_files = True

dump_results = 'd_pup_afl_sdexp.csv'
model_string = 'st.pup.afl'
p0_model = 'st.pup0.afl'
b0_model = 'st.pup.afl0'
shuf_model = 'st.pup0.afl0'

A1 = helper.preprocess_sdexp_dump(dump_results,
                                  batch=307,
                                  full_model=model_string,
                                  p0=p0_model,
                                  b0=b0_model,
                                  shuf_model=shuf_model,
                                  r0_threshold=r0_threshold,
                                  octave_cutoff=octave_cutoff,
                                  path=dump_path)
A1['difficulty'] = [0 if x<2 else 1 for x in A1['difficulty']]

IC = helper.preprocess_sdexp_dump(dump_results,
                                  batch=309,
                                  full_model=model_string,
                                  p0=p0_model,
                                  b0=b0_model,
                                  shuf_model=shuf_model,
                                  r0_threshold=r0_threshold,
                                  octave_cutoff=octave_cutoff,
                                  path=dump_path)
IC['difficulty'] = [0 if x<2 else 1 for x in IC['difficulty']]


# plot individual model results
f, ax = helper.stripplot_df(A1, hue='ON_BF', group_files=group_files)
f.canvas.set_window_title('A1')
f, ax = helper.stripplot_df(IC, hue='ON_BF', group_files=group_files)
f.canvas.set_window_title('IC')

plt.show()