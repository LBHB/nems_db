# KEYWORD TRANSLATIONS CHEATSHEET
Also see: nems/docs/keywords.md

## Loaders / Preprocessors

But in the meantime, most recording_uri keywords are the same except for the insertion
of periods between options and the addition of 'fs' preceeding the sampling rate.
For example:
```
old_loader = 'ozgf100ch18n'
new_loader = 'ozgf.fs100.ch18.n'
```
These are all handled by nems_db.xform_wrappers.get_recording_uri and are separate
from the xforms and modelspec keywords.

The standard, literal 'loader' is now 'ld' in nems0.plugins.default_loaders.
In the future there will likely be a short-cut added to make 'ld' implicit since
every model uses it and all it does is load a recording into memory. However,
in the meantime, the keyword 'ld' will likely need to follow the recording_uri
keyword in any model you specify unless you're loading recordings another way.

Ex:
```
new_loader = 'ozgf.fs100.ch18-ld.n'
```

Other functions that used to be handled by the old 'loaders,' like splitting the data
into est and val sets or masking certain epochs, are now defined as separate
xforms keywords in nems_lbhb.plugins.lbhb_preprocessors. So, to get all the same functionality as the old modelnames, you will need to add a couple extra keywords to add in
what was removed from ozgf, env, psth etc. In cases where multiple functions are almost
always combined, new 'alias' or 'shortcut' keywords may be added to simplify modelnames.
However, it is strongly encouraged that this is only done in cases where the keywords
are virtually always used together and perform logically related functions.

Ex:
```
old_loader = 'ozgf.fs100.ch18'
new_loader1 = 'ozgf.fs100.ch18-ld-splitep-avgep'
new_loader2 = 'ozgf.fs100.ch18-ld-sev'  # sev is a short-cut keyword for splitep-avgep
```

## Modules

Most modules will just need a period inserted before the first numeral.

Ex:
```
old_wc = 'wc15x2'
new_wc = 'wc.15x2'

old_dexp = 'dexp1'
new_dexp = 'dexp.1'
```

But modules with more complicated options have to be moved around a litle more,
and may be combined into a single keyword.

In general, they should adhere to the format:
```
kwhead.requiredOption.otherOption(s)
```

Ex:
```
old_stpz = 'stpz2'
old_stpb = 'stp2b'

new_stpzb = 'stp.2.z.b'
```


## Fitters

Fitter keywords look the most different. They follow the same format as the loaders
and modules, but a lot of options were combined into just two fitters: basic and iter.

Some examples:
```
old = 'fit01'
new = 'basic'

old = 'fit01a'  # or 'basicqk'
new = 'basic.mi1000.t5'  # for max_iterations = 1000 and tolerance = 10^-5

old = 'iter-T3-T5-S1x2x3-ti50-fi10'
new = 'iter.T3,5.S1,2,3.ti50.fi10'
```

Additionally, the fitters are now agnostic to jackknifing and initialization since
those functions have been separated out into new initialization keywords

ex:
```
old = 'state01-jkm'
new = 'init.st-jk.nf5.m-basic' # for state, nfold w/ 5 folds, and use split_for_jackknife
```

If you are having trouble translating any keywords you've been using into the new system,
just ask Jacob! Or take a look at the function definitions in nems0.plugins.default_keywords
(or nems_lbhb.plugins.lbhb_keywords once custom ones are added) if you want to figure out
the parsing for yourself. None of the regex was too complicated so hopefully most of it
is straightforward.
