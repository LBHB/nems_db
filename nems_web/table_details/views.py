import pkgutil

from flask import redirect, Response, url_for, render_template

from nems_web.nems_analysis import app
from nems.db import Session, Tables


@app.route('/cell_details/<cellid>')
def cell_details(cellid):
    # Just redirects to celldb penetration info for now
    # Keeping as separate function incase want to add extra stuff later
    # (as opposed to just putting link directly in javascript)
    session = Session()
    gCellMaster = Tables()['gCellMaster']

    url_root = 'http://hyrax.ohsu.edu/celldb/peninfo.php?penid='
    i = cellid.find('-')
    cellid = cellid[:i]
    try:
        result = (
                session.query(gCellMaster)
                .filter(gCellMaster.cellid == cellid)
                .first()
                )
    except Exception as e:
        print(e)
        return Response(
                "Couldn't open cell file -- "
                "this option is disabled for non-LBHB setup."
                )

    if not result:
        # return an error response instead?
        # or make it a dud link? seems wasteful to refresh page
        # could also redirect to celldb home page
        return redirect(url_for('main_view'))
    penid = result.penid
    session.close()

    return redirect(url_root + str(penid))


# TODO: Need to redo this for the new nems repo.
#       I guess it would just use the new nems.keywords defaults to
#       follow the modelspec[i]['fn'] paths and get the docs there?
#       Or was this even worth redoing?
@app.route('/model_details/<modelname>')
def model_details(modelname):
    return render_template('model_details.html',
                           docs=['function temporarily disabled'])
    # test code below this, won't run until response removed
    keyword_list = modelname.split('_')
    kw_funcs=[]
    for kw in keyword_list:
        for importer, modname, ispkg in pkgutil.iter_modules(nk.__path__):
            try:
                kw_funcs.append(getattr(importer.find_module(modname).load_module(modname),kw))
                break
            except:
                pass
    kwdocs = [
            func.__doc__ + '              '
            if func.__doc__ and not isinstance(func, str)
            else 'blank doc string or missing keyword'
            for func in kw_funcs
            ]
    splitdocs = [
            doc.split('\n')
            for doc in kwdocs
            ]
    for i, doc in enumerate(splitdocs):
        doc.insert(0, '     ')
        if not isinstance(kw_funcs[i], str):
            doc.insert(0, kw_funcs[i].__name__)
        else:
            doc.insert(0, kw_funcs[i])

    return render_template('model_details.html', docs=splitdocs)
    # TODO: parse the stack.append() lines inside each kw function
    #       into a description of which module(s) is/are added and which
    #       arguments are specified
    #       ex: fb18ch100 should be translated to:
    #           load file with baphy_utils, args: fs=100,stimfmt='ozgf',chancount=18
    #           add load_mat with args: fs=100
    #           add crossval with args: defaults
    #       alternatively, just specify all of this in docstring for the func
    #       then pull out the doc strings here? probably much easier.
