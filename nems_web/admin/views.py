import logging
import pkgutil
import importlib
import inspect

from flask import url_for, Response, jsonify
from flask_login import login_required

from nems_web.nems_analysis import app
from nems_web.account_management.views import get_current_user

import nems_web

log =  logging.getLogger(__name__)

# referenced from stackoverflow.com/questions/13317536/
# get-a-list-of-all-routes-defined-in-the-app
def has_no_empty_params(rule):
    defaults = rule.defaults if rule.defaults is not None else ()
    arguments = rule.arguments if rule.arguments is not None else ()
    return len(defaults) >= len(arguments)

@app.route("/site_map")
@login_required
def site_map():
    # only allow users with admin privileges to use this function
    user = get_current_user()
    if user.sec_lvl < 9:
        return Response("Must have admin privileges to view site map")

    # get list of defined url routes and their matching function endpoints
    links = []
    for rule in app.url_map.iter_rules():
        if "GET" in rule.methods and has_no_empty_params(rule):
            url = url_for(rule.endpoint, **(rule.defaults or {}))
            links.append([url, rule.endpoint])

    # search through web directory and match function endpoint strings to the
    # modules that define those functions
    package = nems_web
    for importer, modname, ispkg in pkgutil.iter_modules(package.__path__):
        if ispkg:
            subpkg = importlib.import_module("nems_web.{0}".format(modname))
            sub_name = modname
            for importer, modname, ispkg in pkgutil.iter_modules(subpkg.__path__):
                if "views" not in modname:
                    continue

                mod = importlib.import_module("nems_web.{0}.views".format(sub_name))
                for link in links:
                    function_names = [
                            f[0] for f in
                            inspect.getmembers(mod, inspect.isfunction)
                            ]
                    log.info("testing...")
                    log.info("is: {0}  in  {1}  ?".format(link[1], function_names))
                    if link[1] in function_names:
                        log.info("got past second if statement")
                        # if matching function is found in module, replace
                        # endpoint string with module path
                        path = mod.__file__
                        # chop off everything before nems parent package
                        nems_idx = path.find('nems/nems_web')
                        path = path[nems_idx:]
                        link[1] = path

    log.info("Defined routes:\n")
    html = "<h3> Defined routes: </h3>"
    for link in links:
        log.info("url route for: {0} \n goes to endpoint: {1}".format(link[0], link[1]))
        html += (
                "<br><p> url:    {0} </p><p> is defined in:   {1}</p>"
                .format(link[0], link[1])
                )

    return Response(html)

@app.route('/reload_modules')
@login_required
def reload_modules():
    # import the nems0.module package at time of routing
    package = importlib.import_module('nems0.modules')
    # get list of references to modules in nems0.modules package
    modnames = [
            modname for importer, modname, ispkg
            in pkgutil.iter_modules(package.__path__)
            ]
    mods = [
            importlib.import_module('nems0.modules.{0}'.format(m))
            for m in modnames
            ]
    # for each module in the list, reload it
    for mod in mods:
        importlib.reload(mod)

    return jsonify(success=True)
