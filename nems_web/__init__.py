import logging
import ast
log = logging.getLogger(__name__)

# for each setting defined in db_defaults and web_defaults:
# 1) check if it exists in the environment
# 2) if not, check if appropriate settings.py file exists.
#    -> if it does, use the value there and then export to env
# 3) if not 1 or 2, use value in appropriate defaults.py file and export.


def load_config():
    import os
    from nems_web.configs import defaults

    # leave off 'defaults.py' at end
    configs_path = os.path.dirname(os.path.abspath(defaults.__file__))

    # if web_settings.py exists, import it and grab its __dir__() entries
    # otherwise, use an empty dictionary and create a blank file
    # to point user to the right place.
    try:
        from nems_web.configs import settings
    except ImportError:
        web_path = os.path.join(configs_path, 'settings.py')
        # this should be equivalent to `touch path/to/configs/settings.py`
        with open(web_path, 'a'):
            os.utime(web_path, None)
        log.info("No web_settings.py found in configs directory,"
                 " generating blank file ... ")
        from nems_web.configs import settings

    cached_config = {}
    _init_settings(os.environ, defaults, settings, cached_config)

    return cached_config


def _init_settings(environ, defaults, settings, cache):
    for s in defaults.__dir__():
        if s.startswith('__'):
            # Ignore python magic variables. Everything else in
            # the defaults files should be valid settings.
            pass
        else:
            if s in environ:
                # If it's already in the environment, don't need
                # to do anything else.
                pass
            elif hasattr(settings, s):
                log.info("Found setting: %s in %s, setting "
                         "value in environment ... ", s, settings.__name__)
                d = getattr(settings, s)
                if d is None:
                    d = ''
                environ[s] = str(d)
            else:
                log.info("No value specified for: %s. Using default value "
                         "in %s", s, defaults.__name__)
                d = getattr(defaults, s)
                if d is None:
                    d = ''
                environ[s] = str(d)
            cache[s] = environ[s]


_cached_config = load_config()
# Other modules in the package that need access to the settings can
# pull them from the environment.

# TODO: Alternatively, use a get_setting function like bburan has in nems?
#       So other mods would call nems_db.get_setting(xxx)?


def get_setting(s):
    s = _cached_config.get(s, None)

    # Check for true/false in non-python format to keep literal_eval happy
    if isinstance(s, str) and s.lower() == 'false':
        s = "False"
    if isinstance(s, str) and s.lower() == 'true':
        s = "True"

    # Necessary since environment variables can only hold strings,
    # but config settings some times need to be other types.
    # NOTE: Will not work for dictionaries, but tested fine so far
    #       with strings, lists, ints, floats, and booleans.
    try:
        # s is something other than a string
        s = ast.literal_eval(s)
    except (SyntaxError, ValueError):
        # setting is just a string
        pass
    return s

# TODO: Wouldn't caching the variables in os env make changing them from
#       the .py files difficult? Would probably have to restart
#       console/open new terminal each time. So add some utility function
#       for resetting the variables stored in the os env?
#       Shouldn't be changing config that often anyway but might be annoying.
def reload_settings():
    from os import environ
    global _cached_config

    for s in _cached_config:
        environ.pop(s, None)
    _cached_config = load_config()
