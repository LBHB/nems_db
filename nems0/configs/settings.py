'''
Configuration file for NEMS containing reasonable defaults

Variable names in capital letters indicate this is a setting that can be
overriden by a custom file that the NEMS_CONFIG environment variable points to.
Use virtual environments (via virtualenv or conda) for creating separate
installations of NEMS that use different configuration files.
'''

# Note for developers. See `__init__.py` to understand how this integrates with
# the file specified by NEMS_CONFIG.
import datetime


################################################################################
# System information
################################################################################
# Name of computer
# SYSTEM = socket.gethostname()


################################################################################
# Logging configuration
################################################################################
# Folder to store log file in
# NEMS_LOG_ROOT = '/tmp/nems'

# Filename to save log file in
# NEMS_LOG_FILENAME = datetime.datetime.now().strftime('NEMS %Y-%m-%d %H%M%S.log')

# Format for messages saved to file
# NEMS_LOG_FILE_FORMAT = '[%(relativeCreated)d %(thread)d %(name)s - %(levelname)s] %(message)s'

# Format for messages printed to console
# NEMS_LOG_CONSOLE_FORMAT = '[%(name)s %(levelname)s] %(message)s'

# Logging level for file
# NEMS_LOG_FILE_LEVEL = 'DEBUG'

# Logging level for console
# NEMS_LOG_CONSOLE_LEVEL = 'DEBUG'

################################################################################
# Display tweaks
################################################################################
# Name of computer
FILTER_CMAP = 'bwr'   # alternative is 'RdYlBu_r'
#FILTER_CMAP = 'jet'
#WEIGHTS_CMAP = 'bwr'
#FILTER_INTERPOLATION = (1, 3)


################################################################################
# Plugins Registries
################################################################################
# Keyword Plugins, updates nems0.keywords.defaults
# ex: KEYWORD_PLUGINS = ['/path/to/keywords/', '/second/path/']
KEYWORD_PLUGINS = ['nems_lbhb.plugins.lbhb_keywords',
                   'nems_lbhb.rdt.plugins']

XFORMS_PLUGINS = ['nems_lbhb.plugins.lbhb_loaders',
                  'nems_lbhb.plugins.lbhb_preprocessors',
                  'nems_lbhb.plugins.lbhb_postprocessors',
                  'nems_lbhb.plugins.lbhb_initializers',
                  'nems_lbhb.plugins.lbhb_fitters',
                  'nems_lbhb.rdt.wrapper_plugins']

#LIB_PLUGINS = ['nems_lbhb.columbia_helpers']
LIB_PLUGINS = ['nems_lbhb.modules.state']

################################################################################
# Data & database
################################################################################

NEMS_RESULTS_DIR = '/auto/data/nems_db/results'
NEMS_RECORDINGS_DIR = '/auto/data/nems_db/recordings'

SQL_ENGINE = 'mysql'
#SQL_ENGINE = 'sqlite'

MYSQL_HOST = 'hyrax.ohsu.edu'
MYSQL_USER = 'nems'
MYSQL_PASS = 'nine1997'
MYSQL_DB = 'cell'
MYSQL_PORT = '3306'

# Default paths passed to command prompt for model queue
DEFAULT_EXEC_PATH = '/auto/users/svd/bin/miniconda3/envs/nems/bin/python'
DEFAULT_SCRIPT_PATH = '/auto/users/svd/python/nems_db/nems_fit_single.py'

QUEUE_TICK_EXTERNAL_CMD = '/auto/users/svd/python/nems_db/bin/qsetload'

NEMS_BAPHY_API_PORT = '3003'
NEMS_BAPHY_API_HOST = 'hyrax.ohsu.edu'
USE_NEMS_BAPHY_API = False
#USE_NEMS_BAPHY_API = True
