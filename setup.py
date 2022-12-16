import codecs
import os.path
from setuptools import find_packages, setup

NAME = 'nems_db'

VERSION = '0.2'

with codecs.open('README.md', encoding='utf-8') as f:
    long_description = f.read()

GENERAL_REQUIRES = [
        'pandas', 'pymysql', 'sqlalchemy',
        'numpy', 'scipy', 'matplotlib', 'mpld3', 'boto3',
        'bcrypt', 'pillow', 'numpy', 'scipy', 'requests',
        'h5py', 'configparser', 'tables'
        ]
# pycharm also requires: tornado
# TF modules require tensorflow
# GUI requires pyqt, pyqtgraph

# needed for web server:
OTHER_REQUIRES = [
        'flask', 'flask_restful', 'pandas','pymysql', 'sqlalchemy',
        'numpy', 'scipy', 'matplotlib', 'mpld3', 'boto3', 'bokeh',
        'flask-socketio', 'eventlet', 'bcrypt', 'flask-WTF', 'flask-login',
        'flask-bcrypt', 'flask-assets', 'gevent', 'pillow', 'mplcursors'
        ]

setup(
    name=NAME,
    version=VERSION,
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    zip_safe=True,
    author='LBHB',
    author_email='lbhb.ohsu@gmail.com',
    description='Neural Encoding Model System - Database support',
    long_description=long_description,
    url='http://neuralprediction.org',
    install_requires=GENERAL_REQUIRES,
    classifiers=[],
    entry_points={
        'console_scripts': [
            #'nems-fit-single=nems.application.cmd_launcher:main',
        ],
    }
)
