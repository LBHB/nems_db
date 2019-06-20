""" Reference:
    flask.pocoo.org/snippets/63/ and .../62/
    flask-wtf.readthedocs.io/en/stable/install.html
    flask-login.readthedocs.io/en/latest/

"""

import logging
from urllib.parse import urlparse, urljoin

from flask import redirect, request, url_for, render_template, Response
from flask_login import (
        LoginManager, login_required, login_user, logout_user, current_user
        )
from flask_bcrypt import Bcrypt

from nems_web.nems_analysis import app
from nems.db import Session, Tables
from nems_web.account_management.forms import LoginForm, RegistrationForm

log = logging.getLogger(__name__)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.session_protection = 'basic'


def login_required_callback():
    return Response('Login required')


login_manager.unauthorized = login_required_callback
bcrypt = Bcrypt(app)


@login_manager.user_loader
def load_user(user_id):
    session = Session()
    NarfUsers = Tables()['NarfUsers']
    try:
        # get email match from user database
        # (needs to be stored as unicode per flask-login)
        sqla_user = (
                session.query(NarfUsers)
                .filter(NarfUsers.email == user_id)
                .first()
                )
        if not sqla_user:
            sqla_user = BlankUser()
        # assign attrs from table object to active user instance
        user = User(
                username=sqla_user.username,
                # password should be stored as bcrypt hash
                # (generated when registered)
                password=sqla_user.password,
                labgroup=sqla_user.labgroup,
                sec_lvl=sqla_user.sec_lvl,
                )
        return user
    except Exception as e:
        log.exception(e)
        return None
    finally:
        session.close()


@app.route('/login', methods=['GET', 'POST'])
def login():
    errors = []
    form = LoginForm(request.form)
    if request.method == 'POST':
        if form.validate():
            user = load_user(form.email.data)
            if user:
                if bcrypt.check_password_hash(user.password, form.password.data):
                    user.authenticated = True
                    login_user(user, remember=True)
                    return redirect(url_for('main_view'))
            else:
                errors += ['User returned None']
        else:
            log.info('form errors:')
            log.info(form.errors)
            return render_template(
                    'account_management/login.html',
                    form=form, errors=form.errors,
                    )

    return render_template(
            'account_management/login.html',
            form=form, errors=errors,
            )


@app.route('/logout')
@login_required
def logout():
    user = current_user
    user.authenticated = False
    logout_user()

    return redirect(url_for('main_view'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    errors = ''
    form = RegistrationForm(request.form)
    if request.method == 'POST':
        if form.validate():
            session = Session()
            NarfUsers = Tables()['NarfUsers']
            new_user = NarfUsers(
                    username=form.username.data,
                    password=bcrypt.generate_password_hash(form.password.data),
                    email=(form.email.data).encode('utf-8'),
                    firstname=form.firstname.data,
                    lastname=form.lastname.data,
                    )

            session.add(new_user)
            session.commit()
            session.close()

            return redirect(url_for('login'))
        else:
            log.info('form errors:')
            log.info(form.errors)
            return render_template(
                    'account_management/register.html',
                    form=form, errors=form.errors,
                    )

    return render_template(
            'account_management/register.html',
            form=form, errors=errors,
            )


class User():
    # flask-login required class and methods
    # have this inherit from sqlalchemy base? then
    # these methods just get added on, but all db fields visible as attr.

    def __init__(self, username, password, labgroup=None, sec_lvl=0):
        self.username = username
        self.password = password
        if labgroup:
            self.labgroup = labgroup
        else:
            self.labgroup = 'SPECIAL_NONE_FLAG'
        self.sec_lvl = sec_lvl
        self.authenticated = False

    def is_authenticated(self):
        # self.authenticated should be false until password verified in
        # login function.
        # should be reset to faulse on logout or if session ends.
        return self.authenticated

    def is_active(self):
        # All users active - can implement checks here (like e-mail conf)
        # if desired.
        return True

    def is_anonymous(self):
        # No users are anonymous - can implement this later if desired.
        return False

    def get_id(self):
        # user_id should be the unicode rep of e-mail address
        # (should be stored in db table in that format)
        session = Session()
        NarfUsers = Tables()['NarfUsers']
        user_id = (
                session.query(NarfUsers.email)
                .filter(NarfUsers.username == self.username)
                .first()
                )
        session.close()
        return user_id


class BlankUser():
    """ Blank dummy user, only used to avoid errors when checking for
    labgroup/username match in other views.

    """

    username = ''
    password = ''
    labgroup = 'SPECIAL_NONE_FLAG'
    sec_lvl = '1'
    authenticated = False

    def __init__(self):
        pass


def get_current_user():
    """Uses flask-login's current_user function to load the current global
    user object. If it returns flask-login's default mixin class instead of
    a User object from the class defined in this module, it will return a
    BlankUser object. Otherwise, it will return the currently stored User.

    """

    user = current_user
    try:
        # Default flask_login user class doesn't have this attribute,
        # so if no user is logged in this will throw an error.
        labgroup = user.labgroup
    except:
        user = BlankUser()
    return user


# Functions from snippets 62
def is_safe_url(target):
    ref_url = urlparse(request.host_url)
    test_url = urlparse(urljoin(request.host_url, target))
    return ((test_url.scheme in ('http', 'https'))
            and (ref_url.netloc == test_url.netloc))


def get_redirect_target():
    for target in request.values.get('next'), request.referrer:
        if not target:
            continue
        if is_safe_url(target):
            return target


def redirect_back(endpoint, **values):
    target = request.form['next']
    if not target or not is_safe_url(target):
        target = url_for(endpoint, **values)
    return redirect(target)
