"""Forms for account management views."""

from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField
from wtforms.validators import (
        DataRequired, Length, EqualTo, Email, InputRequired, ValidationError
        )

from nems.db import Session, Tables


class UsernameAvailable():
    # http://wtforms.readthedocs.io/en/latest/validators.html

    def __init__(self, message=None):
        if message:
            self.message = message
        else:
            self.message = "Username already exists. Please choose another."

    def __call__(self, form, field):
        session = Session()
        Users = Tables()['Users']
        exists = (
                session.query(Users)
                .filter(Users.username == field.data)
                .first()
                )
        if exists:
            session.close()
            raise ValidationError(self.message)
        session.close()


class EmailAvailable():

    def __init__(self, message=None):
        if message:
            self.message = message
        else:
            self.message = "An account with that email address already exists."

    def __call__(self, form, field):
        session = Session()
        Users = Tables()['Users']
        exists = (
                session.query(Users)
                .filter(Users.email == field.data)
                .first()
                )
        if exists:
            session.close()
            raise ValidationError(self.message)
        session.close()


class LoginForm(FlaskForm):
    email = StringField(
            'Email',
            validators=[
                    DataRequired(),
                    InputRequired(),
                    ],
            )
    password = PasswordField(
            'Password',
            validators=[
                    DataRequired(),
                    InputRequired(),
                    ],
            )


class RegistrationForm(FlaskForm):
    username = StringField(
            'Username',
            validators=[
                    DataRequired(),
                    InputRequired(),
                    Length(min=4, max=25),
                    UsernameAvailable(),
                    ],
            )
    email = StringField(
            'Email',
            validators=[
                    DataRequired(),
                    InputRequired(),
                    Length(min=6),
                    # crude reg-ex validation per wtforms website, but
                    # probably good enough for now.
                    Email('Must use a valid e-mail address'),
                    EmailAvailable(),
                    ],
            )
    password = PasswordField(
            'Password',
            validators=[
                    DataRequired(),
                    InputRequired(),
                    EqualTo('confirm',
                            message='Passwords must match'
                            ),
                    ],
            )
    confirm = PasswordField(
            'Repeat password'
            )

    firstname = StringField(
            'First name',
            validators=[
                    DataRequired(),
                    ],
            )

    lastname = StringField(
            'Last name',
            validators=[
                    DataRequired(),
                    ],
            )
