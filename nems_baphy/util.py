import os


def load_env_vars(req_vars):
    ''' Returns a dict of the credentials found in the environment. '''
    env = os.environ
    creds = dict((k, env[k]) for k in req_vars if k in env)
    return creds


def ensure_env_vars(req_vars):
    ''' 
    Returns a dict of the credentials found in the environment. 
    Raises an exception if any variables were not found.
    '''
    creds = load_env_vars(req_vars)
    if not all(c in creds for c in req_vars):
        raise ValueError('Required environment variables not all provided: ' +
                         str(req_vars))
    return creds
