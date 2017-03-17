import pystan as st
from functools import wraps

class Model(object):
    def __init__(self, *args, **kwargs):
        print('start')

    def __enter__(stanfunc):
        print('enter')
        stanfunc._staring_names = globals().keys()
        return stanfunc

    def __exit__(stanfunc, type, value, traceback):
        print('exit')
        current_names = globals()
        new_names = current_names.keys().difference(stanfunc._starting_names)
        data = {k:current_names[k] for k in new_names}
        @wraps(stanfunc)
        def _stanfunc(*args, **kwargs):
            return stanfunc(*args, data=data, **kwargs)
        return _stanfunc

    def __call__(stanfunc):
        st.stan(**vars(stanfunc))