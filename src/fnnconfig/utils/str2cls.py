import importlib

from fnnconfig import *

def new_cls(fullname: str, args=[], kwargs={}):
    module_name_ = '.'.join(fullname.split('.')[:-1])
    cls_name_ = fullname.split('.')[-1]
    
    module_ = importlib.import_module(module_name_)
    cls_ = getattr(module_, cls_name_)
    obj_ = cls_(*args, **kwargs)

    return obj_


def dict2cls(d: dict, recursive=False):
    t = d.pop('type')
    if type(t) == str:
        if recursive:
            for k in d:
                if type(d[k]) == dict \
                    and 'type' in d[k] \
                    and k not in set(['optimizer', 'module']):
                    d[k] = dict2cls(d[k])
        return new_cls(t, kwargs=d)
    else:
        return t(**d)
