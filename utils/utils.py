

from typing import Union, Callable
import pandas as pd
import numpy as np
from collections import defaultdict

def SanityCheck(data : Union[pd.Series, np.array], type_ : Callable):
    assert callable(type_) , 'Type specified must be callable'
    
    def purge(val, type_):
        try:
            return type_(val)
        except:
            return np.nan
        
    if isinstance(data, pd.Series):
        return data.apply(lambda x: purge(x, type_)) 
    elif isinstance(data, dict):
        temp = [purge(x, type_) for x in data.values()]
        return dict(zip(data.keys(), temp))
    elif isinstance(data, np.array):
        return np.array([purge(x, type_) for x in data])
    else:
        raise NotImplementedError
    

        