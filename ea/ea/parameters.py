"""
This package allows us to manage adaptive parameters
"""
import numpy as np

def is_value_right(value):
    return is_float(value) or value == 'n' or value == 'a'

LP=0

def get_parameter(name, value, normal_values, PS):
    """
    Get a parameter value
    """
    global LP
    LP = 10*PS
    mean, std = normal_values

    if is_float(value):
        return value
    elif value == 'n':
        return {'name': name, 'type': 'normal', 'mean': mean, 'std': std}
    elif value == 'a':
        return {'name': name, 'type': 'learn', 'mean': 0.5, 'std': 0.1, 'count' : 0, 'success': []}
    else:
        raise ValueError(name)

def is_float(value):
    """
    Return if the value is float
    """
    try:
        float(value)
        return True
    except ValueError:
        return False
    except TypeError:
        return False
        
def get_value(parameter):
    """
    Get the current value of a parameter
    """
    if is_float(parameter):
        value = float(parameter)
    else:
        value = np.random.normal(parameter['mean'], parameter['std'])

        if parameter['type'] == 'learn':
            parameter['count'] += 1
            parameter['value'] = value

    return value

def parameter_result(parameter, successful):
    """
    Indicate if previous get_value obtain succesful results or not..

    After LP iterations this information is used for update the values
    """
    # Ignore if it is not learn mode
    if is_float(parameter):
        return
    elif parameter['type'] != 'learn':
        return

    global LP

    if successful:
        parameter['success'].append(parameter['value'])

    if parameter['count'] == LP:

        if len(parameter['success']) > 1:
            parameter['mean'] = np.mean(parameter['success'])

#        print parameter['name'], parameter['mean']
        parameter['success'] = []
        parameter['count'] = 0
