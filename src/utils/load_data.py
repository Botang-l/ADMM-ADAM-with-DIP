import numpy as np
from scipy.io import loadmat

def get_matlab_variable(mat_file_path):
    matlab_variable = loadmat(mat_file_path)

    matlab_variable_name = list(matlab_variable.keys())[-1]
    return matlab_variable[matlab_variable_name]

def load_data():
    return \
        get_matlab_variable("src/data/reference.mat"), \
        get_matlab_variable("src/data/mask.mat"), \
        get_matlab_variable("src/data/lr_0_1.mat")