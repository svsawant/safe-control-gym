

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
x_slice_at = 0.0

load_individual = False
# load_individual = True
# load_all = False
load_all = True
# ROA = 'small'
# ROA = 'large'
ROA = 'true'

current_file_path = os.path.abspath(os.path.dirname(__file__))

if load_all:
    # fig_name = '/ROA_data/ROA_3D_slice_x_{:0.2f}.pickle'\
    #                                     .format(x_slice_at)
    fig_name = '/ROA_data/ROA_3D_wo_first.pickle'
    abs_fig_name = current_file_path + fig_name
    figx = pickle.load(open(abs_fig_name, 'rb'))
    figx.show() # Show the figure, edit it, etc.!

elif load_individual:
    fig_name = '/ROA_data/ROA_{}_wo_first.pickle'.format(ROA)
    abs_fig_name = current_file_path + fig_name
    figx = pickle.load(open(abs_fig_name, 'rb'))
    figx.show() # Show the figure, edit it, etc.!

input("Press Enter to continue...") # Wait for input to close the figure