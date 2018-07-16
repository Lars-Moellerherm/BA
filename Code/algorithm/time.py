import functions as func
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py as h5
import scipy as sc
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import timeit


data = func.reading_data(True,-1)

number_tel = data[['array_event_id','run_id','num_triggered_telescopes']].drop_duplicates()
print("Teleskop anzahl im mittel : ",number_tel['num_triggered_telescopes'].mean())
