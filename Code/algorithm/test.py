import pandas as pd
import functions as func
import h5py as h5


data = func.reading_data(False, 100)

data = func.calc_scaled_width_and_length(data)

print(data)
