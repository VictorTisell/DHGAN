import tensorflow as tf
import pandas as pd
import numpy as np
import quantlib as ql
import datetime
from utils import Option_data


datafolder_path = 'datafolder'
filename = 'quotedata_SPX.csv'
filepath = datafolder_path + '/' + filename
data = pd.read_csv(filepath, sep = ';')
calendar = ql.UnitedStates()
Ks = [3000, 3100, 3200, 3300, 3400]
ttms = [0, 10, 21, 31, 62]
settings = {'start': datetime.date(2020,9,11),
            'day_count': ql.Actual365Fixed(),
            'calendar': ql.UnitedStates(),
            'strikes': Ks,
            'maturities':ttms}
options = Option_data(data, settings)
print(options.price_array)
