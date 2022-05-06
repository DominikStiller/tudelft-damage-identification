import numpy as np
import pandas as pd
import matplotlib.pyplot as pl



def force_displacement(filename):
    '''
    plots the force against the displacement
    '''

    data = pd.read_csv(f"../../data/{filename}", delimiter='\t', encoding='utf-8', header=2)
    data = data.drop(index = 0)

    data = data.rename(columns={'250 kN MTS Force':'force', 'Running Time':'runtime', 'Time':'time', '250 kN MTS Displacement':'displacement'})
    for column in data.columns:
        data[column] = [x.replace(',', '.') for x in data[column]]
        data[column] = data[column].astype(float)
    data['force'] = -data['force']
    data['displacement'] = -data['displacement']
    pl.plot(data['displacement'], data['force'])
    pl.xlabel('Displacement [mm]')
    pl.ylabel('Force [kN]')
    pl.title()
    pl.show()


force_displacement("specimen_comp90.dat")