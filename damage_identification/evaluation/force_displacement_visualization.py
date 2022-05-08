import numpy as np
import pandas as pd
import matplotlib.pyplot as pl



def force_displacement(filename):
    '''
    plots the force against the displacement
    '''

    data = pd.read_csv(f"../../data/{filename}", delimiter='\t', encoding='utf-8', header=2, error_bad_lines=False)
    data = data.drop(index = data.index[data['Time'].str.isalpha()])

    colreplacement = {'.* kN MTS Force':'force', 'Running Time':'runtime', 'Time':'time', '.* MTS Displacement':'displacement'}
    data.columns = data.columns.to_series().replace(colreplacement, regex=True)

    for column in data.columns:
        data[column] = [x.replace(',', '.') for x in data[column]]
        data[column] = data[column].astype(float)

    data['force'], data['displacement'] = -data['force'], -data['displacement']

    pl.plot(data['displacement'], data['force'])
    pl.xlabel('Displacement [mm]')
    pl.ylabel('Force [kN]')
    pl.title('Placeholder')
    pl.show()
    #pl.savefig()
    print('done')

force_displacement("specimen.dat")