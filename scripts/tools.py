import numpy as np

def load_snapshot( file_name ):
  import h5py as h5
  print( f'Loading file: {file_name}')
  file = h5.File( file_name, 'r' )
  time = file.attrs['time'][0]
  density = file['density'][...]
  return { 'time':time, 'density':density }
