import numpy as np
from tools import load_snapshot
import matplotlib.pyplot as plt

base_dir = '/home/bvillase/code/particle_mesh/'

file_0 = base_dir + 'snapshots/snapshot_0.h5'
data_0 = load_snapshot( file_0 )
density_0 = data_0['density']

file_1 = base_dir + 'snapshots/snapshot_0_sorted.h5'
data_1 = load_snapshot( file_1 )
density_1 = data_1['density']


diff = np.abs( density_1 - density_0 )
indices = np.where( density_0 > 0)
diff[indices] /= density_0[indices]

