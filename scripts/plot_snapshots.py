import numpy as np
from tools import load_snapshot
import matplotlib.pyplot as plt
from collapse_analitical import compute_radius_evolution, get_radius

base_dir = '/home/bvillase/code/particle_in_cell/'


grav_const = 1
initial_density = 1
initial_radius = 0.25
analytical_time, analytical_radius = compute_radius_evolution( grav_const, initial_density, initial_radius )

for snap_id in range(50):
  file_name = base_dir + f'snapshots/snapshot_{snap_id}.h5'
  data = load_snapshot( file_name )
  time = data['time']
  density = data['density']
  nx, ny, nz = density.shape

  density_slice = density[nx//2, :, :]

  figure_name = base_dir + f'figures/fig_{snap_id}.png'

  fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12,10))
  cut = nz/2

  radius = get_radius( time, analytical_time, analytical_radius )
  print( f'Analytical radius: {radius} ')


  img = ax1.imshow( density_slice, extent=[0,1,0,1],  cmap='copper' )


  box = dict(facecolor = 'gray', alpha = 0.5, boxstyle='round' )
  ax1.text(0.24, 0.92, 'time: {0:.2f}'.format(time), c='white', fontsize=20, horizontalalignment='right', verticalalignment='center', transform=ax1.transAxes, bbox=box )
  
  circle = plt.Circle((0.5, 0.5), radius, color='white', fill=False, linewidth=1.5, ls='--')
  ax1.add_patch( circle )

  fig.savefig( figure_name, dpi=300, bbox_inches='tight' )
  fig.clf()

  print( f'Saved figure: {figure_name}')