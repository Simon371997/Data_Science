import numpy as np
import laspy
import h5py
import open3d as o3d
#import pptk

pc_ply = o3d.io.read_point_cloud("data_dropbox/fragment.ply")

print("Shape of Points: ", np.asarray(pc_ply.points).shape)
print("Shape of Colors: ", np.asarray(pc_ply.colors).shape)

#visualization
o3d.visualization.draw_geometries([pc_ply])