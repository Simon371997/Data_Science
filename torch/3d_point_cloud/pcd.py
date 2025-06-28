import numpy as np
import laspy
import h5py
import open3d as o3d

pc_pcd = o3d.io.read_point_cloud("data_dropbox/pcl_CSite1_orig-utm.pcd")
print(pc_pcd)

print("Shape of Points: ", np.asarray(pc_pcd.points).shape)
print("Shape of Colors: ", np.asarray(pc_pcd.colors).shape) # no color data

o3d.visualization.draw_geometries([pc_pcd])