import open3d as o3d
import numpy as np


pcd = o3d.io.read_point_cloud("data_dropbox/bunny.pcd")
print(np.asarray(pcd.points).shape)
print(np.asarray(pcd.colors).shape)

o3d.visualization.draw_geometries([pcd])
