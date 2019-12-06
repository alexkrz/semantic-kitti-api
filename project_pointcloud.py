import os
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from numpy.linalg import inv
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt
from generate_sequential import parse_calibration, parse_poses

# Define paths
ply_path = os.path.abspath("C:/Users/EGGT7P6/Data/Kitti/map/sequences/07/fused1.ply")
input_folder = os.path.abspath("C:/Users/EGGT7P6/Data/Kitti/dataset/sequences/07")
image = "000007.png"
image_path = os.path.join(input_folder, "image_0", image)

# Read and process pointcloud
plydata = PlyData.read(ply_path)
vertices = np.matrix(plydata['vertex'].data.tolist())
pc_points = vertices[:, :3]
pc_points = np.hstack((pc_points, np.ones((len(pc_points), 1))))
pc_colors = vertices[:, 3:6]

# Get Lidar pose and camera calibration
calibration = parse_calibration(os.path.join(input_folder, "calib.txt"))
poses = parse_poses(os.path.join(input_folder, "poses.txt"), calibration)
pose = poses[7]
Tr_Global2Lidar = inv(pose)
print(Tr_Global2Lidar)
Tr_Lidar2Camera = calibration["Tr"]
print(Tr_Lidar2Camera)
A = np.matrix(calibration["P0"][:3, :3])
pose_x = pose[0, 3]

# Transform points into camera coordinate frame
tr_points = (Tr_Lidar2Camera * (Tr_Global2Lidar * pc_points.T)).T
print(len(tr_points))
idx = [i for i in range(len(tr_points[:, 2])) if tr_points[i, 2] > 0.5] # Filter all points that are in front of the camera
tr_points = tr_points[idx]
print((len(tr_points)))
tr_colors = pc_colors[idx]

# Read image
img = plt.imread(image_path)
width = np.shape(img)[1]
height = np.shape(img)[0]

# Project points into image
result = A * tr_points[:, :3].T
result = result / result[2]
result = result.T
result = np.hstack((result[:, :2], tr_points[:, 2]))
result = np.hstack((result, tr_colors))
idx = []
for i in range(len(result)):
    point = result[i]
    if 0 <= point[0, 0] <= width and 0 <= point[0, 1] <= height:  # Filter all points that lie inside the image
        idx.append(i)
image_points = result[idx]
image_points = np.array(image_points)

# Generate depth image
depth_img = np.ones(np.shape(img)) * 1000
for i in range(len(image_points)):
    point = image_points[i]
    if point[2] < depth_img[int(point[1]), int(point[0])]:
        depth_img[int(point[1]), int(point[0])] = point[2]
depth_y, depth_x = np.where(depth_img != 1000)
depth_values = depth_img[depth_y, depth_x]

# Extract semantically labelled points that are in the foreground of the image
idx = [i for i in range(len(image_points))
       if depth_img[int(image_points[i, 1]), int(image_points[i, 0])] - 0.5 <= image_points[i, 2] <=
       depth_img[int(image_points[i, 1]), int(image_points[i, 0])] + 0.5]
image_points = image_points[idx]

# Plot image
plt.imshow(img, cmap='gray')
plt.scatter(x=image_points[:, 0], y=image_points[:, 1], s=0.5, c=image_points[:,3:]/255.0)
plt.show()