import os
import matplotlib
matplotlib.use('TkAgg')
import math
import numpy as np
from numpy.linalg import inv
import pandas as pd
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt

def parse_calibration(filename):
    """ read calibration file with given filename

        Returns
        -------
        dict
            Calibration matrices as 4x4 numpy arrays.
    """
    calib = {}

    calib_file = open(filename)
    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        calib[key] = pose

    calib_file.close()

    return calib


def parse_poses(filename, calibration):
    """ read poses file with per-scan poses from given filename

        Returns
        -------
        list
            list of poses as 4x4 numpy arrays.
    """
    file = open(filename)

    poses = []

    Tr = calibration["Tr"]
    Tr_inv = inv(Tr)

    for line in file:
        values = [float(v) for v in line.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

    return poses


plydata = PlyData.read('C:/Users/EGGT7P6/Data/Kitti/map/sequences/07/fused1_mesh_reduced_1_8.ply')
input_folder = os.path.abspath("C:/Users/EGGT7P6/Data/Kitti/dataset/sequences/07")
image = "000007.png"
image_path = os.path.join(input_folder, "image_0", image)
calibration = parse_calibration(os.path.join(input_folder, "calib.txt"))
poses = parse_poses(os.path.join(input_folder, "poses.txt"), calibration)
pose = poses[7]
pose_inv =  inv(pose)
#R = np.matrix(pose[:3, :3])
#t = np.matrix(pose[:3, 3]).T
T = calibration["Tr"][:3, :]
A = np.matrix(calibration["P0"][:3, :3])
pose_x = pose[0, 3]

img = plt.imread(image_path)
width = np.shape(img)[1]
height = np.shape(img)[0]
print(width, height)

#Filter pointcloud
idx = [i for i, x in enumerate(plydata['vertex']['x']) if x > pose_x]
vertices = np.hstack((np.array(plydata['vertex'][idx].tolist()), np.array(idx)[:, None]))
faces = np.stack(plydata['face']['vertex_indices'])

check = np.isin(faces, idx).nonzero()[0]
rows = np.unique(check)
extracted_faces = faces[rows]

print(len(vertices))


projected_points = []

#Point3D = np.matrix([10.214251, 5.711651, 0.994958, 1]).T
for i in range(len(vertices)):
    if i % 100000 == 0:
        print(i)
    Point3D = np.matrix([vertices[i,0], vertices[i,1], vertices[i,2], 1]).T
    tpoint = pose_inv * Point3D
    r = math.sqrt(tpoint[0]**2 + tpoint[1]**2 + tpoint[2]**2)
    lidar2camera = T * tpoint
    result = A * lidar2camera
    result = result / result[2]
    Point2DRGBi = np.array([result.item(0), result.item(1), r, vertices[i,3], vertices[i,4], vertices[i,5], vertices[i,7]])
    projected_points.append(Point2DRGBi)

points = pd.DataFrame(projected_points, columns=["x", "y", "range", "red", "green", "blue", "index"])
points.to_csv('projected_points_reduced.txt', sep="\t", index=False)
print(points.index)

faces = pd.DataFrame(extracted_faces, columns=["index1", "index2", "index3"])
faces.to_csv('extracted_faces_reduced.txt', sep="\t", index=False)
print(faces.index)