import os
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_points = pd.read_csv('projected_points_reduced.txt', sep="\t")
points = df_points.values
df_faces = pd.read_csv('extracted_faces_reduced.txt', sep="\t")
faces = df_faces.values
input_folder = os.path.abspath("C:/Users/EGGT7P6/Data/Kitti/dataset/sequences/07")
image = "000007.png"
image_path = os.path.join(input_folder, "image_0", image)
img = plt.imread(image_path)
width = np.shape(img)[1]
height = np.shape(img)[0]

idx = []
for i, point in enumerate(points):
    if 0 <= point[0] <= width and 0 <= point[1] <= height:
        idx.append(i)

print(len(points))
print(len(idx))
plotpoints = points[idx]
indices = np.array(plotpoints[:,6], dtype=int)
#print(plotpoints[:,3:6])
range = plotpoints[:, 2]
range = (range - np.min(range)) / (np.max(range) - np.min(range))
plotpoints[:, 2] = range

# Generate depth image
depth_img = np.ones(np.shape(img)) * 1000
for i, point in enumerate(plotpoints):
    if point[2] < depth_img[int(point[1]), int(point[0])]:
        depth_img[int(point[1]), int(point[0])] = point[2]

depth_y, depth_x = np.where(depth_img != 1000)
depth_values = depth_img[depth_y, depth_x]

plt.imshow(img, cmap='gray')
#plt.imshow(depth_img)
plt.scatter(x=depth_x, y=depth_y, s=0.5, c=depth_values)
#plt.scatter(x=plotpoints[:,0], y=plotpoints[:,1], s=0.1, c=range, cmap='hsv')

plt.show()
