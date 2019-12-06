import os
import shutil
from collections import deque
import numpy as np
import time
import yaml
from plyfile import PlyData, PlyElement
from generate_sequential import parse_calibration, parse_poses

# Define paths
input_dir = os.path.abspath("P:/datasets/kitti/odometry/dataset")
output_dir = os.path.abspath("O:/akurz/maplearning_groundtruth_generation/semantic_kitti")
sequence = "07"
input_folder = os.path.join(input_dir, "sequences", sequence)
output_folder = os.path.join(output_dir, "sequence" + sequence)

if os.path.exists(output_folder):
    print("Output folder '{}' already exists!".format(output_folder))
    answer = input("Overwrite? [y/N] ")
    if answer != "y":
        print("Aborted.")
        exit(1)
else:
    os.makedirs(output_folder)

shutil.copy(os.path.join(input_folder, "poses.txt"), output_folder)
shutil.copy(os.path.join(input_folder, "calib.txt"), output_folder)

scan_files = [
    f for f in sorted(os.listdir(os.path.join(input_folder, "velodyne")))
    if f.endswith(".bin")
]

# Load config file
DATA = yaml.safe_load(open("config/semantic-kitti.yaml", 'r'))
color_map = DATA["color_map"]

# Initialize variables
start_time = time.time()

history = deque()

calibration = parse_calibration(os.path.join(input_folder, "calib.txt"))
poses = parse_poses(os.path.join(input_folder, "poses.txt"), calibration)

progress = 10

print("Processing Sequence {} ...".format(sequence), end="\n", flush=True)

file_counter = 1
concated_points = []
concated_labels = []
for i, f in enumerate(scan_files):
    # read scan and labels, get pose
    scan_filename = os.path.join(input_folder, "velodyne", f)
    scan = np.fromfile(scan_filename, dtype=np.float32)

    scan = scan.reshape((-1, 4))

    label_filename = os.path.join(input_folder, "labels", os.path.splitext(f)[0] + ".label")
    labels = np.fromfile(label_filename, dtype=np.uint32)
    labels = labels.reshape((-1))

    # convert points to homogenous coordinates (x, y, z, 1)
    points = np.ones((scan.shape))
    points[:, 0:3] = scan[:, 0:3]
    remissions = scan[:, 3]

    pose = poses[i]

    # Transform scan to global coordinate system
    tpoints = np.matmul(pose, points.T).T

    concated_points = np.append(concated_points, tpoints)
    concated_points = concated_points.reshape((-1, 4))
    concated_labels = np.append(concated_labels, labels)
    concated_labels.reshape((-1, 1))


    if i % int(len(scan_files)/10) == 0 and i > 0:
        print("Batch %d/%d" % (file_counter, 10))
        progress = progress + 10

        # Write points and labels to .ply
        vertex = []
        label_color = np.array([np.array(color_map.get(label, [0, 0, 0])) for label in concated_labels], dtype=np.ubyte)
        data = np.concatenate((concated_points[:, :3], label_color), axis=1)
        for row in data:
            vertex.append(tuple(row))
        vertex = np.array(vertex,
                          dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        el = PlyElement.describe(vertex, 'vertex')
        output_filename = "fused" + str(file_counter) + ".ply"
        PlyData([el]).write(os.path.join(output_folder, output_filename))
        file_counter += 1

        # Delete history
        concated_points = []
        concated_labels = []

print("Finished.")

print("Execution Time: {}".format(time.time() - start_time) + "s")