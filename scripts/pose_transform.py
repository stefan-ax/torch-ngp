import os
import numpy as np
from tqdm.auto import tqdm
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import pdist, squareform
import re
import argparse

basepath = "data/venus-rough-1-positions-tests/colmap_text"


def rot3to4(rotation: np.ndarray):
    assert rotation.shape == (3, 3)
    result = np.eye(4)
    result[:3, :3] = rotation.copy()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pose transform (circle approximator)')
    parser.add_argument('path', type=str, help='Path to images.txt')
    parser.add_argument('--constant_axis', type=str, help="Axis to keep constant. Default: y", default='y')
    # Remember axes colors: x-red, y-green, z-blue

    args = parser.parse_args()

    assert os.path.isfile(args.path)

    to_write_lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        "# Number of images: ???, mean observations per image: ???"
    ]

    new_lines = []

    with open(args.path, "r") as fin:
        for line in fin.readlines()[0::2]:
            if line[0] != "#":
                IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME = line.split()

                new_lines.append([IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME])

    os.rename(args.path, args.path[:-4] + '_old' + '.txt')

    # Sort the lines by id
    sorted_lines = sorted(new_lines, key=lambda line: int(re.findall(r'\d+', line[-1])[0]))

    # Get the diameter of the circle by the mean of all of them
    if args.constant_axis == 'y':
        TX_TZ = []
        for IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME in sorted_lines:
            TX_TZ.append([float(TX), float(TZ)])

        TX_TZ = np.array(TX_TZ)
        distances = squareform(pdist(TX_TZ))
        diameters = distances.max(axis=1)
        mean_diameter = np.mean(diameters)
        mean_radius = mean_diameter / 2
    else:
        raise NotImplementedError('Sorry, coming soon')

    # Fix coordinates
    N = len(sorted_lines)
    for i in tqdm(range(N)):
        IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME = sorted_lines[i]
        theta = 2 * np.pi * i / N

        NEW_TX = mean_radius * np.sin(theta)
        NEW_TY = 0.0
        NEW_TZ = mean_radius * np.cos(theta)
        NEW_T = np.array([NEW_TX, NEW_TY, NEW_TZ])

        # Creating a quaternion that rotates around a given point
        # according to https://www.euclideanspace.com/maths/geometry/affine/aroundPoint/index.htm
        # Better finding: https://github.com/colmap/colmap/issues/434

        norm_array = lambda l: np.array(l) / np.linalg.norm(np.array(l))

        if theta != 0:
            if args.constant_axis == 'z':
                if np.cos(theta) >= 0:
                    adjusted_theta = - theta
                else:
                    adjusted_theta = theta + np.pi

                quat = Quaternion(axis=[np.sign(np.cos(theta)),
                                        0.0,
                                        0.0],
                                  angle=np.pi / 2) * \
                       Quaternion(axis=[0.0,
                                        1.0,
                                        0.0],
                                  angle=adjusted_theta) * \
                       Quaternion(axis=[0.0,
                                        0.0,
                                        1.0],
                                  angle=np.pi / 2)
            elif args.constant_axis == 'y':
                if np.sin(theta) >= 0:
                    adjusted_theta = np.pi - theta
                elif np.sin(theta) < 0:
                    adjusted_theta = - np.pi + theta

                quat = Quaternion(axis=[0.0,
                                        -np.sign(np.sin(theta)),
                                        0.0],
                                  angle=adjusted_theta)

        else:
            quat = Quaternion(axis=[0., 1., 0.], radians=np.pi)

        R = Rotation.from_matrix(quat.rotation_matrix)

        new_transform_quat = Quaternion(matrix=R.as_matrix().T)
        NEW_T = -R.as_matrix().T.dot(NEW_T)
        NEW_TX, NEW_TY, NEW_TZ = NEW_T

        NEW_QW, NEW_QX, NEW_QY, NEW_QZ = new_transform_quat  # QW, QX, QY, QZ

        to_write_lines.append(" ".join(
            [IMAGE_ID, str(NEW_QW), str(NEW_QX), str(NEW_QY), str(NEW_QZ), str(NEW_TX), str(NEW_TY), str(NEW_TZ),
             CAMERA_ID, NAME]) + "\n")
        # to_write_lines.append("\n")

    with open(args.path, "w") as fout:
        for line in to_write_lines: fout.write(line + "\n")
