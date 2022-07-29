import os
import numpy as np
from tqdm.auto import tqdm
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
import re
import argparse
import sqlite3


def rot3to4(rotation: np.ndarray):
    assert rotation.shape == (3, 3)
    result = np.eye(4)
    result[:3, :3] = rotation.copy()
    return result


def write_cameras(path):
    fullpath = os.path.join(path, "cameras.txt")

    to_write_lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        "# Number of cameras: 1",
        "1 SIMPLE_RADIAL 624 574 1000.0 312 287 1.0",
    ]

    with open(fullpath, "w") as fout:
        for line in to_write_lines: fout.write(line + "\n")


def write_images(path, db, N=100):
    fullpath = os.path.join(path, "images.txt")
    cursor = db.cursor()

    to_write_lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        "# Number of images: ???, mean observations per image: ???"
    ]

    # Fetch database names
    db_list = cursor.execute("SELECT image_id, name FROM images")
    sorted_db_list = sorted(db_list, key=lambda elem: elem[0])
    assert [elem[0] for elem in sorted_db_list] == [*range(1, len(sorted_db_list) + 1)]

    for i, IMAGE in sorted_db_list:
        theta = 2 * np.pi * i / N

        TX = np.sin(theta)
        TY = np.cos(theta)
        TZ = 0.0
        T = np.array([TX, TY, TZ])

        # Creating a quaternion that rotates around a given point
        # according to https://www.euclideanspace.com/maths/geometry/affine/aroundPoint/index.htm
        # Better finding: https://github.com/colmap/colmap/issues/434

        if theta != 0.0:

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

        else:
            quat = Quaternion(axis=[1., 0., 0.], radians=np.pi / 2)

        R = Rotation.from_matrix(quat.rotation_matrix)

        transform_quat = Quaternion(matrix=R.as_matrix().T)
        T = -R.as_matrix().T.dot(T)
        TX, TY, TZ = T

        QW, QX, QY, QZ = transform_quat  # QW, QX, QY, QZ

        to_write_lines.append(" ".join(
            [str(i), str(QW), str(QX), str(QY), str(QZ), str(TX), str(TY), str(TZ),
             str(1), IMAGE]) + "\n")

        cursor.execute(f"""UPDATE images
                           SET prior_qw = {QW}, prior_qx = {QX}, prior_qy = {QY}, prior_qz = {QZ},
                           prior_tx = {TX}, prior_ty = {TY}, prior_tz = {TZ}
                           WHERE image_id == {i};""")

        conn.commit()

    with open(fullpath, "w") as fout:
        for line in to_write_lines: fout.write(line + "\n")


def write_points3d(path):
    fullpath = os.path.join(path, "points3D.txt")

    to_write_lines = ["# 3D point list with one line of data per point:",
                      "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
                      "# Number of points: ???, mean track length: ???"
                      ]

    with open(fullpath, "w") as fout:
        for line in to_write_lines: fout.write(line + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pose generator')
    parser.add_argument('path', type=str, help='Path to generate cameras.txt, images.txt, points3D.txt')
    parser.add_argument('--database_path', type=str, help='Path to the database, optionally '
                                                          'inferred from the path if possible')
    args = parser.parse_args()

    os.makedirs(args.path, exist_ok=True)

    if args.database_path is not None:
        conn = sqlite3.connect(args.database_path)
    else:
        db_path = os.path.join(os.path.dirname(os.path.dirname(args.path)), "database.db")
        conn = sqlite3.connect(db_path)

    write_cameras(args.path)
    write_images(args.path, conn, N=100)
    write_points3d(args.path)
