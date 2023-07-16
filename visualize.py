import sys
import os
import math

import numpy as np
import matplotlib
import matplotlib.pylab as plt


src_folder = sys.argv[1] + "/raw_data/"
dst_folder = sys.argv[1] + "/render/"
os.makedirs(dst_folder, exist_ok=True)
data_type = np.float32
data_type_size = 4

blue = np.array([[0., 0., 1.]], data_type)
yellow = np.array([[1., 215/255, 0.]], data_type)

mask = None
count = 0

while True:
    count += 1
    file_path = os.path.join(src_folder, f"potential{count}.scalar")
    try:
        with open(file_path, "rb") as file:
            print("Generating image from", file_path)
            size = int.from_bytes(file.read(4), "little")

            buf = file.read(size**2 * data_type_size)
            ten = np.frombuffer(buf, dtype=data_type)
            ten = ten.reshape(size, size)  # in the order of y x
            mat = ten[::-1]  # flip y

            mat = np.array(mat)
            mask = np.isnan(mat)
            mat[mask] = 0.0

            mat = 0.5 * mat + 0.5 # convert the range to [0, 1]
            mat = mat.clip(0.0, 1.0)
            mat.resize(size, size, 1)
            mask.resize(size, size, 1)
            alpha = mat.copy()
            alpha.fill(1.0)
            alpha[mask] = 0.0
            rgb = mat * blue + (1. - mat) * yellow
            img = np.dstack((rgb, alpha)) # add alpha
            plt.imsave(os.path.join(dst_folder, f"potential{count}.png"), img)
    except FileNotFoundError:
        break

    # hard-coded visualizer for the boundary estimate example.
    file_path = os.path.join(src_folder, f"boundary_solution{count}.scalar")
    try:
        with open(file_path, "rb") as file:
            print("Generating image from", file_path)
            size = int.from_bytes(file.read(4), "little")
            buf = file.read(size * data_type_size)
            mat = np.frombuffer(buf, dtype=data_type)

            num_segments = 720
            x, y, s, c = [], [], [], []
            data =[]
            for i in range(num_segments):
                radius = 0.5
                v1 = np.array((
                    radius * math.sin((2 * math.pi) * i / num_segments), radius * math.cos((2 * math.pi) * i / num_segments)
                ))
                v2 = np.array((
                    radius * math.sin((2 * math.pi) * (i + 1) / num_segments), radius * math.cos((2 * math.pi) * (i + 1) / num_segments)
                ))
                center = (v1 + v2) / 2.
                value = 0.5 * mat[i] + 0.5 # convert the range to [0, 1]
                value = max(0.0, min(1.0, value))
                color = value * blue + (1. - value) * yellow
                color = matplotlib.colors.to_hex(color)
                x.append(center[0])
                y.append(center[1])
                s.append(1.)
                c.append(color)
                data += [(v1[0], v2[0]), (v1[1], v2[1]), color]

            fig = plt.figure()
            fig.set_figheight(10)
            fig.set_figwidth(10)
            ax = fig.add_subplot(111)
            axis_size = 0.52
            ax.set_xlim([-axis_size, axis_size])
            ax.set_ylim([-axis_size, axis_size])
            ax.plot(*data, linewidth=10)
            ax.set_aspect(1)
            ax.axis("off")
            plt.savefig(os.path.join(dst_folder, f"boundary{count}.png"), bbox_inches='tight', pad_inches=0)
    except FileNotFoundError:
        pass

        
