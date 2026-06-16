"""Make the simulator test path image"""
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

import matplotlib.pyplot as plt
import numpy as np




# Config
vehicle_list = ["veh_0", "veh_1", "veh_2"]
topic = "odom"

resolution = 0.06367
scale_x = 1.0
scale_y = 1.0
origin = [-40.0, -70.0]

# Map values
# start_x = 245
# start_y = 43
# crop_x1 = 130
# crop_x2 = 1032
# crop_y1 = 322
# crop_y2 = 1004

# Image values
start_x = 253
start_y = 60
crop_x1 = 120
crop_x2 = 975
crop_y1 = 300
crop_y2 = 1100


# Setup
desired_topics = {veh: f"/{veh}/{topic}" for veh in vehicle_list}

typestore = get_typestore(Stores.ROS2_FOXY)

image_path = Path("C:\\workspace\\convoy_sim\\offtrack_data\\data\\MBF_Image.png")
# image_path = Path("C:\\workspace\\convoy_sim\\offtrack_data\\data\\MBF_Map.png")
mbf_image = plt.imread(image_path)[crop_y1:crop_y2,crop_x1:crop_x2]

convoy_bagpath = Path('C:\\workspace\\convoy_sim\\offtrack_data\\data\\alyssa_test_convoy_1')
teleop_bagpath = Path('C:\\workspace\\convoy_sim\\offtrack_data\\data\\alyssa_test_teleop_1')


# =========== Helpers =========== #

def world_to_pixel(x, y, px0 = None, py0 = None):
    px = -(y - origin[1]) / resolution * scale_x
    py = -(x - origin[0]) / resolution * scale_y

    if px0 is None or py0 is None:
        px0 = px[0]
        py0 = py[0]

        px = px - px[0] + start_x
        py = py - py[0] + start_y

        return px, py, px0, py0

    else:
        px = px - px0 + start_x
        py = py - py0 + start_y

        return px, py



def get_data(path):
    vehicle_data = {veh: {} for veh in vehicle_list}
    start_time = None

    with AnyReader([path], default_typestore=typestore) as reader:
        for veh in vehicle_list:
            times = []
            xs = []
            ys = []

        
            connections = [x for x in reader.connections if x.topic == desired_topics[veh]]
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                msg = reader.deserialize(rawdata, connection.msgtype)

                time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                if start_time is None:
                    start_time = time

                times.append(time - start_time)
                xs.append(msg.pose.pose.position.x)
                ys.append(msg.pose.pose.position.y)

            vehicle_data[veh] = {
                "time": np.array(times),
                "x": np.array(xs),
                "y": np.array(ys),
            }

    return vehicle_data





# =========== Plotting  =========== #
vehicle_data_convoy = get_data(convoy_bagpath)
vehicle_data_teleop = get_data(teleop_bagpath)

max_time_convoy = max(vehicle_data_convoy[veh]["time"][-1] for veh in vehicle_list)
max_time_teleop = max(vehicle_data_teleop[veh]["time"][-1] for veh in vehicle_list)
max_time = max(max_time_convoy, max_time_teleop)

norm = plt.Normalize(vmin=0, vmax=max_time)
cmap = plt.cm.inferno

fig, axs = plt.subplots(1,2, figsize=(16,8))



# Convoy
ax = axs[0]
ax.imshow(mbf_image)
for veh in vehicle_list:
    if veh == "veh_0":
        x,y, veh_0_px0, veh_0_py0 = world_to_pixel(vehicle_data_convoy[veh]["x"], vehicle_data_convoy[veh]["y"])
    else:
        x,y = world_to_pixel(vehicle_data_convoy[veh]["x"], vehicle_data_convoy[veh]["y"], px0=veh_0_px0, py0=veh_0_py0)

    ax.scatter(x,y, c=vehicle_data_convoy[veh]["time"], cmap=cmap, norm=norm, s=3)

# Teleop
ax = axs[1]
ax.imshow(mbf_image)
for veh in vehicle_list:
    if veh == "veh_0":
        x,y, veh_0_px0, veh_0_py0 = world_to_pixel(vehicle_data_teleop[veh]["x"], vehicle_data_teleop[veh]["y"])
    else:
        x,y = world_to_pixel(vehicle_data_teleop[veh]["x"], vehicle_data_teleop[veh]["y"], px0=veh_0_px0, py0=veh_0_py0)

    sc = ax.scatter(x,y, c=vehicle_data_teleop[veh]["time"], cmap=cmap, norm=norm, s=3)

fig.colorbar(sc, ax=[axs[0], axs[1]], label="Elapsed time (s)", orientation='horizontal', location='bottom', shrink=0.8)

plt.show()