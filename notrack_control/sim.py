# A simulator for off-track convoy operations
import time
import math
from collections import deque
from enum import IntEnum

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

from vehicle import Vehicle
from ekf import EKF
from mpc import MPC

class LEADER_MODES(IntEnum):
    MANUAL = 0
    CIRCLE = 1
    RANDOM = 2


# Config
num_followers = 4
sit_time = 2

leader_mode = LEADER_MODES.CIRCLE
leader_delta = math.pi/12
leader_vel = 0.8

sim_dt = 0.1
arrow_len = 0.25
world_size = 2
max_past_poses = 10000

r_lidar_x = 0.05
r_lidar_y = 0.05
r_pozyx_r = 0.05

lidar_fov = math.pi * 60/180
do_visualize_lidar_fov = False

do_enable_crosstalk = True
do_debug_print = False

do_drop_lidar = False
do_drop_pozyx = False





# Setup
vehicles = {
    0: Vehicle(id=0, x=0, y=-1, theta=0, v=0),
    # 1: Vehicle(id=1, x=-0.5, y=-1, theta=0, v=0),
    # 2: Vehicle(id=2, x=-1, y=-1, theta=0, v=0),
}
for i in range(num_followers):
    k = i+1
    vehicles[k] = Vehicle(id=k, x=-0.5*k, y=-1, theta=0, v=0)

filters = {
    # 1: EKF(time=time.time(), veh_id=1, target_id=0),
    # 2: EKF(time=time.time(), veh_id=2, target_id=1),
}
for i in range(num_followers):
    k = i+1
    filters[k] = EKF(time=time.time(), veh_id=k, target_id=i, x0 = -0.5*k)

controllers = {
    # 1: MPC(),
    # 2: MPC(),
}
for i in range(num_followers):
    k = i+1
    controllers[k] = MPC()

inputs = {
    0: {'delta': 0.0, 'vel': 0.0}, 
    # 1: {'delta': 0.0, 'vel': 0.0},  
    # 2: {'delta': 0.0, 'vel': 0.0},  
}
for i in range(num_followers):
    k = i+1
    inputs[k] = {'delta': 0.0, 'vel': 0.0}



# More Setup
nprandom = np.random.default_rng(seed=2025)

colors = {0: "#0DFF86", 1: "#FF520D", 2: "#00FFD5", 3: "#D9FF00", 4: "#F705F7", 5: "#00CE0A"}
plt.ion()
fig, ax = plt.subplots(figsize=(8,8))
fig.tight_layout()

trails = {id: deque(maxlen=10000) for id in vehicles}





leader_manual = {'delta': 0.0, 'vel': 0.0}

def on_key_press(event):
    if event.key == 'up':
        leader_manual['vel'] = min(leader_manual['vel'] + 0.1, 1.0)
    elif event.key == 'down':
        leader_manual['vel'] = max(leader_manual['vel'] - 0.1, 0.0)
    elif event.key == 'left':
        leader_manual['delta'] = min(leader_manual['delta'] + math.pi/12, math.pi/4)
    elif event.key == 'right':
        leader_manual['delta'] = max(leader_manual['delta'] - math.pi/12, -math.pi/4)

fig.canvas.mpl_connect('key_press_event', on_key_press)







# Helpers

def get_lidar_meas(veh_id_center, veh_id_target) -> tuple:
    """Generate a noisy lidar centroid"""
    if do_drop_lidar:
        return None, None

    veh_target = vehicles[veh_id_target]
    veh_center = vehicles[veh_id_center]

    dx = veh_target.x - veh_center.x
    dy = veh_target.y - veh_center.y

    # Check if target is within FOV before adding noise
    bearing = math.atan2(dy, dx) - veh_center.theta
    bearing = math.atan2(math.sin(bearing), math.cos(bearing))  # wrap to [-pi, pi]

    if abs(bearing) > lidar_fov / 2:
        return None, None

    dx += nprandom.normal(0, r_lidar_x)
    dy += nprandom.normal(0, r_lidar_y)

    # Rotate into follower body frame
    c, s = math.cos(-veh_center.theta), math.sin(-veh_center.theta)
    x_meas = c*dx - s*dy
    y_meas = s*dx + c*dy

    return x_meas, y_meas

def get_pozyx_meas(veh_id_center, veh_id_target) -> float:
    """Generate a noisy lidar centroid"""
    if do_drop_pozyx:
        return None
    
    veh_target = vehicles[veh_id_target]
    veh_center = vehicles[veh_id_center]

    dx = veh_target.x - veh_center.x
    dy = veh_target.y - veh_center.y
    r = math.sqrt(dx**2 + dy**2) + nprandom.normal(0,r_pozyx_r)
    return r




def get_leader_inputs(t):
    period = 1
    t = t % period
    
    # if t < 10:
    #     return {'delta': math.pi/6, 'vel': 0.5}
    # elif t < 20:
    #     return {'delta': math.pi/6, 'vel': 0.5}

    if leader_mode == LEADER_MODES.CIRCLE:
        return {'delta': leader_delta, 'vel': leader_vel}
    elif leader_mode == LEADER_MODES.RANDOM and t < 0.2:
        return {'delta': nprandom.uniform(math.pi/12, math.pi/4), 'vel': nprandom.uniform(0,0.8)}


# Main loop

try:
    start_time = time.time()
    last_time = start_time


    while True:
        current_time = time.time()
        current_dt = current_time - last_time

        if current_dt > sim_dt:
            last_time = current_time

            # State Estimate Updates
            for ekf in filters.values():
                if ekf is not None:
                    if do_enable_crosstalk:
                        delta, vel = inputs[ekf.veh_id].values()
                        delta_t, vel_t = inputs[ekf.target_id].values()
                    else: 
                        delta = None
                        vel = None
                        delta_t = None
                        vel_t = None
                    
                    # Lidar update
                    x_meas_lidar, y_meas_lidar = get_lidar_meas(ekf.veh_id, ekf.target_id)
                    ekf.update_lidar(x_meas_lidar, y_meas_lidar, current_time, delta=delta, vel=vel, delta_t=delta_t, vel_t=vel_t)

                    # POZYX update
                    r_meas_pozyx = get_pozyx_meas(ekf.veh_id, ekf.target_id)
                    ekf.update_pozyx(r_meas_pozyx, current_time, delta=delta, vel=vel, delta_t=delta_t, vel_t=vel_t)

                    # Final predict update
                    ekf.update_predict(current_time, delta=delta, vel=vel, delta_t=delta_t, vel_t=vel_t)


            # Control inputs
            
            ## Leader control
            if leader_mode == LEADER_MODES.MANUAL:
                inputs[0] = leader_manual
            else:
                new_input = get_leader_inputs(current_time - start_time)
                if new_input is not None:
                    inputs[0] = new_input

            ## Controller Updates
            for id,mpc in controllers.items():
                if mpc is not None:
                    x_targ, y_targ, theta_targ, vel_targ = filters[id].x
                    vel, delta = mpc.solve(x_targ, y_targ, theta_targ)
                    inputs[id] = {"delta": delta, "vel": vel}

            if current_time - start_time < sit_time:
                    for id,input in inputs.items():
                        inputs[id] = {"delta": 0, "vel": 0.005}
            

            # True dynamics
            for veh in vehicles.values():
                delta, vel = inputs[veh.id].values()
                veh.step(vel, delta, current_dt)

            # Record poses
            for id, veh in vehicles.items():
                trails[id].append((veh.x, veh.y))
            


            # Plotting
            ax.clear()

            for id, trail in trails.items():
                if len(trail) > 1:
                    xs, ys = zip(*trail)
                    ax.plot(xs, ys, linestyle='-', alpha=0.3, color=colors[id])

            for veh in vehicles.values():
                # Lidar FOV
                if do_visualize_lidar_fov:
                    wedge = Wedge((veh.x, veh.y), 2.5,
                        math.degrees(veh.theta - lidar_fov/2),
                        math.degrees(veh.theta + lidar_fov/2),
                        color=colors[veh.id], alpha=0.05)
                    ax.add_patch(wedge)

                # Vehicle
                ax.plot(veh.x, veh.y, marker='o', linestyle="", color=colors[veh.id], label=f"veh {veh.id}")
                ax.annotate("", xytext=(veh.x, veh.y), xy=(veh.x+arrow_len*math.cos(veh.theta), veh.y+arrow_len*math.sin(veh.theta)), arrowprops=dict(arrowstyle="->"))

            for ekf in filters.values():
                veh = vehicles[ekf.veh_id]
                c, s = math.cos(veh.theta), math.sin(veh.theta)
                ex_world = veh.x + c*ekf.x[0] - s*ekf.x[1]
                ey_world = veh.y + s*ekf.x[0] + c*ekf.x[1]
                theta_world = veh.theta + ekf.x[2]
                ax.plot(ex_world, ey_world, marker='+', linestyle="", color=colors[veh.id], label=f"veh {ekf.target_id} from veh {ekf.veh_id}")
                ax.annotate("", xytext=(ex_world, ey_world), xy=(ex_world+arrow_len*math.cos(theta_world), ey_world+arrow_len*math.sin(theta_world)), arrowprops=dict(arrowstyle="->", color=colors[veh.id]))

            for id,mpc in controllers.items():
                veh = vehicles[id]
                c, s = math.cos(veh.theta), math.sin(veh.theta)
                x,y,theta = mpc.target
                ex_world = veh.x + c*x - s*y
                ey_world = veh.y + s*x + c*y
                theta_world = veh.theta + theta
                ax.plot(ex_world, ey_world, marker='x', linestyle="", color=colors[veh.id], label=f"veh {id} target pt")
                # ax.annotate("", xytext=(ex_world, ey_world), xy=(ex_world+arrow_len*math.cos(theta_world), ey_world+arrow_len*math.sin(theta_world)), arrowprops=dict(arrowstyle="->"))

            ax.set_xlim(-world_size, world_size)
            ax.set_ylim(-world_size, world_size)
            ax.set_title("Simulator")
            ax.legend()
            plt.pause(0.001)

            if not plt.get_fignums():
                break

            if do_debug_print:
                
                print("VEHICLES")
                print(f"time: id | true x | true y | true theta | true v")
                for id, veh in vehicles.items():
                    print(f"{current_time}s: {id} | {veh.x} | {veh.y} | {veh.theta} | {veh.v}")

                print("\nFILTERS")
                print(f"time: id | ekf x | ekf y | ekf theta | ekf v")
                for id, ekf in filters.items():
                    print(f"{current_time}s: {id} | {ekf.x[0]} | {ekf.x[1]} | {ekf.x[2]} | {ekf.x[3]}")
                          
                


except KeyboardInterrupt:
    plt.close()

except Exception as e:
    raise

finally:
    print("Shutting down sim")