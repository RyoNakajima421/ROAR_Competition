"""Use surveyed corner clearances and a clean racing lap to correct an unsafe approach."""
from math import atan2, cos, hypot, remainder, sin, tau
from pathlib import Path
import json

import numpy as np


class FenceGuard:
    def __init__(self):
        here = Path(__file__).parent
        self.settings = json.loads((here/"guard_settings.json").read_text())
        with np.load(here/"guard_reference.npz", allow_pickle=False) as data:
            self.rows = data["rows"]
        self.index = 0

    def apply(self, driver, control):
        xy = driver.location_sensor.get_last_gym_observation()[:2]
        vx, vy = driver.velocity_sensor.get_last_gym_observation()[:2]
        speed = hypot(vx, vy)
        pos = self.rows[:, :2]
        if speed < 4.:
            self.index = max(0,min(len(pos)-1, driver.num_ticks-1))
            return control
        lo, hi = max(0, self.index-6), min(len(pos),self.index+18)
        gaps = pos[lo:hi]-xy
        i = lo + int(np.argmin(np.einsum("ij,ij->i",gaps,gaps)))
        self.index = i
        first = self.rows[i]
        if hypot(*(first[:2]-xy)) > 4. or i+15 >= len(pos):
            return control
        windows = self.settings.get("steering_windows")
        if windows and not any(start <= driver.current_waypoint_idx <= end for start,end in windows):
            return control
        vx_ref, vy_ref = first[2:4]
        course = atan2(vy_ref, vx_ref)
        error = -sin(course)*(xy[0]-first[0]) + cos(course)*(xy[1]-first[1])
        v_error = speed*sin(remainder(atan2(vy,vx) - course,tau))
        # Positive lateral error moves the car away from the right fence.
        dt = np.arange(4,16)*.05
        offset = error + np.clip(dt*v_error,-.9,.9)
        future = self.rows[i+4:i+16]
        right = np.min(future[:,4]+offset)
        left = np.min(future[:,5]-offset)
        margin = self.settings["margin"]
        deadband = self.settings.get("relative_deadband", 0.)
        # When enabled, the deadband allows for the reference car's own close passes.
        right_target = min(margin, np.min(future[:,4])-deadband) if deadband else margin
        left_target = min(margin, np.min(future[:,5])-deadband) if deadband else margin
        avoid_right = max(0., right_target-right)
        avoid_left = max(0., left_target-left)
        delta = np.clip(self.settings["gain"]*(avoid_left-avoid_right),-.035,.035)
        zone = next((row for row in self.settings.get("speed_envelopes", ())
                     if row[0] <= driver.current_waypoint_idx <= row[1]), None)
        if zone is not None:
            _, _, allowance = zone
            extra_kph = (speed-hypot(*first[2:4]))*3.6 - allowance
            braking = np.max(self.rows[max(0,i-4):i+6,6])
            if extra_kph > 0. and (braking > .6 or min(right,left)<margin):
                demand = min(1., max(float(control["brake"]), .32+.13*extra_kph))
                control = dict(control,brake=demand,throttle=min(.25,control["throttle"]))
            if min(right,left) < margin+.10:
                tracking = .024*error + .021*v_error
                delta = np.clip(delta+tracking,-.052,.052)
        updated = control
        if delta:
            updated = dict(control,steer=float(np.clip(control["steer"]+delta,-1.,1.)))
        if min(right,left) < margin and speed*3.6 > hypot(vx_ref,vy_ref)*3.6+1.25 and first[6] > .8:
            updated = dict(updated,throttle=min(updated["throttle"],first[7]),brake=max(updated["brake"],first[6]))
        return updated
