"""Track a previously driven three-lap trajectory with pose and speed feedback."""
from math import atan2, cos, hypot, remainder, sin, tau
from pathlib import Path
import json

import numpy as np


class LapReference:
    def __init__(self):
        root = Path(__file__).parent
        with np.load(root/"lap_reference.npz", allow_pickle=False) as source:
            self.rows = source["trajectory"]
        self.xy = self.rows[:, :2]
        self.segment = np.diff(self.xy, axis=0)
        self.squared_length = np.maximum(np.sum(self.segment**2, axis=1), 1.e-9)
        self.tick = 0
        self.index = 0
        self.settings = json.loads((root/"reference_settings.json").read_text())

    def control(self, driver, stock):
        self.tick += 1
        xy = np.asarray(driver.location_sensor.get_last_gym_observation()[:2])
        vx, vy = driver.velocity_sensor.get_last_gym_observation()[:2]
        yaw = driver.rpy_sensor.get_last_gym_observation()[2]
        speed = hypot(vx, vy)
        if self.tick < 30:
            self.index = self.tick - 1
            return stock
        start, end = max(0, self.index-4), min(len(self.segment), self.index+16)
        rel = xy-self.xy[start:end]
        fraction = np.clip(np.einsum("ij,ij->i", rel, self.segment[start:end]) /
                           self.squared_length[start:end], 0., 1.)
        gaps = rel - fraction[:, None]*self.segment[start:end]
        local = int(np.argmin(np.einsum("ij,ij->i", gaps, gaps)))
        i, f = start+local, float(fraction[local])
        self.index = i
        if np.linalg.norm(gaps[local]) > 4.:
            return stock
        weight = 1.
        if "corners" in self.settings:
            index = driver.current_waypoint_idx
            weight = max((min(1., (index-lo)/5., (hi-index)/5.)
                          for lo, hi in self.settings["corners"] if lo < index < hi), default=0.)
            if weight == 0.:
                return stock
        a, b = self.rows[i], self.rows[i+1]
        vref = a[3:5] + f*(b[3:5]-a[3:5])
        course = atan2(vref[1], vref[0])
        heading = a[2] + f*remainder(b[2]-a[2], tau)
        lateral = -sin(course)*gaps[local, 0]+cos(course)*gaps[local, 1]
        yaw_error = remainder(yaw-heading, tau)
        course_error = remainder(atan2(vy, vx)-course, tau)
        # Take pedal values from one sample so sharp transitions stay discrete.
        ref = a if f < .5 else b
        steer, gas, brake = ref[5:8]
        lateral_gain, heading_gain, course_gain = self.settings.get("gains", (.032,.43,.23))
        for lo, hi, gains in self.settings.get("corner_gains", ()):
            if lo <= driver.current_waypoint_idx <= hi:
                lateral_gain, heading_gain, course_gain = gains
                break
        if "blend" in self.settings:
            weight *= self.settings["blend"]
        steer += self.settings["feedback"]*(lateral_gain*lateral+heading_gain*yaw_error+course_gain*course_error)
        extra = 0.
        for first, last, kph in self.settings["speed_windows"]:
            if first < driver.current_waypoint_idx < last:
                extra = kph
                break
        speed_error = (hypot(*vref)-speed)*3.6+extra
        if abs(speed_error) > .45 and (brake > .01 or gas < .99 or speed_error < -3.):
            gas = np.clip(gas + .16*speed_error, 0., 1.)
            brake = np.clip(brake - .12*speed_error, 0., 1.)
        if self.settings.get("stock_pedals") and not extra:
            gas, brake = stock["throttle"], stock["brake"]
        return dict(stock, steer=float(np.clip((1.-weight)*stock["steer"] + weight*steer,-1.,1.)),
                    throttle=float((1.-weight)*stock["throttle"]+weight*gas),
                    brake=float((1.-weight)*stock["brake"]+weight*brake))
