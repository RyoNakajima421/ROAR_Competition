import asyncio
import json
from math import atan2, cos, hypot, pi, remainder, sin, tau
from pathlib import Path

import numpy as np

from baseline_submission import RoarCompetitionSolution as BaseController
from fence_guard import FenceGuard
from lap_reference import LapReference
from ppo_core import CaptureVehicle, Features, NumpyPolicy, pedal_action


class RoarCompetitionSolution(BaseController):
    def __init__(self, waypoints, vehicle, *sensors, **kwargs):
        self.output_vehicle = vehicle
        self.baseline_action = CaptureVehicle(vehicle)
        self.route = Features(waypoints)
        self.pedal_policy = NumpyPolicy(Path(__file__).with_name("ppo_policy.npz"))
        self.reference = LapReference()
        self.fence_guard = FenceGuard()
        self.previous_phase = None
        self.s3_released = False
        self.s4_released = False
        super().__init__(waypoints, self.baseline_action, *sensors, **kwargs)
        self.waypoint_line.apply_widen([
            (2803, 2848, -.35), (2848, 2893, .35),
            (980, 1025, .20), (1025, 1070, -.20),
            (1740, 1785, .85), (1785, 1830, -.85),
        ])
        self.apply_s6_offsets()

    def apply_s6_offsets(self):
        line = self.waypoint_line.line_locations
        adjustments = json.loads(Path(__file__).with_name("s6_offsets.json").read_text(encoding="utf-8"))
        original = [point.copy() for point in line]
        for index, offset in adjustments:
            dx, dy = (original[index+1]-original[index-1])[:2]
            length = hypot(dx, dy)
            line[index][0] -= offset * dy / length
            line[index][1] += offset * dx / length

    def next_waypoint_smooth(self, current_speed, vehicle_location):
        target = super().next_waypoint_smooth(current_speed, vehicle_location)
        if self.current_section != 9:
            return target

        index = self.get_lookahead_index(current_speed)
        if not 2580 < index < 2610:
            return target

        # The squared taper also smooths the curvature at each end.
        rise = .5 * (1. - cos(2. * pi * (index - 2580) / 30.))
        offset = .35 * rise ** 2
        points = self.maneuverable_waypoints
        before, after = points[index - 1].location, points[index + 1].location
        dx, dy = after[:2] - before[:2]
        tangent_norm = hypot(float(dx), float(dy))
        location = target.location.copy()
        if tangent_norm:
            location[0] -= offset * float(dy) / tangent_norm
            location[1] += offset * float(dx) / tangent_norm
        return self.new_RoarPyWaypoint(location)

    def sector_pedals(self, base):
        phase = float(self.route.cumulative[self.route.index] / self.route.total)
        if self.previous_phase is not None and self.previous_phase - phase > .5:
            self.s3_released = False
            self.s4_released = False
        self.previous_phase = phase

        control = base
        if .308 <= phase <= .311 and not self.s3_released and float(base["brake"]) >= .85:
            vx, vy = self.velocity_sensor.get_last_gym_observation()[:2]
            if hypot(float(vx), float(vy)) * 3.6 <= 180.2:
                self.s3_released = True
                control = dict(base, brake=.10, throttle=1.)
        elif .908 < phase < .914 and self.current_waypoint_idx in (2381, 2382):
            if base["brake"] == 1. and base["throttle"] == 0.:
                control = dict(base, throttle=1., brake=0.)
        elif .489 <= phase <= .496 and 1271 <= self.current_waypoint_idx <= 1275:
            vx, vy = self.velocity_sensor.get_last_gym_observation()[:2]
            speed = 3.6 * hypot(vx, vy)
            gas, brake = float(base["throttle"]), float(base["brake"])
            if brake > .85:
                taper = min(1., max(0., (192.9 - speed) / 3.))
                if self.s4_released:
                    taper *= min(1., max(0., (188.8 - speed) / 2.))
                    strength = 0.7 * taper
                else:
                    strength = taper
                    if strength >= .25:
                        self.s4_released = True
                if strength > 0.:
                    control = dict(base, brake=brake + strength * (.40 - brake),
                                   throttle=gas + strength * max(0., .65 - gas))
        return control

    async def step(self):
        self.baseline_action.calls = 0
        self.baseline_action.action = None
        await super().step()
        if self.baseline_action.calls != 1:
            raise RuntimeError("Expected one baseline action")

        base = self.baseline_action.action
        obs = self.route.observe(
            self.location_sensor.get_last_gym_observation(),
            self.velocity_sensor.get_last_gym_observation(),
            self.rpy_sensor.get_last_gym_observation(),
            base,
        )
        residual = self.pedal_policy.predict(obs)
        self.route.previous_action = np.clip(residual.astype(np.float64), -1., 1.).astype(np.float32)
        control = self.sector_pedals(pedal_action(base, residual, obs))

        # Spread brake onset over 3.6 m, about one tick at entry speed.
        if 1229 <= self.current_waypoint_idx <= 1236:
            x, y = self.location_sensor.get_last_gym_observation()[:2]
            vx, vy = self.velocity_sensor.get_last_gym_observation()[:2]
            if 3.6 * hypot(vx, vy) >= 252.:
                distance = .728 * (82.88 - x) + .686 * (200.22 - y)
                ramp = min(1., max(0., (distance + 0.85) / 3.6))
                control = dict(control, brake=ramp, throttle=1. - ramp)
        index = self.current_waypoint_idx
        if 440 < index < 495 and control["brake"] < .03 and control["throttle"] >= .95:
            vx, vy = self.velocity_sensor.get_last_gym_observation()[:2]
            yaw = float(self.rpy_sensor.get_last_gym_observation()[2])
            slip = remainder(atan2(vy, vx) - yaw, tau)
            steer = float(control["steer"])
            if steer * slip > 0. and abs(slip) > .17:
                envelope = min(1., (index - 440.) / 5., (495. - index) / 7.)
                trim = min(abs(steer) * .40, 0.076 * (abs(slip) - .17) * envelope)
                control = dict(control, steer=steer - trim * (1. if steer > 0. else -1.))
        if 2534 <= self.current_waypoint_idx <= 2538 and control["brake"] < .90:
            _, y = self.location_sensor.get_last_gym_observation()[:2]
            vx, vy = self.velocity_sensor.get_last_gym_observation()[:2]
            speed = 3.6 * hypot(vx, vy)
            if 255.8 <= y <= 258.5 and 193.25 < speed <= 196.:
                position_weight = min(1., (y - 255.5) / 1.05)
                speed_weight = min(1., (speed - 193.25) / .5)
                brake = max(control["brake"], 0.88 * position_weight * speed_weight)
                control = dict(control, brake=brake, throttle=min(control["throttle"], 1. - brake))

        control = self.reference.control(self, control)
        control = self.fence_guard.apply(self, control)
        await self.output_vehicle.apply_action(control)
        # The actor API can return before CARLA applies the command.
        await asyncio.sleep(.002)
        return control
