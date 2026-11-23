"""Route observations and PPO pedal inference."""
import copy
import math
import numpy as np

OBS_SIZE = 26
PREVIEW_METERS = (10., 20., 40., 60., 100., 160.)
PEDAL_WINDOW = (.42, .75)
PEDAL_RAMP = .025
class CaptureVehicle:
    def __init__(self, vehicle):
        self.wrapped = vehicle
        self.action = None
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    async def apply_action(self, action):
        self.action = copy.deepcopy(action)
        self.calls += 1
        return True


def pedal_window_weight(obs):
    phase = (math.atan2(float(obs[12]), float(obs[13])) / (2 * math.pi)) % 1.
    distance = min(phase - PEDAL_WINDOW[0], PEDAL_WINDOW[1] - phase)
    ramp = max(0., min(1., distance / PEDAL_RAMP))
    return .5 - .5 * math.cos(math.pi * ramp)


def pedal_action(base, action, obs):
    action = np.asarray(action, dtype=np.float64).reshape(2)
    if not np.isfinite(action).all():
        raise ValueError("Non-finite policy action")
    action = np.clip(action, -1., 1.)
    strength = pedal_window_weight(obs) * .02
    control = base.copy()
    for key, amount in zip(("throttle", "brake"), action):
        if strength and amount:
            control[key] = float(np.clip(float(base[key]) + strength * amount, 0., 1.))
    return control


class Features:
    """Observations expected by the exported pedal policy."""
    def __init__(self, waypoints):
        self.points = np.asarray([w.location[:2] for w in waypoints], dtype=np.float64)
        if len(self.points) < 40:
            raise ValueError("Too few waypoints")
        self.delta = np.roll(self.points, -1, axis=0) - self.points
        self.lengths = np.linalg.norm(self.delta, axis=1)
        self.cumulative = np.r_[0., np.cumsum(self.lengths)]
        self.total = float(self.cumulative[-1])
        if self.total < 100:
            raise ValueError("Invalid track length")
        self.index = None
        self.last_yaw = None
        self.previous_action = np.zeros(2, dtype=np.float32)

    def observe(self, location, velocity, rpy, base, dt=.05):
        p = np.asarray(location, dtype=np.float64)[:2]
        v = np.asarray(velocity, dtype=np.float64)[:2]
        yaw = float(np.asarray(rpy).reshape(-1)[2])
        n = len(self.points)
        if self.index is None:
            self.index = int(np.argmin(np.sum((self.points-p)**2, axis=1)))
        else:
            candidates = (self.index + np.arange(-5, 31)) % n
            self.index = int(candidates[np.argmin(np.sum((self.points[candidates]-p)**2, axis=1))])
        i = self.index
        tangent = self.delta[i] / max(float(self.lengths[i]), 1e-6)
        normal = np.array([-tangent[1], tangent[0]])
        error = p-self.points[i]
        lateral = float(np.dot(error, normal))
        route_yaw = math.atan2(tangent[1], tangent[0])
        heading = yaw-route_yaw
        yaw_rate = 0. if self.last_yaw is None else math.atan2(math.sin(yaw-self.last_yaw), math.cos(yaw-self.last_yaw))/dt
        self.last_yaw = yaw
        forward = np.array([math.cos(yaw), math.sin(yaw)])
        left = np.array([-math.sin(yaw), math.cos(yaw)])
        phase = 2*math.pi*self.cumulative[i]/self.total
        values = [np.dot(v, forward)/80., np.dot(v, left)/30., np.linalg.norm(v)/80.,
                  lateral/10., math.sin(heading), math.cos(heading), yaw_rate/3.,
                  float(base["steer"]), float(base["throttle"]), float(base["brake"]),
                  *self.previous_action, math.sin(phase), math.cos(phase)]
        s = self.cumulative[i] + np.clip(np.dot(error, tangent), 0., self.lengths[i])
        for distance in PREVIEW_METERS:
            target = float((s+distance) % self.total)
            j = min(int(np.searchsorted(self.cumulative, target, side="right")-1), n-1)
            alpha = (target-self.cumulative[j])/max(float(self.lengths[j]), 1e-6)
            q = self.points[j] + alpha*self.delta[j] - p
            values.extend([np.dot(q, forward)/100., np.dot(q, left)/100.])
        obs = np.asarray(values, dtype=np.float32)
        if obs.shape != (OBS_SIZE,) or not np.isfinite(obs).all():
            raise ValueError("Invalid observation")
        return np.clip(obs, -10., 10.)


class NumpyPolicy:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.layers = [(data["w0"], data["b0"]), (data["w1"], data["b1"])]
            self.head = (data["w2"], data["b2"])

    def predict(self, obs):
        x = np.asarray(obs, dtype=np.float32)
        for w, b in self.layers:
            x = np.tanh(w @ x + b)
        w, b = self.head
        return np.clip(w @ x + b, -1., 1.).astype(np.float32)
