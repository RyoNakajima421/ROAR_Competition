# ROAR Fall 2026 Monza test entry

Autonomous driving agent for Berkeley Monza v1.1. The entry point is
[`competition_code/submission.py`](competition_code/submission.py).

## Run locally

Set up [ROAR_PY and CARLA 0.9.12](https://roar.gitbook.io/roar-competition-documentation/installation)
and start the Monza v1.1 server on localhost, port 2000. In the competition Python
environment, run the following from the repository root.

```shell
cd competition_code
python competition_runner.py
```

The agent uses NumPy from the standard competition environment. The policy weights,
waypoints, and reference data are included. The official evaluator and infrastructure
files are unmodified.

## Controller

The base controller uses pure pursuit for steering and track curvature for speed
planning. A small PPO policy adjusts throttle and brake in a limited part of the
lap. The entry also adjusts steering in the first chicane, blends brake commands
by position and speed in S4 and at the final hairpin, and uses the measured pose
and velocity to correct departures from the reference line.

The agent waits 2 ms after sending each control command through the public actor
API. In local testing this kept control delivery aligned with the evaluator's
50 ms simulation steps.

## Local test result

The controller completed eight three-lap races in **318.80 s** each with zero
contacts. Four races followed CARLA restarts and four followed another race on
the same server. The splits were **110.90 s**, **104.15 s**, and **103.75 s**.
These are local measurements using the official evaluator.

## Credits

The base controller and racing line come from the
[HKIS Racing Summer 2026 entry](https://github.com/micahc123/roar2026),
which builds on the
[University High School controller](https://github.com/MightyMark3/ROAR_1_S25).
The evaluator comes from the
[ROAR competition starter repository](https://github.com/augcog/ROAR_Competition).
