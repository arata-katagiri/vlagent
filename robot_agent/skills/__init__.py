"""Three layers above the joints.

Layer 1  body.py      the motion layer: poses, trajectories, gripper, perception
Layer 2  registry.py  skills: task-level functions, built-in or written by the model
Layer 3  the planner  sequences of skills for one request (agent/planner.py)

rehearse.py runs a skill in a snapshot of the world and measures what happened.
sandbox.py  compiles model-written code with no imports and no file or network access.
"""
