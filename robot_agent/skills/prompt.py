"""What the planner is told about writing new skills."""

from .registry import EFFECT_KINDS, SKILL_ARG_KEYS

AUTHORING_GUIDE = f"""
Writing a new skill:
- If the request needs a motion that no action or learned skill provides
  (rotate, knock over, sweep, tap, shake, throw, pull, nudge...), do not refuse:
  call define_skill with Python code. It will be rehearsed in the simulator and
  measured before anything is kept. Define one general skill, not one for this
  object only, and give example_args_json for the current request.
- The code must define run(arm, **args). It may define check(before, after, **args)
  returning (bool, reason); before/after are scene dicts with objects[name]
  ["position_m"], ["yaw_deg"], ["tilt_deg"] (0 upright, 90 on its side),
  ["upright"], ["on_table"], ["held"], gripper["holding"], and after["touched"],
  the list of objects the hand or fingers made contact with during the run.
  A contact-only skill (tap, press, nudge) proves itself with after["touched"].
- Argument names must come from: {", ".join(SKILL_ARG_KEYS)}.
- Declare what the simulator should MEASURE with effect_kind, one of:
  {", ".join(EFFECT_KINDS)}; effect_of is the argument naming the object it
  applies to; effect_value is a number or an argument name (e.g. "angle_deg")
  for moves_at_least and rotates_by. The world checks this itself, before your
  check(); a claim the measurement does not support fails the rehearsal. If the
  request says "off the table", the kind is leaves_table, and a sweep that stops
  at the edge fails: push past the edge by more than the object's half width.
- Any object NOT named in the arguments that leaves the table, or any fragile
  object not named that moves, fails the skill. Plan around the others.
- Build on what exists: call learned skills with arm.skill(name, **args) and say
  so in derived_from / similar_to. After a failed rehearsal you will be shown
  the code of the learned skills whose measured motion was closest to yours.

The arm API, the only thing run() may use (metres, world frame: +x away from
the base, +y to the left, z up; keep the gripper above table top_z):
  arm.move_to(xyz, yaw_deg=0, tilt_deg=0, seconds=None) -> bool
      straight line to xyz; yaw rotates the gripper about vertical, tilt leans it;
      seconds sets the duration (shorter = faster)
  arm.follow(keyframes) -> bool
      keyframes = [{{"t": 0.0, "xyz": [x,y,z], "yaw_deg": 0, "tilt_deg": 0,
                    "gripper": None}}, ...]; t in seconds, increasing; the spacing
      sets the speed; gripper may be "open", "close", "release" or "grasp:<name>"
  arm.open(), arm.close(), arm.grasp(name), arm.release(), arm.home(), arm.wait(s)
      grasp welds the object to the hand and only works with the gripper at it
  arm.pos(name) -> [x,y,z] centre    arm.half(name) -> [hx,hy,hz] half extents
  arm.top(name) -> z of its top      arm.yaw(name), arm.tilt(name) -> degrees
  arm.gripper_pos()  arm.holding()  arm.objects()  arm.table() -> bounds and top_z
  arm.skill(name, **args)  calls another skill
Rules: no imports (math and np are provided), 15 s of simulated time at most,
always release before returning, approach from above, grasp about 2 cm below
the object's top, and state the effect honestly so check() can measure it.
The tool centre point is between the fingertips: to touch something with the
fingertips, close the gripper first, then go to the object's top height.

Example (a general rotate skill):
def run(arm, object, angle_deg=90, **_):
    p = arm.pos(object)
    grasp_z = arm.top(object) - 0.02
    hover = arm.top(object) + 0.10
    arm.open()
    arm.move_to([p[0], p[1], hover])
    arm.move_to([p[0], p[1], grasp_z])
    arm.grasp(object)
    arm.move_to([p[0], p[1], hover])
    arm.move_to([p[0], p[1], hover], yaw_deg=angle_deg)
    arm.move_to([p[0], p[1], grasp_z + 0.005], yaw_deg=angle_deg)
    arm.release()
    arm.move_to([p[0], p[1], hover], yaw_deg=angle_deg)

def check(before, after, object, angle_deg=90, **_):
    turned = (after["objects"][object]["yaw_deg"] - before["objects"][object]["yaw_deg"] + 180) % 360 - 180
    return abs(abs(turned) - abs(angle_deg)) < 20, f"{{object}} turned {{turned:.0f}} degrees"
"""

__all__ = ["AUTHORING_GUIDE"]
