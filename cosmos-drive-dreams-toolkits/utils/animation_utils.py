"""
Copyright 2022 by Jiahui Huang. All rights reserved.
This file is part of PyCG toolbox and is released under "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

from enum import Enum

import numpy as np
from copy import deepcopy
from pycg.isometry import Isometry, Quaternion
from collections import defaultdict


"""
Guidance to add a new animated attribute:
    - In SceneAnimator: add setter/getter of animators. add how the attributes should be applied in set_frame.
    - In render._update_XXX_engine: add how the viewer should re-load the new attributes.
    - In render.keyframer.callbacks: add how information from the scene should be loaded.
"""


def quaternion_uniform_flip(q_list):
    """
    Flip the directions of quaternions to prevent from not taking the nearest path.
    """
    new_q_list = deepcopy(q_list)
    if len(q_list) < 2:
        return new_q_list

    last_q = new_q_list[0]
    for qi in range(1, len(new_q_list)):
        if np.dot(new_q_list[qi].q, last_q.q) < 0:
            new_q_list[qi].q = -new_q_list[qi].q
        last_q = new_q_list[qi]

    return new_q_list


class BaseInterpolator:
    def __init__(self):
        self._keyframes = {}
        self._precomputed = False       # For some interpolators this is needed.

    def set_keyframe(self, t, value):
        self._keyframes[t] = value
        self._precomputed = False

    def remove_keyframe(self, t):
        del self._keyframes[t]
        self._precomputed = False

    def ordered_keyframes(self):
        kf = list(self._keyframes.items())
        return sorted(kf, key=lambda t: t[0])

    def ordered_times(self):
        return sorted(list(self._keyframes.keys()))

    def get_value(self, t):
        raise NotImplementedError

    def get_first_t(self):
        if len(self._keyframes) == 0:
            return None
        return min(list(self._keyframes.keys()))

    def get_last_t(self):
        if len(self._keyframes) == 0:
            return None
        return max(list(self._keyframes.keys()))

    def send_blender(self, uuidx: str, data_path: str, index: int, negate: bool = False):
        raise NotImplementedError


class ConstantInterpolator(BaseInterpolator):
    """
    Values are taken as the first predecessor. If not exist, then use the first successor.
        - USD: Held interpolation type
        - Blender: Constant keyframe
    """
    def get_value(self, t):
        all_times = self.ordered_times()
        if t < all_times[0]:
            return self._keyframes[all_times[0]]
        if t >= all_times[-1]:
            return self._keyframes[all_times[-1]]
        nki = np.searchsorted(all_times, t, side='right')
        return self._keyframes[all_times[nki - 1]]


class LinearInterpolator(BaseInterpolator):
    """
    Values are linearly interpolated from nearby keyframes.
        - USD: Linear interpolation
        - Blender: Linear keyframe
    """
    def get_value(self, t):
        all_times = self.ordered_times()
        if t < all_times[0]:
            return self._keyframes[all_times[0]]
        if t >= all_times[-1]:
            return self._keyframes[all_times[-1]]

        nki = np.searchsorted(all_times, t, side='right')
        dt = all_times[nki] - all_times[nki - 1]

        alpha = (t - all_times[nki - 1]) / dt
        prev_value = self._keyframes[all_times[nki - 1]]
        next_value = self._keyframes[all_times[nki]]
        return (1 - alpha) * prev_value + alpha * next_value
    

class LinearQuaternionInterpolator(BaseInterpolator):
    """
    Specially designed for quaternions.
        - USD: Slerp interpolation as shown:
            https://graphics.pixar.com/usd/release/api/interpolation_8h.html#USD_LINEAR_INTERPOLATION_TYPES
        - Blender: does not support quaternion slerp yet.
    """
    def set_keyframe(self, t, value):
        assert isinstance(value, Quaternion)
        super().set_keyframe(t, value)

    def precompute(self):
        kfs = self.ordered_keyframes()
        new_qs = quaternion_uniform_flip([t[1] for t in kfs])
        for new_q, (time, _) in zip(new_qs, kfs):
            self._keyframes[time] = new_q
        self._precomputed = True

    def get_value(self, t):
        if not self._precomputed:
            self.precompute()

        all_times = self.ordered_times()
        if t < all_times[0]:
            return self._keyframes[all_times[0]]
        if t >= all_times[-1]:
            return self._keyframes[all_times[-1]]
        nki = np.searchsorted(all_times, t, side='right')
        dt = all_times[nki] - all_times[nki - 1]

        alpha = (t - all_times[nki - 1]) / dt
        prev_q = self._keyframes[all_times[nki - 1]]
        next_q = self._keyframes[all_times[nki]]
        return Quaternion.slerp(prev_q, next_q, alpha)


class BezierInterpolator(BaseInterpolator):
    """
    Cubic bezier curve (https://en.wikipedia.org/wiki/B%C3%A9zier_curve), used nearly everywhere in DCC apps!
    Given P0 P1 P2 P3, where P1 and P2 are (half-)handles of P0 and P3, then the curve goes:
        B(t) = lerp{ lerp[ lerp(P0,P1), lerp(P1,P2) ], lerp[ lerp(P1,P2), lerp(P2,P3) ] }  -- 6 lerps needed!
    where
        lerp(Px, Py) = (1-t) Px + t Py.
    this is equivalent to:
        B(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t)t^2 P2 + t^3 P3      0 <= t <= 1

    - Blender: Auto-clamped keyframes, with 'Handle Smoothing' set to 'None'.
        See: blender source code: source/blender/blenkernel/intern/curve.cc - calchandleNurb_intern.
        (The continuous acceleration model is hard to implement here because it involves solving a linear equation)
    - USD: not supported.
    """
    class BezierTriplet:
        def __init__(self):
            # vec[0], vec[1], vec[2] are 2D coords of left-handle, middle, right-handle.
            self.vec = np.zeros((3, 2))

    def __init__(self):
        super().__init__()
        self._bezier_triplets = []

    def precompute(self):
        self._bezier_triplets = []

        kfs = self.ordered_keyframes()
        for (prev_t, prev_value), (cur_t, cur_value), (next_t, next_value) in zip(
                [(None, None)] + kfs[:-1], kfs, kfs[1:] + [(None, None)]):
            cur_triplet = BezierInterpolator.BezierTriplet()
            self._bezier_triplets.append(cur_triplet)

            p2 = np.array([cur_t, cur_value])
            cur_triplet.vec[1] = p2

            if prev_t is not None and next_t is not None:
                p1 = np.array([prev_t, prev_value])
                p3 = np.array([next_t, next_value])
                len_a = cur_t - prev_t
                len_b = next_t - cur_t
                # Average slope with x length = 2.
                tvec = (p3 - p2) / len_b + (p2 - p1) / len_a
                # Never make the handle lengths too different
                len_a = min(len_a, 5.0 * len_b)
                len_b = min(len_b, 5.0 * len_a)
                cur_triplet.vec[0] = p2 - (tvec / (2 * 2.5614)) * len_a
                cur_triplet.vec[2] = p2 + (tvec / (2 * 2.5614)) * len_b
            else:
                if prev_t is None and next_t is None:
                    delta_t = 1.0
                elif prev_t is None:
                    delta_t = (next_t - cur_t) / 2.5614
                elif next_t is None:
                    delta_t = (cur_t - prev_t) / 2.5614
                cur_triplet.vec[0] = [cur_t - delta_t, cur_value]
                cur_triplet.vec[2] = [cur_t + delta_t, cur_value]
                continue

            ydiff1 = prev_value - cur_value
            ydiff2 = next_value - cur_value
            left_violate, right_violate = False, False
            if (ydiff1 <= 0.0 and ydiff2 <= 0.0) or (ydiff1 >= 0.0 and ydiff2 >= 0.0):
                # Clamp at extrema.
                cur_triplet.vec[0][1] = cur_value
                cur_triplet.vec[2][1] = cur_value
            else:
                # Left handle exceeds previous kf's value...
                if (ydiff1 <= 0.0 and prev_value > cur_triplet.vec[0][1]) or \
                        (ydiff1 > 0.0 and prev_value < cur_triplet.vec[0][1]):
                    cur_triplet.vec[0][1] = prev_value
                    left_violate = True
                # Right handle exceeds next kf's value...
                if (ydiff1 <= 0.0 and next_value < cur_triplet.vec[2][1]) or \
                        (ydiff1 > 0.0 and next_value > cur_triplet.vec[2][1]):
                    cur_triplet.vec[2][1] = next_value
                    right_violate = True
            h1_x = cur_triplet.vec[0][0] - p2[0]
            h2_x = p2[0] - cur_triplet.vec[2][0]
            if left_violate:        # <-- take left's slope if both are violated
                cur_triplet.vec[2][1] = p2[1] + ((p2[1] - cur_triplet.vec[0][1]) / h1_x) * h2_x
            elif right_violate:
                cur_triplet.vec[0][1] = p2[1] + ((p2[1] - cur_triplet.vec[2][1]) / h2_x) * h1_x

        self._precomputed = True

    def visualize(self):
        if not self._precomputed:
            self.precompute()

        import matplotlib.pyplot as plt
        xr = np.linspace(0.0, 1.0, 50)[None, :]
        all_times = self.ordered_times()
        for nki in range(len(all_times) - 1):
            p0, p1 = self._bezier_triplets[nki].vec[1], self._bezier_triplets[nki].vec[2]
            p2, p3 = self._bezier_triplets[nki + 1].vec[0], self._bezier_triplets[nki + 1].vec[1]
            p = ((1 - xr) ** 3) * p0[:, None] + (3 * (1 - xr) ** 2 * xr) * p1[:, None] + \
                (3 * (1 - xr) * xr ** 2) * p2[:, None] + (xr ** 3) * p3[:, None]
            plt.plot(p[0], p[1])
        plt.show()

    def get_value(self, t):
        if not self._precomputed:
            self.precompute()

        all_times = self.ordered_times()
        if t < all_times[0]:
            return self._keyframes[all_times[0]]
        if t >= all_times[-1]:
            return self._keyframes[all_times[-1]]
        nki = np.searchsorted(all_times, t, side='right')

        p0 = self._bezier_triplets[nki - 1].vec[1]
        p1 = self._bezier_triplets[nki - 1].vec[2]
        p2 = self._bezier_triplets[nki].vec[0]
        p3 = self._bezier_triplets[nki].vec[1]

        o3t = -p0 + 3 * p1 - 3 * p2 + p3
        o2t = 3 * p0 - 6 * p1 + 3 * p2
        o1t = -3 * p0 + 3 * p1
        o0t = p0 - t

        from scipy.optimize import fsolve
        f = lambda x: o3t[0] * x ** 3 + o2t[0] * x ** 2 + o1t[0] * x + o0t[0]
        fy = lambda x: o3t[1] * x ** 3 + o2t[1] * x ** 2 + o1t[1] * x + o0t[1] + t
        fprime = lambda x: 3 * o3t[0] * x ** 2 + 2 * o2t[0] * x + o1t[0]
        t_target = fsolve(f, [0.5], fprime=fprime)[0]

        return fy(t_target)


class SquadQuaternionInterpolator(BaseInterpolator):
    """
    A spline [Shoemake, 1985] that evaluates faster than Bezier with fewer slerps:
    This is used in QGLViewer:
        B(t) = slerp[ slerp(P0, P3, t), slerp(P1, P2, t), 2t(1-t) ]  -- 3 slerps needed.
    Note that this is not equivalent to BezierQuaternionCurve, see '../examples/anime.py' for more details.
    Also: https://devtalk.blender.org/t/quaternion-interpolation/15883/15
    BezierQuaternionCurve is not implemented -- the control points are hard to choose!

    Neither supported in blender nor USD, but should be the best choice to interpolate arbitrary poses.
    """
    class KeyFrame:
        def __init__(self):
            self.iso = Quaternion()
            self.tangent = Quaternion()

    def __init__(self):
        super().__init__()
        self._q_keyframes = []

    def precompute(self):
        self._q_keyframes = []

        kfs = self.ordered_keyframes()
        new_qs = quaternion_uniform_flip([t[1] for t in kfs])
        for new_q, (time, _) in zip(new_qs, kfs):
            self._keyframes[time] = new_q
        all_times = self.ordered_times()
        for ki, ti in enumerate(all_times):
            q_prev = self._keyframes[all_times[max(ki - 1, 0)]]
            q_next = self._keyframes[all_times[min(ki + 1, len(all_times) - 1)]]
            cur_q_kf = SquadQuaternionInterpolator.KeyFrame()
            cur_q_kf.iso = self._keyframes[ti]
            cur_q_kf.tangent = Isometry(q=cur_q_kf.iso).tangent(
                Isometry(q=q_prev), Isometry(q=q_next)).q
            self._q_keyframes.append(cur_q_kf)

        self._precomputed = True

    def get_value(self, t):
        if not self._precomputed:
            self.precompute()

        all_times = self.ordered_times()
        if t < all_times[0]:
            return self._keyframes[all_times[0]]
        if t >= all_times[-1]:
            return self._keyframes[all_times[-1]]
        nki = np.searchsorted(all_times, t, side='right')

        dt = all_times[nki] - all_times[nki - 1]
        alpha = (t - all_times[nki - 1]) / dt

        prev_kf = self._q_keyframes[nki - 1]
        next_kf = self._q_keyframes[nki]

        qab = Quaternion.slerp(prev_kf.iso, next_kf.iso, alpha)
        qtgt = Quaternion.slerp(prev_kf.tangent, next_kf.tangent, alpha)
        q = Quaternion.slerp(qab, qtgt, 2.0 * alpha * (1.0 - alpha))

        return q


class InterpType(Enum):
    CONSTANT = 0
    LINEAR = 1
    BEZIER = 2


class AnimatorBase:
    """
    Base class for animators.

    animator is a collection of interpolators for each component. for example,
    position has x, y, z interpolators, rotation has qx, qy, qz, qw interpolators.
    """
    def __init__(self):
        self.interpolators = {}
        # For better export-ability
        self.keyframes = {}
        self._precomputed = False

    def add_interpolator(self, name, inst):
        assert name not in self.interpolators.keys()
        self.interpolators[name] = inst
        return self.interpolators[name]

    def set_keyframe(self, t, value):
        raise NotImplementedError

    def remove_keyframe(self, t):
        self._precomputed = False
        for itp in self.interpolators.values():
            itp.remove_keyframe(t)

    def ordered_times(self):
        all_times = sum([itp.ordered_times() for itp in self.interpolators.values()], [])
        return sorted(set(all_times))

    def get_first_t(self):
        """Get the first time of the animation by checking all interpolators.

        Returns:
            First time of the animation, or None if no times are available
        """
        first_t_list = [itp.get_first_t() for itp in self.interpolators.values()]
        first_t_list = [t for t in first_t_list if t is not None]
        return min(first_t_list) if len(first_t_list) > 0 else None

    def get_last_t(self):
        """Get the last time of the animation by checking all interpolators.

        Returns:
            Last time of the animation, or None if no times are available
        """
        last_t_list = [itp.get_last_t() for itp in self.interpolators.values()]
        last_t_list = [t for t in last_t_list if t is not None]
        return max(last_t_list) if len(last_t_list) > 0 else None

    def get_value(self, t, raw: bool = False):
        raise NotImplementedError


class ScalarAnimator(AnimatorBase):
    def __init__(self, interp_type: InterpType, blender_attribute: str = "scale", blender_attribute_count: int = 3):
        super().__init__()
        self.interp_type = interp_type
        self.blender_attribute = blender_attribute
        self.blender_attribute_count = blender_attribute_count
        if self.interp_type == InterpType.LINEAR:
            self.interp = self.add_interpolator('value', LinearInterpolator())
        elif self.interp_type == InterpType.BEZIER:
            self.interp = self.add_interpolator('value', BezierInterpolator())
        else:
            self.interp = self.add_interpolator('value', ConstantInterpolator())

    def set_keyframe(self, t, value):
        self.interp.set_keyframe(t, value)

    def get_value(self, t, raw: bool = False):
        return self.interp.get_value(t)
    

class FreePoseAnimator(AnimatorBase):
    """A class to animate pose (position and rotation) of objects in a scene.

    This class handles interpolation of both position and rotation components of an Isometry pose.
    For rotation, it supports two interpolation modes:
        - BLENDER: Interpolates quaternion components separately and renormalizes (matches Blender behavior)
        - MANIFOLD: Interpolates quaternions directly on the manifold for better linearity

    Args:
        interp_type: Type of interpolation to use (LINEAR, BEZIER)
        rotation_type: Type of rotation interpolation (BLENDER or MANIFOLD)
    """

    class RotationType(Enum):
        BLENDER = 0     # separate 4 components of quaternion and re-normalize them afterwards.
        MANIFOLD = 1    # interpolate on manifold, will give better linearity / quadratic-ity

    def __init__(self, interp_type: InterpType, rotation_type: RotationType = RotationType.BLENDER):
        super().__init__()
        self.interp_type = interp_type
        self.rotation_type = rotation_type
        # Create position interpolators
        if self.interp_type == InterpType.LINEAR:
            self.interp_x = self.add_interpolator('x', LinearInterpolator())
            self.interp_y = self.add_interpolator('y', LinearInterpolator())
            self.interp_z = self.add_interpolator('z', LinearInterpolator())
            # Create rotation interpolators based on rotation type
            if self.rotation_type == FreePoseAnimator.RotationType.BLENDER:
                self.interp_qw = self.add_interpolator('qw', LinearInterpolator())
                self.interp_qx = self.add_interpolator('qx', LinearInterpolator())
                self.interp_qy = self.add_interpolator('qy', LinearInterpolator())
                self.interp_qz = self.add_interpolator('qz', LinearInterpolator())
            else:
                self.interp_q = self.add_interpolator('q', LinearQuaternionInterpolator())
        elif self.interp_type == InterpType.BEZIER:
            self.interp_x = self.add_interpolator('x', BezierInterpolator())
            self.interp_y = self.add_interpolator('y', BezierInterpolator())
            self.interp_z = self.add_interpolator('z', BezierInterpolator())
            # Create rotation interpolators based on rotation type
            if self.rotation_type == FreePoseAnimator.RotationType.BLENDER:
                self.interp_qw = self.add_interpolator('qw', BezierInterpolator())
                self.interp_qx = self.add_interpolator('qx', BezierInterpolator())
                self.interp_qy = self.add_interpolator('qy', BezierInterpolator())
                self.interp_qz = self.add_interpolator('qz', BezierInterpolator())
            else:
                self.interp_q = self.add_interpolator('q', SquadQuaternionInterpolator())

    def set_keyframe(self, t, value: Isometry):
        """Set a keyframe for the pose at time t.

        Args:
            t: Time of the keyframe
            value: Isometry pose to set at this keyframe
        """
        assert isinstance(value, Isometry)
        self._precomputed = False

        # Set position keyframes
        self.interp_x.set_keyframe(t, value.t[0])
        self.interp_y.set_keyframe(t, value.t[1])
        self.interp_z.set_keyframe(t, value.t[2])

        # Set rotation keyframes based on rotation type
        if self.rotation_type == FreePoseAnimator.RotationType.BLENDER:
            self.interp_qw.set_keyframe(t, value.q.q[0])
            self.interp_qx.set_keyframe(t, value.q.q[1])
            self.interp_qy.set_keyframe(t, value.q.q[2])
            self.interp_qz.set_keyframe(t, value.q.q[3])
        else:
            self.interp_q.set_keyframe(t, value.q)

    def precompute(self):
        """Precompute quaternion flips to ensure shortest path interpolation.
        
        This is only needed for BLENDER rotation type.
        """
        self._precomputed = True
        if self.rotation_type == FreePoseAnimator.RotationType.BLENDER:
            # Flip quaternions to avoid non-closest rotation.
            kf_qw, kf_qx = self.interp_qw.ordered_keyframes(), self.interp_qx.ordered_keyframes(),
            kf_qy, kf_qz = self.interp_qy.ordered_keyframes(), self.interp_qz.ordered_keyframes()
            q_list = [Quaternion([
                kf_qw[t][1], kf_qx[t][1], kf_qy[t][1], kf_qz[t][1]
            ]) for t in range(len(kf_qw))]
            new_qs = quaternion_uniform_flip(q_list)
            for time, new_q in zip(self.interp_qw.ordered_times(), new_qs):
                self.interp_qw.set_keyframe(time, new_q.q[0])
                self.interp_qx.set_keyframe(time, new_q.q[1])
                self.interp_qy.set_keyframe(time, new_q.q[2])
                self.interp_qz.set_keyframe(time, new_q.q[3])

    def get_value(self, t, raw: bool = False):
        """Get interpolated pose at time t.

        Args:
            t: Time to evaluate pose at
            raw: If True, returns raw interpolated values (not implemented)

        Returns:
            Isometry object representing interpolated pose
        """
        if not self._precomputed:
            self.precompute()

        # Interpolate position
        new_x = self.interp_x.get_value(t)
        new_y = self.interp_y.get_value(t)
        new_z = self.interp_z.get_value(t)

        # Interpolate rotation based on rotation type
        if self.rotation_type == FreePoseAnimator.RotationType.BLENDER:
            new_qw = self.interp_qw.get_value(t)
            new_qx = self.interp_qx.get_value(t)
            new_qy = self.interp_qy.get_value(t)
            new_qz = self.interp_qz.get_value(t)
            new_q = np.array([new_qw, new_qx, new_qy, new_qz])
            new_q = Quaternion(new_q / (np.linalg.norm(new_q) + 1.0e-6))
        else:
            new_q = self.interp_q.get_value(t)
        new_iso = Isometry(q=new_q, t=[new_x, new_y, new_z])
        return new_iso


class SpinPoseAnimator(AnimatorBase):
    def __init__(self, interp_type: InterpType, center, spin_axis: str = '+Y'):
        super().__init__()
        self.interp_type = interp_type
        self.center = center
        self.spin_axis = Isometry._str_to_axis(spin_axis)
        if self.interp_type == InterpType.LINEAR:
            self.interp = self.add_interpolator('angle', LinearInterpolator())
        elif self.interp_type == InterpType.BEZIER:
            self.interp = self.add_interpolator('angle', BezierInterpolator())

    def set_keyframe(self, t, value):
        assert isinstance(value, float)
        self.interp.set_keyframe(t, value)

    def get_value(self, t, raw: bool = False):
        new_angle = self.interp.get_value(t)
        if raw:
            return new_angle
        return Isometry.from_axis_angle(self.spin_axis, radians=new_angle, t=self.center)


class SceneAnimator:
    """
    A class to manage and control animations in a scene.
    
    This class handles animation events for various scene elements like cameras, objects, and lights.
    It maintains a mapping of animation events for each object and attribute, and provides methods
    to set/get animators and control animation playback.

    The self.events dictionary maps object UUIDs to their animations. For example:
    {
        "obj11111111": {
            "pose": FreePoseAnimator(...),  # Animates object position/rotation
            "scale": ScalarAnimator(...) # Animates object scale
            "visible": ScalarAnimator(...) # Animates object visibility
        },
        "camera_base": {
            "relative_camera": FreePoseAnimator(...) # Animates camera base movement
        },
        "relative_camera": {
            "pose": FreePoseAnimator(...) # Animates relative camera (to camera base) movement
        },
        "sun": {
            "pose": FreePoseAnimator(...) # Animates light
        },
        "free": {
            "attrib_name": FreePoseAnimator(...) # Executes arbitrary commands at keyframes
        }
    }

    Args:
        scene: The scene object that this animator will control
    """
    def __init__(self, scene):
        self.scene = scene
        # uuid -> attributes -> animator
        #   or 'free' -> 'command' -> animator  
        self.events = defaultdict(dict)
        self.current_frame = 0

        # Frame range for the animation
        self._start_frame = 0
        self._end_frame = 0

    def is_enabled(self):
        """Check if animation is enabled by verifying frame range."""
        s, e = self.get_range()
        return s < e

    def get_range(self):
        """
        Calculate and return the frame range for all animations.
        
        Returns:
            tuple: (start_frame, end_frame) containing the full frame range
        """
        frame_max = self._end_frame
        frame_min = self._start_frame
        # Find min/max frames across all animators
        for obj_attrib in self.events.values():
            for obj_interp in obj_attrib.values():
                cur_first, cur_last = obj_interp.get_first_t(), obj_interp.get_last_t()
                if cur_last is not None:
                    frame_max = max(frame_max, cur_last)
                if cur_first is not None:
                    frame_min = min(frame_min, cur_first)
        self._start_frame = frame_min
        self._end_frame = frame_max
        return frame_min, frame_max

    def set_range(self, start_frame, end_frame):
        """
        Set the animation frame range and validate against actual keyframe range.
        
        Args:
            start_frame: Starting frame number
            end_frame: Ending frame number
        """
        self._start_frame, self._end_frame = start_frame, end_frame
        self.get_range() 
        # Warn if the provided range is not set accurately.
        if self._start_frame != start_frame:
            print(f"Warning (set_range): {self._start_frame} vs. {start_frame}!")
        if self._end_frame != end_frame:
            print(f"Warning (set_range): {self._end_frame} vs. {end_frame}!")


    def set_frame(self, t):
        """
        Set all objects in the scene to a specific animation frame. 
        
        Args:
            t: Frame number to set
        """
        # Apply each animator's value to the scene at frame t
        for obj_uuid, obj_attribs in self.events.items():
            for attrib_name, attrib_interp in obj_attribs.items():
                if attrib_interp.get_first_t() is None:
                    continue
                attrib_val = attrib_interp.get_value(t)
                # Handle different object types
                if obj_uuid == "relative_camera":
                    if attrib_name == "pose":
                        self.scene.relative_camera_pose = attrib_val
                    else:
                        raise NotImplementedError
                    
                elif obj_uuid == "camera_base":
                    self.scene.camera_base = attrib_val

                elif obj_uuid == "free":
                    eval(f"self.scene.{attrib_name} = attrib_val")

                elif attrib_name == "pose":
                    if obj_uuid in self.scene.objects.keys():
                        self.scene.objects[obj_uuid].pose = attrib_val
                    if obj_uuid in self.scene.lights.keys():
                        self.scene.lights[obj_uuid].pose = attrib_val

                elif attrib_name == "scale":
                    self.scene.objects[obj_uuid].scale = attrib_val

                elif attrib_name == "visible":
                    self.scene.objects[obj_uuid].visible = attrib_val
                    
        self.current_frame = t

    def set_current_frame(self):
        """Set scene to current frame."""
        self.set_frame(self.current_frame)
        
    def get_animator(self, obj_idx, attr_idx):
        """Get animator for a specific object and attribute.

        Args:
            obj_idx: Object identifier/name, e.g. "obj11111111"
            attr_idx: Attribute identifier/name, e.g. "pose"

        Returns:
            Animator instance if found, None otherwise
        """
        try:
            return self.events[obj_idx][attr_idx]
        except KeyError:
            return None

    def set_relative_camera(self, animator, no_override: bool = False):
        """Set animator for relative camera pose.

        Args:
            animator: Animator instance to control camera pose
            no_override: If True, won't override existing animator
        """
        if self.get_relative_camera() is not None and no_override:
            return
        self.events["relative_camera"]["pose"] = animator

    def get_relative_camera(self):
        """Get animator for relative camera pose.

        Returns:
            Animator instance controlling relative camera pose
        """
        return self.get_animator("relative_camera", "pose")

    def set_camera_base(self, animator, no_override: bool = False):
        """Set animator for camera base pose.

        Args:
            animator: Animator instance to control camera base pose
            no_override: If True, won't override existing animator
        """
        if self.get_camera_base() is not None and no_override:
            return
        self.events["camera_base"]["pose"] = animator

    def get_camera_base(self):
        """Get animator for camera base pose.

        Returns:
            Animator instance controlling camera base pose
        """
        return self.get_animator("camera_base", "pose")

    def set_object_pose(self, obj_uuid: str, animator):
        """Set animator for object pose.

        Args:
            obj_uuid: Object identifier/name
            animator: Animator instance to control object pose
        """
        assert obj_uuid in self.scene.objects.keys()  # Verify object exists
        self.events[obj_uuid]["pose"] = animator

    def set_object_scale(self, obj_uuid: str, animator):
        """Set animator for object scale.

        Args:
            obj_uuid: Object identifier/name
            animator: Animator instance to control object scale
        """
        assert obj_uuid in self.scene.objects.keys()  # Verify object exists
        self.events[obj_uuid]["scale"] = animator

    def set_object_visibility(self, obj_uuid: str, animator):
        """Set animator for object visibility.

        Args:
            obj_uuid: Object identifier/name
            animator: Animator instance to control object visibility
        """
        assert obj_uuid in self.scene.objects.keys()  # Verify object exists
        self.events[obj_uuid]["visible"] = animator

    def set_light_pose(self, light_name: str, animator):
        """Set animator for light pose.

        Args:
            light_name: Light identifier/name
            animator: Animator instance to control light pose
        """
        assert light_name in self.scene.lights.keys()  # Verify light exists
        self.events[light_name]["pose"] = animator

    def set_sun_pose(self, animator, sun_name: str = 'sun'):
        """Set animator for sun light pose.

        Args:
            animator: Animator instance to control sun light pose
        """
        self.set_light_pose(sun_name, animator)

    def get_light_pose(self, light_name):
        """Get animator for light pose.

        Args:
            light_name: Light identifier/name

        Returns:
            Animator instance controlling light pose
        """
        return self.get_animator(light_name, "pose")

    def get_sun_pose(self):
        """Get animator for sun light pose.

        Returns:
            Animator instance controlling sun light pose
        """
        return self.get_light_pose('sun')

    def set_free_command(self, command: str, animator):
        """Set animator for arbitrary scene command.

        Args:
            command: Command string to animate, evaluated as self.scene.command
            animator: Animator instance to control command
        """
        self.events['free'][command] = animator
