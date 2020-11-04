import math

import tensorflow as tf


def _inverse_coefficents(k):
    return [
        -k[0],
        3.0 * (k[0] * k[0]) - k[1],
        -12.0 * (k[0] * k[0]) * k[0] + 8.0 * k[0] * k[1],
        55.0 * (k[0] * k[0]) * (k[0] * k[0]) - 55.0 * (k[0] * k[0]) * k[1] + 5.0 * (k[1] * k[1]),
    ]


def _distort(r, k):
    ik = _inverse_coefficents(k)
    return r * (
        1.0
        + ik[0] * (r * r)
        + ik[1] * ((r * r) * (r * r))
        + ik[2] * ((r * r) * (r * r)) * (r * r)
        + ik[3] * ((r * r) * (r * r)) * ((r * r) * (r * r))
    )


def _equidistant_r(theta, f):
    return f * theta


def _rectilinear_r(theta, f):
    return f * tf.math.tan(tf.clip_by_value(theta, 0.0, math.pi * 0.5))


def _equisolid_r(theta, f):
    return 2.0 * f * tf.math.sin(theta * 0.5)


def project(V, dimensions, projection, f, centre, k):

    #  Perform the projection math
    theta = tf.math.acos(V[:, 0])
    rsin_theta = tf.math.rsqrt(1.0 - tf.square(V[:, 0]))
    if projection == "RECTILINEAR":
        r_u = _rectilinear_r(theta, f)
    elif projection == "EQUISOLID":
        r_u = _equisolid_r(theta, f)
    elif projection == "EQUIDISTANT":
        r_u = _equidistant_r(theta, f)
    else:
        r_u = tf.zeros_like(theta)

    r_d = _distort(r_u, k)

    # Screen as y,x
    screen = tf.stack([r_d * V[:, 2] * rsin_theta, r_d * V[:, 1] * rsin_theta], axis=1)

    # Sometimes floating point error makes x > 1.0
    # In this case we are basically the centre of the screen anway
    screen = tf.where(tf.math.is_finite(screen), screen, 0.0)

    # Convert to pixel coordinates
    return (tf.cast(dimensions, tf.float32) * 0.5) - screen - centre
