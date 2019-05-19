// Ray marching outside a Schwarzschild black hole.
// Doesn't do anything particularly clever or interesting: it's doing
// straightforward Euler integration of the geodesic equations.
// We're doing this the typical raytracing way (starting at the camera,
// and seeing where a ray would have come from); we don't need to worry
// about that, though, because as long as we stay outside the event
// horizon, reversing the direction of a ray causes it to exactly retrace
// its path. (This is not true of Kerr black holes.)

__constant float GM = 10.;

// Note: we could have used this during the iteration step instead of calling
// null_dt each time. But using null_dt instead ensures we don't drift away
// from being a null path.
float d2t(float r, float dt, float dr) {
  return -2. * GM / (r * (r - 2. * GM)) * dr * dt;
}

// Second derivative of r.
float d2r(float r, float dt, float dr, float dtheta) {
  return - GM / (r * r * r) * (r - 2. * GM) * dt * dt
    + GM / (r * (r - 2. * GM)) * dr * dr
    + (r - 2. * GM) * dtheta * dtheta;
}

// Second derivative of theta.
float d2theta(float r, float dr, float dtheta) {
  return - 2. / r * dtheta * dr;
}

// Given r, dr, and dtheta, determine the value for dt that will make this a null path.
float null_dt(float r, float dr, float dtheta) {
  float q = 1. - 2. * GM / r;

  // Doesn't actually matter which root we take--dt only appears squared
  // or not at all in d2r and d2theta.
  return sqrt((dr * dr / (q * q)) + r * r * dtheta * dtheta / q);
}

__constant int NUM_ITER = 1000000;
__constant float TS = 0.01;

// The main kernel! Computes the "outcome" for each ray in the given range,
// which in turn consists of:
// - whether the ray falls into the black hole, and if so, the angle of the point where
//   it crosses the event horizon
// - whether the ray escapes to infinity, and if so, the angle at which it escapes
//
// outcomes[i] = 0 for rays that are captured, and 1 for rays that escape.
__kernel void gen_outcomes(
                           __global float *angles,
                           __global unsigned char *outcomes,
                           float min,
                           float max,
                           int num,
                           float start_r
                           ) {
  // Small fudge factor here, because if we get too close, Euler's method can blow up.
  float min_r = 2 * GM + 0.0001;
  int slot = get_global_id(0);
  float i = (float)(slot);

  // Figure out which way the ith ray actually points (in rectangular coordinates)
  float frac = i / (float)(num - 1);
  float ray_amt = max * frac + min * (1. - frac);
  float dx = ray_amt;
  float dz = 1.;

  float r = start_r;
  float theta = M_PI; // We assume we're starting at x=0.

  // Convert to Schwarzschild coordinates
  float dr = -start_r * dz / r;
  float dtheta = - start_r * dx / (r*r);

  bool hit = false;

  for (int t = 0; t < NUM_ITER; t++) {
    // We're only considering null paths; dt is determined by r, dr, dtheta,
    // and the condition that the path is null.
    float dt = null_dt(r, dr, dtheta);

    // For r and theta, on the other hand, we need to solve differential equations.
    // We're using plain old Euler's method. First, get the second derivatives.
    float ddr = d2r(r, dt, dr, dtheta);
    float ddtheta = d2theta(r, dr, dtheta);

    // From those, compute the derivatives.
    dr += TS * ddr;
    dtheta += TS * ddtheta;

    // And finally r and theta themselves.
    r += TS * dr;
    theta += TS * dtheta;

    if (r <= min_r) {
      // We've fallen into the black hole.
      hit = true;
      break;
    }

    if (r > 500.) {
      // We're far enough that the direction we're going is pretty much the
      // direction we'll keep going.
      break;
    }
  }

  if (hit) {
    // If the ray was captured, what matters is the angle of its position.
    outcomes[slot] = 0;
    angles[slot] = M_PI/2. - theta;
  } else {
    // If the ray escaped, what matters is the angle of its *direction*.
    // (In practice, though, this should be pretty similar to the angle
    // of its position--this math might not really be necessary.)
    float dx = r * cos(theta) * dtheta + sin(theta) * dr;
    float dz = -r * sin(theta) * dtheta + cos(theta) * dr;

    outcomes[slot] = 1;
    angles[slot] = atan2(dz, dx);
  }
}
