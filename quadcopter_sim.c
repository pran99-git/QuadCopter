/*
 * quadcopter_sim.c
 *
 * C conversion of the MATLAB quadcopter flight simulator (Assignment 3 - Part 2).
 * Simulates a quadcopter's rigid-body dynamics with a PID attitude controller.
 *
 * Concepts implemented:
 *   - Rigid-body 6-DOF dynamics (Newton-Euler formulation)
 *   - SO(3) rotation matrix and body-frame angular kinematics
 *   - PD attitude controller (P=9, D=5, I=0)
 *   - Euler forward integration
 *   - Motor mixing (thrust + torque allocation)
 *
 * Build:  gcc -O2 -lm -o quadcopter_sim quadcopter_sim.c
 */

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* =========================================================================
 * DATA TYPES
 * =========================================================================
 * We replace MATLAB's dynamic struct with a plain C struct.
 * Every field maps 1-to-1 to the MATLAB state struct fields.
 * ========================================================================= */

/* A 3-element column vector  */
typedef struct { double v[3]; } Vec3;

/* A 3x3 matrix stored in row-major order (R[row][col]) */
typedef struct { double m[3][3]; } Mat3;

/*
 * QuadState - the master state struct.
 *
 * The MATLAB code accumulates every physical and control variable inside
 * a single 'state' struct that is passed by value through every function.
 * We mirror that design here so the logic stays readable.
 */
typedef struct {

    /* ---- Physical constants ---- */
    double g;   /* gravitational acceleration, 9.81 m/s^2                   */
    double m;   /* total mass, 0.8 kg                                        */
    double L;   /* arm length (motor-to-CG), 0.25 m                         */
    double k;   /* thrust coefficient  F = k * omega^2,  k = 3e-3 N/(rad/s)^2 */
    double kb;  /* drag / reaction torque coefficient, kb = 1e-7            */
    double kd;  /* translational drag constant, 0.25 N·s/m                  */
    double Ix;  /* moment of inertia about body X axis, 5e-3 kg·m^2         */
    double Iy;  /* moment of inertia about body Y axis, 5e-3 kg·m^2         */
    double Iz;  /* moment of inertia about body Z axis, 10e-3 kg·m^2        */
    double dt;  /* simulation time-step, 0.01 s                              */

    /* ---- PID gains ---- */
    double P;   /* proportional gain = 9                                     */
    double I;   /* integral gain     = 0  (wind-up guard keeps it off)       */
    double D;   /* derivative gain   = 5                                     */

    /* ---- Translational state (inertial / world frame) ---- */
    Vec3 x;     /* position [x, y, z], initial z = 50 m                     */
    Vec3 xdot;  /* linear velocity [vx, vy, vz]                              */
    Vec3 a;     /* linear acceleration [ax, ay, az]                          */

    /* ---- Rotational state (Euler angles, body frame) ---- */
    Vec3 theta;    /* Euler angles  [phi, theta, psi]  = [roll, pitch, yaw]  */
    Vec3 thetadot; /* time-derivatives of Euler angles                       */
    Vec3 omega;    /* angular velocity expressed in the body frame            */
    Vec3 omegadot; /* angular acceleration in the body frame                 */

    /* ---- Frames ---- */
    Mat3 R; /* rotation matrix: body-frame -> world-frame  (3x3 DCM)        */
    Mat3 W; /* kinematic mapping: thetadot -> omega                          */

    /* ---- Control signals ---- */
    double thrust;  /* total collective thrust (scalar, body-frame +z)       */
    Vec3   tau;     /* torque vector [tau_x, tau_y, tau_z] in body frame     */
    Vec3   error;   /* PID error  = P*theta + I*integral + D*thetadot        */
    Vec3   integral;/* accumulated integral of theta over time                */

} QuadState;

/* =========================================================================
 * MATH HELPERS
 * =========================================================================
 * MATLAB has built-in matrix ops.  In C we need small inline helpers.
 * ========================================================================= */

/*
 * mat3_mul_vec3 - multiply a 3x3 matrix by a 3-vector.
 *
 * This replaces expressions like:  T = R * [0; 0; thrust]
 * which appear in update_acceleration().
 */
static Vec3 mat3_mul_vec3(const Mat3 *A, const Vec3 *b)
{
    Vec3 out;
    for (int i = 0; i < 3; i++) {
        out.v[i] = A->m[i][0]*b->v[0]
                 + A->m[i][1]*b->v[1]
                 + A->m[i][2]*b->v[2];
    }
    return out;
}

/*
 * mat3_solve - solve A*x = b for x  (Gaussian elimination, 3x3).
 *
 * Used to invert the kinematic mapping W in update_thetadot():
 *     thetadot = W \ omega   (MATLAB backslash)
 */
static Vec3 mat3_solve(const Mat3 *A, const Vec3 *b)
{
    /* Work on augmented matrix [A | b] */
    double aug[3][4];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) aug[i][j] = A->m[i][j];
        aug[i][3] = b->v[i];
    }

    /* Forward elimination with partial pivoting */
    for (int col = 0; col < 3; col++) {
        /* Find pivot row */
        int pivot = col;
        for (int row = col+1; row < 3; row++)
            if (fabs(aug[row][col]) > fabs(aug[pivot][col])) pivot = row;
        /* Swap */
        for (int j = 0; j < 4; j++) {
            double tmp = aug[col][j]; aug[col][j] = aug[pivot][j]; aug[pivot][j] = tmp;
        }
        /* Eliminate below */
        for (int row = col+1; row < 3; row++) {
            double factor = aug[row][col] / aug[col][col];
            for (int j = col; j < 4; j++)
                aug[row][j] -= factor * aug[col][j];
        }
    }

    /* Back substitution */
    Vec3 x;
    for (int i = 2; i >= 0; i--) {
        x.v[i] = aug[i][3];
        for (int j = i+1; j < 3; j++)
            x.v[i] -= aug[i][j] * x.v[j];
        x.v[i] /= aug[i][i];
    }
    return x;
}

/*
 * cross3 - 3-vector cross product  c = a x b
 *
 * Used in update_omegadot() for the gyroscopic term:
 *     omegadot = I \ (tau - cross(omega, I*omega))
 */
static Vec3 cross3(const Vec3 *a, const Vec3 *b)
{
    Vec3 c;
    c.v[0] = a->v[1]*b->v[2] - a->v[2]*b->v[1];
    c.v[1] = a->v[2]*b->v[0] - a->v[0]*b->v[2];
    c.v[2] = a->v[0]*b->v[1] - a->v[1]*b->v[0];
    return c;
}

/* =========================================================================
 * SETUP
 * ========================================================================= */

/*
 * setup_state()  <->  MATLAB setup_state()
 *
 * Initialises every field of QuadState to its physical starting value.
 *
 * Key flight-dynamics note:
 *   theta is seeded with rand(3,1) - a small random Euler-angle disturbance.
 *   This tests whether the PD controller can recover attitude from a
 *   perturbed initial condition (it should drive theta -> [0,0,0]).
 */
QuadState setup_state(void)
{
    QuadState s;
    memset(&s, 0, sizeof(s));   /* zero everything first */

    /* Physical constants */
    s.g  = 9.81;
    s.m  = 0.8;
    s.L  = 0.25;
    s.k  = 3e-3;
    s.kb = 1e-7;
    s.kd = 0.25;
    s.Ix = 5e-3;
    s.Iy = 5e-3;
    s.Iz = 10e-3;
    s.dt = 1e-2;

    /* PID gains */
    s.P = 9.0;
    s.I = 0.0;
    s.D = 5.0;

    /*
     * Initial position: hovering at z = 50 m.
     * x[0]=x, x[1]=y, x[2]=z  in the world (inertial) frame.
     */
    s.x.v[0] = 0.0;
    s.x.v[1] = 0.0;
    s.x.v[2] = 50.0;

    /*
     * Random initial attitude disturbance [0,1) rad on each Euler angle.
     * Simulates the drone being placed with an imperfect orientation.
     * The controller must bring these to zero.
     */
    s.theta.v[0] = (double)rand() / RAND_MAX;   /* phi   (roll)  */
    s.theta.v[1] = (double)rand() / RAND_MAX;   /* theta (pitch) */
    s.theta.v[2] = (double)rand() / RAND_MAX;   /* psi   (yaw)   */

    /* All velocities, accelerations, torques start at zero */
    return s;
}

/* =========================================================================
 * ROTATION / KINEMATIC FRAME UPDATES
 * ========================================================================= */

/*
 * update_R()  <->  MATLAB update_R()
 *
 * Builds the ZYX (3-2-1) Direction Cosine Matrix (DCM) from Euler angles.
 *
 * R transforms a vector from the BODY frame to the WORLD (inertial) frame:
 *       v_world = R * v_body
 *
 * The 3x3 entries implement the standard ZYX rotation composition:
 *       R = Rz(psi) * Ry(theta) * Rx(phi)
 *
 * Flight dynamics use:
 *   - phi   (theta[0]) = roll  around X
 *   - theta (theta[1]) = pitch around Y
 *   - psi   (theta[2]) = yaw   around Z
 *
 * This matrix is later used in update_acceleration() to rotate the
 * body-frame thrust vector into the world frame so we can compute
 * the net force acting on the vehicle.
 */
static void update_R(QuadState *s)
{
    double phi   = s->theta.v[0];
    double theta = s->theta.v[1];
    double psi   = s->theta.v[2];

    /* Row 0 */
    s->R.m[0][0] =  cos(psi)*cos(theta);
    s->R.m[0][1] =  cos(psi)*sin(phi)*sin(theta) - cos(phi)*sin(psi);
    s->R.m[0][2] =  sin(phi)*sin(psi) + cos(phi)*cos(psi)*sin(theta);
    /* Row 1 */
    s->R.m[1][0] =  cos(theta)*sin(psi);
    s->R.m[1][1] =  cos(phi)*cos(psi) + sin(phi)*sin(psi)*sin(theta);
    s->R.m[1][2] =  cos(phi)*sin(psi)*sin(theta) - cos(psi)*sin(phi);
    /* Row 2 */
    s->R.m[2][0] = -sin(theta);
    s->R.m[2][1] =  cos(theta)*sin(phi);
    s->R.m[2][2] =  cos(phi)*cos(theta);
}

/*
 * update_W()  <->  MATLAB update_W()
 *
 * Builds the kinematic mapping matrix W such that:
 *       omega_body = W * thetadot
 *
 * where omega_body is the angular velocity expressed in the BODY frame,
 * and thetadot = [phi_dot, theta_dot, psi_dot] are the Euler-angle rates.
 *
 * This mapping is NOT simply the identity because Euler-angle rates are
 * NOT the components of the body angular velocity vector.  The W matrix
 * accounts for the sequential nature of the ZYX rotation.
 *
 * Its inverse (W\omega) is used to go back from body angular rates to
 * Euler-angle rates in update_thetadot().
 *
 * Note: W becomes singular at theta = ±90° (gimbal lock).
 */
static void update_W(QuadState *s)
{
    double phi   = s->theta.v[0];
    double theta = s->theta.v[1];

    /* Row 0: phi contribution to omega */
    s->W.m[0][0] = 1.0;
    s->W.m[0][1] = 0.0;
    s->W.m[0][2] = -sin(theta);
    /* Row 1: theta contribution to omega */
    s->W.m[1][0] = 0.0;
    s->W.m[1][1] =  cos(phi);
    s->W.m[1][2] =  cos(theta)*sin(phi);
    /* Row 2: psi contribution to omega */
    s->W.m[2][0] = 0.0;
    s->W.m[2][1] = -sin(phi);
    s->W.m[2][2] =  cos(theta)*cos(phi);
}

/* =========================================================================
 * CONTROLLER
 * ========================================================================= */

/*
 * apply_control()  <->  MATLAB apply_control()
 *
 * This is the flight controller.  It implements a PD attitude stabilisation
 * loop plus a collective-thrust hover computation.
 *
 * ---- Hover thrust (collective / collective-pitch) ----
 *
 *   For the vehicle to hover (xddot = 0) the vertical component of thrust
 *   must equal gravity:
 *       T * cos(phi)*cos(theta) = m*g
 *       => T = m*g / (cos(phi)*cos(theta))
 *
 *   This compensates for the tilt: as the vehicle rolls/pitches the thrust
 *   vector tilts, so we need more total thrust to maintain altitude.
 *
 * ---- PD attitude controller ----
 *
 *   The desired attitude is theta = [0, 0, 0] (level flight).
 *   Error = P*theta + I*integral(theta) + D*thetadot
 *
 *   With I=0 this is a straight PD controller.
 *   The error drives the torque commands, which in turn dictate the
 *   per-motor speed squared inputs (in[1..4]).
 *
 * ---- Motor mixing (allocation matrix) ----
 *
 *   A quadcopter has 4 rotors.  Each rotor i produces:
 *       - thrust:          F_i = k * in_i     (in_i = omega_i^2)
 *       - reaction torque: Q_i = kb * in_i    (opposing rotation direction)
 *
 *   The X-configuration mixing assigns:
 *       in(1) = T/(4k) - e1*Ix/(2kL) - e3*Iz/(4kb)   (front-left,  CW)
 *       in(2) = T/(4k)               - e2*Iy/(2kL) + e3*Iz/(4kb)   (front-right, CCW)
 *       in(3) = T/(4k) + e1*Ix/(2kL)              - e3*Iz/(4kb)   (rear-right,  CW)
 *       in(4) = T/(4k)               + e2*Iy/(2kL) + e3*Iz/(4kb)   (rear-left,  CCW)
 *
 *   Then the resulting body torques are:
 *       tau_x = L*k*(in1 - in3)        roll torque
 *       tau_y = L*k*(in2 - in4)        pitch torque
 *       tau_z = kb*(in1 - in2 + in3 - in4)  yaw torque (reaction)
 */
static void apply_control(QuadState *s)
{
    double in[4];   /* motor inputs (proportional to rotor omega^2) */

    /* --- Hover thrust ---
     * Tilt-compensated so the vehicle doesn't sink when it rolls/pitches. */
    s->thrust = s->m * s->g / (cos(s->theta.v[0]) * cos(s->theta.v[1]));

    /* --- PD error signal ---
     * error = P * theta  +  I * integral  +  D * thetadot
     * With I=0, this simplifies to:  error = P*theta + D*thetadot
     */
    for (int i = 0; i < 3; i++)
        s->error.v[i] = s->P * s->theta.v[i]
                      + s->I * s->integral.v[i]
                      + s->D * s->thetadot.v[i];

    double e1 = s->error.v[0];   /* roll  error */
    double e2 = s->error.v[1];   /* pitch error */
    double e3 = s->error.v[2];   /* yaw   error */

    /* --- Integral accumulation (anti-windup) ---
     * Accumulate integral term.  If the integral grows too large,
     * reset it to prevent integrator windup (I=0 so this is harmless here,
     * but the structure is preserved for future tuning with I > 0).
     */
    for (int i = 0; i < 3; i++)
        s->integral.v[i] += s->theta.v[i] * s->dt;

    /* Anti-windup: reset integral if any component is too large */
    double max_int = 0.0;
    for (int i = 0; i < 3; i++)
        if (fabs(s->integral.v[i]) > max_int) max_int = fabs(s->integral.v[i]);
    if (max_int > 0.1 * s->dt)
        memset(&s->integral, 0, sizeof(s->integral));

    /* --- Motor mixing: compute per-motor input (omega_i^2) --- */
    double T  = s->thrust;
    double T4k = T / (4.0 * s->k);     /* each motor's share of hover thrust */

    in[0] = T4k - e1*s->Ix/(2.0*s->k*s->L)                        - e3*s->Iz/(4.0*s->kb);
    in[1] = T4k                       - e2*s->Iy/(2.0*s->k*s->L)  + e3*s->Iz/(4.0*s->kb);
    in[2] = T4k + e1*s->Ix/(2.0*s->k*s->L)                        - e3*s->Iz/(4.0*s->kb);
    in[3] = T4k                       + e2*s->Iy/(2.0*s->k*s->L)  + e3*s->Iz/(4.0*s->kb);

    /* --- Torque computation from motor inputs ---
     *
     * tau_x (roll):  differential thrust between motors 1 & 3 (left-right pair)
     * tau_y (pitch): differential thrust between motors 2 & 4 (front-rear pair)
     * tau_z (yaw):   differential reaction torques (CW vs CCW rotors)
     */
    s->tau.v[0] = s->L * s->k  * (in[0] - in[2]);
    s->tau.v[1] = s->L * s->k  * (in[1] - in[3]);
    s->tau.v[2] = s->kb        * (in[0] - in[1] + in[2] - in[3]);
}

/* =========================================================================
 * DYNAMICS UPDATES
 * ========================================================================= */

/*
 * update_acceleration()  <->  MATLAB update_acceleration()
 *
 * Computes the world-frame linear acceleration using Newton's 2nd law:
 *
 *   m * a = F_gravity + F_thrust + F_drag
 *
 * In component form (dividing by m):
 *   a = [0, 0, -g]                        (gravity, always -z in world frame)
 *     + R * [0, 0, T] / m                 (thrust rotated to world frame)
 *     - kd * xdot / m                     (translational drag, opposes velocity)
 *
 * The thrust vector is [0, 0, T] in the BODY frame (rotors push upward
 * along the body z-axis).  We multiply by R to get it in the world frame.
 */
static void update_acceleration(QuadState *s)
{
    update_R(s);    /* ensure R is current for this theta */

    /* Thrust vector in body frame: only z-component is non-zero */
    Vec3 thrust_body = {{0.0, 0.0, s->thrust}};

    /* Rotate thrust into world frame:  T_world = R * thrust_body */
    Vec3 thrust_world = mat3_mul_vec3(&s->R, &thrust_body);

    /* a = gravity + thrust/m - drag/m */
    s->a.v[0] =  0.0 + thrust_world.v[0]/s->m - s->kd*s->xdot.v[0]/s->m;
    s->a.v[1] =  0.0 + thrust_world.v[1]/s->m - s->kd*s->xdot.v[1]/s->m;
    s->a.v[2] = -s->g + thrust_world.v[2]/s->m - s->kd*s->xdot.v[2]/s->m;
}

/*
 * update_omega()  <->  MATLAB update_omega()
 *
 * Converts Euler-angle rates (thetadot) to body angular velocity (omega):
 *
 *   omega = W * thetadot
 *
 * This is needed because the equations of motion (Newton-Euler) are
 * written in the body frame using omega, but the state is tracked via
 * Euler angles.
 */
static void update_omega(QuadState *s)
{
    update_W(s);    /* ensure W is current for this theta */
    s->omega = mat3_mul_vec3(&s->W, &s->thetadot);
}

/*
 * update_omegadot()  <->  MATLAB update_omegadot()
 *
 * Implements Euler's rotational equation of motion:
 *
 *   I * omegadot = tau - omega x (I * omega)
 *   => omegadot  = I^-1 * [tau - omega x (I * omega)]
 *
 * - tau:  control torques from the motor mixing
 * - I:    diagonal inertia tensor [Ix, Iy, Iz]
 * - omega x (I*omega): gyroscopic (Coriolis) term; for a symmetric body
 *   this is the coupling between the three rotation axes caused by the
 *   body's own spin.  It must be subtracted to get the correct angular
 *   acceleration.
 *
 * Because I is diagonal, I^-1 is trivial: 1/Ix, 1/Iy, 1/Iz.
 */
static void update_omegadot(QuadState *s)
{
    /* I * omega (element-wise since I is diagonal) */
    Vec3 I_omega;
    I_omega.v[0] = s->Ix * s->omega.v[0];
    I_omega.v[1] = s->Iy * s->omega.v[1];
    I_omega.v[2] = s->Iz * s->omega.v[2];

    /* Gyroscopic term: omega x (I*omega) */
    Vec3 gyro = cross3(&s->omega, &I_omega);

    /* Net rotational forcing: tau - gyro */
    Vec3 net;
    net.v[0] = s->tau.v[0] - gyro.v[0];
    net.v[1] = s->tau.v[1] - gyro.v[1];
    net.v[2] = s->tau.v[2] - gyro.v[2];

    /* omegadot = I^-1 * net  (I^-1 is just 1/diag for diagonal I) */
    s->omegadot.v[0] = net.v[0] / s->Ix;
    s->omegadot.v[1] = net.v[1] / s->Iy;
    s->omegadot.v[2] = net.v[2] / s->Iz;
}

/* =========================================================================
 * INTEGRATION (STATE ADVANCEMENT)
 * ========================================================================= */

/*
 * update_thetadot()  <->  MATLAB update_thetadot()
 *
 * Recovers Euler-angle rates from the (just-updated) body angular velocity:
 *
 *   thetadot = W^-1 * omega   (MATLAB: thetadot = W \ omega)
 *
 * We use Gaussian elimination (mat3_solve) since W is a dense 3x3 matrix.
 */
static void update_thetadot(QuadState *s)
{
    update_W(s);
    s->thetadot = mat3_solve(&s->W, &s->omega);
}

/*
 * advance()  <->  MATLAB advance()
 *
 * Performs one Euler forward-integration step for all state variables.
 *
 * Euler integration:  x(t+dt) = x(t) + x_dot(t) * dt
 *
 * Integration order (matches MATLAB exactly):
 *   1. omega    += omegadot * dt       update body angular velocity
 *   2. thetadot  = W^-1 * omega        convert back to Euler rates
 *   3. theta    += thetadot * dt       advance Euler angles
 *   4. xdot     += a * dt             advance linear velocity
 *   5. x        += xdot * dt          advance position
 *
 * The order matters: thetadot is recomputed from the newly integrated omega
 * before it is used to advance theta, which is the correct causal order.
 */
static void advance(QuadState *s)
{
    /* Step 1: integrate angular velocity (body frame) */
    for (int i = 0; i < 3; i++)
        s->omega.v[i] += s->omegadot.v[i] * s->dt;

    /* Step 2: convert updated omega back to Euler-angle rates */
    update_thetadot(s);

    /* Step 3: integrate Euler angles */
    for (int i = 0; i < 3; i++)
        s->theta.v[i] += s->thetadot.v[i] * s->dt;

    /* Step 4: integrate linear velocity */
    for (int i = 0; i < 3; i++)
        s->xdot.v[i] += s->a.v[i] * s->dt;

    /* Step 5: integrate position */
    for (int i = 0; i < 3; i++)
        s->x.v[i] += s->xdot.v[i] * s->dt;
}

/* =========================================================================
 * TELEMETRY OUTPUT  (replaces MATLAB graphical 'paint')
 * =========================================================================
 * The MATLAB paint() function renders a 3D figure.  In plain C we don't
 * have a graphics library, so we print CSV-formatted telemetry to stdout
 * instead.  The data can be piped into Python/MATLAB/Excel for plotting.
 * ========================================================================= */

static void print_header(void)
{
    printf("step,"
           "x,y,z,"
           "vx,vy,vz,"
           "phi_deg,theta_deg,psi_deg,"
           "phi_dot,theta_dot,psi_dot,"
           "thrust,"
           "tau_x,tau_y,tau_z\n");
}

static void print_state(int step, const QuadState *s)
{
    printf("%d,"
           "%.4f,%.4f,%.4f,"     /* position          */
           "%.4f,%.4f,%.4f,"     /* velocity          */
           "%.4f,%.4f,%.4f,"     /* Euler angles (deg)*/
           "%.4f,%.4f,%.4f,"     /* thetadot          */
           "%.4f,"               /* thrust            */
           "%.6f,%.6f,%.6f\n",   /* torques           */
           step,
           s->x.v[0],    s->x.v[1],    s->x.v[2],
           s->xdot.v[0], s->xdot.v[1], s->xdot.v[2],
           s->theta.v[0]*(180.0/M_PI),   /* convert rad -> deg for readability */
           s->theta.v[1]*(180.0/M_PI),
           s->theta.v[2]*(180.0/M_PI),
           s->thetadot.v[0], s->thetadot.v[1], s->thetadot.v[2],
           s->thrust,
           s->tau.v[0], s->tau.v[1], s->tau.v[2]);
}

/* =========================================================================
 * MAIN SIMULATION LOOP
 * =========================================================================
 * Mirrors the top-level Simulator() function in MATLAB.
 *
 * The loop runs for 1000 steps at dt=0.01 s -> 10 seconds of simulated time.
 *
 * Each iteration:
 *   1. apply_control      - PD controller computes thrust + torques
 *   2. update_acceleration- Newton's 2nd law -> linear acceleration
 *   3. update_omega       - kinematics: thetadot -> omega
 *   4. update_omegadot    - Euler's equation -> angular acceleration
 *   5. advance            - Euler integration: advance all states by dt
 *   6. print_state        - telemetry output (replaces MATLAB paint)
 * ========================================================================= */

int main(void)
{
    srand(42);   /* fixed seed for reproducible random theta disturbance */

    QuadState state = setup_state();
    print_header();

    for (int i = 0; i < 1000; i++) {

        /* 1. Controller: compute thrust and torques from current attitude */
        apply_control(&state);

        /* 2. Translational dynamics: F=ma in world frame */
        update_acceleration(&state);

        /* 3. Rotational kinematics: thetadot -> omega (body frame) */
        update_omega(&state);

        /* 4. Rotational dynamics: Euler's equation for omegadot */
        update_omegadot(&state);

        /* 5. Integrate all state variables forward by dt */
        advance(&state);

        /* 6. Log telemetry (stdout CSV) */
        print_state(i, &state);
    }

    return 0;
}
