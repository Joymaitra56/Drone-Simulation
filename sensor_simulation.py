import os
import math
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import mujoco
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

# ============================================================
#  Quaternion Helper Functions (MuJoCo format: [w, x, y, z])
# ============================================================

def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def rotate_vector_by_quat(v, q):
    q_v = np.array([0.0, v[0], v[1], v[2]])
    q_conj = quat_conjugate(q)
    return quat_mul(quat_mul(q, q_v), q_conj)[1:]

# Copy of terrain height function to keep the suite independent
_PADS = [
    ("Home",  -4.0, -4.0, 0.6),
    ("Pad 1", -4.0,  4.0, 0.4),
    ("Pad 2",  4.0,  4.0, 0.4),
    ("Pad 3",  4.0, -4.0, 0.4),
    ("Pad 4",  0.0,  2.0, 0.4),
    ("Pad 5",  0.0, -2.0, 0.4),
]
def get_terrain_height(x, y):
    h_val  = 120.0
    h_val += 60.0 * math.sin(0.4 * x)  * math.cos(0.4 * y)
    h_val += 30.0 * math.sin(1.1 * x + 0.5) * math.cos(0.9 * y - 0.2)
    h_val += 15.0 * math.sin(2.5 * x)  * math.sin(2.8 * y)
    h_val +=  5.0 * math.sin(6.0 * x)  * math.cos(5.5 * y)
    flatten_factor = 1.0
    for _, px, py, pr in _PADS:
        dx = x - px;  dy = y - py
        d = math.sqrt(dx*dx + dy*dy)
        if d < pr:
            flatten_factor = 0.0;  break
        elif d < pr + 0.4:
            t = (d - pr) / 0.4
            f = 3.0*t*t - 2.0*t*t*t
            if f < flatten_factor:
                flatten_factor = f
    return 0.4 * (h_val * flatten_factor / 255.0)

# ============================================================
#  1D Kalman Filter Class (Kinematic [pos, vel]^T)
# ============================================================

class KalmanFilter1D:
    def __init__(self, init_pos, init_vel, q_acc, r_meas):
        # State: [pos, vel]^T
        self.x = np.array([init_pos, init_vel], dtype=np.float64)
        # Error covariance
        self.P = np.eye(2, dtype=np.float64) * 0.1
        # Acceleration process noise variance
        self.q_acc = q_acc
        # Measurement noise variance
        self.r_meas = r_meas

    def predict(self, acc, dt):
        A = np.array([[1.0, dt],
                      [0.0, 1.0]], dtype=np.float64)
        B = np.array([0.5 * dt**2, dt], dtype=np.float64)
        # State transition
        self.x = A @ self.x + B * acc
        # Process covariance update
        Q = np.array([[0.25 * dt**4, 0.5 * dt**3],
                      [0.5 * dt**3,  dt**2]], dtype=np.float64) * self.q_acc
        self.P = A @ self.P @ A.T + Q

    def update(self, meas):
        H = np.array([1.0, 0.0], dtype=np.float64)
        # Innovation
        y = meas - self.x[0]
        # Innovation covariance
        S = np.dot(H, self.P @ H) + self.r_meas
        # Kalman Gain
        K = (self.P @ H) / S
        # State update
        self.x = self.x + K * y
        # Covariance update
        self.P = (np.eye(2, dtype=np.float64) - np.outer(K, H)) @ self.P

# ============================================================
#  Sensor Simulation and Filtering Suite
# ============================================================

class SensorSimulationSuite:
    def __init__(self, dt=0.005):
        self.dt = dt
        self.step_count = 0

        # Drone joint and body properties (assigned at first step)
        self.drone_body_id = None
        self.qpos_adr = None
        self.qvel_adr = None

        # State tracking for acceleration numerical differentiation
        self.prev_vel_world = None

        # GPS status (rate: 10Hz, every 20 steps of 0.005s)
        self.gps_update_period = 20
        self.last_gps_x = 0.0
        self.last_gps_y = 0.0
        self.last_gps_z = 0.0

        # Filter states
        # 1. EMA (α = 0.15)
        self.ema_x = 0.0
        self.ema_y = 0.0
        self.ema_z = 0.0
        self.ema_alt = 0.0
        self.ema_front = 0.0
        self.ema_alpha = 0.15

        # 2. Complementary Altitude (Z)
        self.comp_z = 0.0
        self.comp_vz = 0.0
        self.comp_z_alpha = 0.98

        # 3. Complementary Attitude (Roll / Pitch)
        self.comp_roll = 0.0
        self.comp_pitch = 0.0
        self.comp_att_alpha = 0.98
        self.gyro_roll_integrated = 0.0
        self.gyro_pitch_integrated = 0.0

        # 4. Kalman Filters (X, Y, Z)
        self.kf_x = None
        self.kf_y = None
        self.kf_z = None

        # Noise Injection Standard Deviations (Realistic parameters)
        self.sigma_gps_xy = 0.05      # GPS horizontal: 5cm
        self.sigma_gps_z  = 0.10      # GPS vertical: 10cm
        self.sigma_acc    = 0.15      # Accelerometer: 0.15 m/s^2
        self.sigma_gyro   = 0.02      # Gyroscope: 0.02 rad/s
        self.sigma_baro   = 0.10      # Barometer: 10cm
        self.sigma_alt    = 0.015     # Altitude sensor: 1.5cm
        self.sigma_front  = 0.03      # Front distance: 3cm

        # Data logs for comparative analysis
        self.log_time = []
        # Ground Truths
        self.log_gt_x, self.log_gt_y, self.log_gt_z = [], [], []
        self.log_gt_vx, self.log_gt_vy, self.log_gt_vz = [], [], []
        self.log_gt_roll, self.log_gt_pitch, self.log_gt_yaw = [], [], []
        self.log_gt_alt, self.log_gt_front = [], []
        # Noisy Raw Sensors
        self.log_raw_gps_x, self.log_raw_gps_y, self.log_raw_gps_z = [], [], []
        self.log_raw_baro_z = []
        self.log_raw_acc_x, self.log_raw_acc_y, self.log_raw_acc_z = [], [], []
        self.log_raw_gyro_x, self.log_raw_gyro_y, self.log_raw_gyro_z = [], [], []
        self.log_raw_alt, self.log_raw_front = [], []
        # Filtered States
        self.log_filt_ema_x, self.log_filt_ema_y, self.log_filt_ema_z = [], [], []
        self.log_filt_comp_z = []
        self.log_filt_comp_roll, self.log_filt_comp_pitch = [], []
        self.log_filt_gyro_roll, self.log_filt_gyro_pitch = [], []
        self.log_filt_acc_roll, self.log_filt_acc_pitch = [], []
        self.log_filt_kf_x, self.log_filt_kf_y, self.log_filt_kf_z = [], [], []
        self.log_filt_ema_alt, self.log_filt_ema_front = [], []

    def _read_ground_truth(self, model, data):
        # Resolve body and joint addresses on the first step
        if self.drone_body_id is None:
            self.drone_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drone")
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "drone_joint")
            self.qpos_adr = model.jnt_qposadr[joint_id]
            self.qvel_adr = model.jnt_dofadr[joint_id]

        qa, va = self.qpos_adr, self.qvel_adr
        pos = data.qpos[qa:qa+3].copy()
        quat = data.qpos[qa+3:qa+7].copy()
        vel = data.qvel[va:va+3].copy()
        omega_world = data.qvel[va+3:va+6].copy()

        # Euler angles from quaternion [w, x, y, z]
        w, x, y, z = quat
        roll  = math.atan2(2.0*(w*x + y*z), 1.0 - 2.0*(x*x + y*y))
        sinp  = max(-1.0, min(1.0, 2.0*(w*y - z*x)))
        pitch = math.asin(sinp)
        yaw   = math.atan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))

        return pos, vel, quat, roll, pitch, yaw, omega_world

    def step(self, model, data):
        self.step_count += 1
        t = data.time

        # Read Ground Truth states
        pos_gt, vel_gt, quat_gt, roll_gt, pitch_gt, yaw_gt, omega_world = self._read_ground_truth(model, data)
        x_gt, y_gt, z_gt = pos_gt
        vx_gt, vy_gt, vz_gt = vel_gt

        # Initialize Kalman Filters at step 1
        if self.kf_x is None:
            self.kf_x = KalmanFilter1D(x_gt, vx_gt, q_acc=0.5, r_meas=self.sigma_gps_xy**2)
            self.kf_y = KalmanFilter1D(y_gt, vy_gt, q_acc=0.5, r_meas=self.sigma_gps_xy**2)
            self.kf_z = KalmanFilter1D(z_gt, vz_gt, q_acc=0.5, r_meas=self.sigma_baro**2)
            self.ema_x, self.ema_y, self.ema_z = x_gt, y_gt, z_gt
            self.comp_z = z_gt
            self.comp_vz = vz_gt
            self.comp_roll = roll_gt
            self.comp_pitch = pitch_gt
            self.gyro_roll_integrated = roll_gt
            self.gyro_pitch_integrated = pitch_gt
            self.ema_alt = z_gt - get_terrain_height(x_gt, y_gt)
            self.ema_front = 10.0
            self.prev_vel_world = vel_gt.copy()
            return

        # Calculate ground truth linear acceleration in world frame numerically
        acc_world_gt = (vel_gt - self.prev_vel_world) / self.dt
        self.prev_vel_world = vel_gt.copy()

        # Ground truth proper acceleration (sensed by accelerometer): acceleration minus gravity
        acc_proper_world = acc_world_gt + np.array([0.0, 0.0, 9.81])
        acc_proper_body = rotate_vector_by_quat(acc_proper_world, quat_conjugate(quat_gt))

        # Ground truth angular velocity in body frame
        omega_body = rotate_vector_by_quat(omega_world, quat_conjugate(quat_gt))

        # ============================================================
        #  1. Simulate Raw Sensors (Noise Injection)
        # ============================================================

        # A. GPS Sensor: Slow rate (10Hz), updates every 20 steps
        if self.step_count % self.gps_update_period == 0 or self.step_count == 1:
            self.last_gps_x = x_gt + np.random.normal(0.0, self.sigma_gps_xy)
            self.last_gps_y = y_gt + np.random.normal(0.0, self.sigma_gps_xy)
            self.last_gps_z = z_gt + np.random.normal(0.0, self.sigma_gps_z)
        
        gps_x, gps_y, gps_z = self.last_gps_x, self.last_gps_y, self.last_gps_z

        # B. IMU Accelerometer: High rate (200Hz), noise std dev = 0.15 m/s^2
        acc_x_raw = acc_proper_body[0] + np.random.normal(0.0, self.sigma_acc)
        acc_y_raw = acc_proper_body[1] + np.random.normal(0.0, self.sigma_acc)
        acc_z_raw = acc_proper_body[2] + np.random.normal(0.0, self.sigma_acc)
        raw_acc = np.array([acc_x_raw, acc_y_raw, acc_z_raw])

        # C. IMU Gyroscope: High rate (200Hz), noise std dev = 0.02 rad/s
        gyro_x_raw = omega_body[0] + np.random.normal(0.0, self.sigma_gyro)
        gyro_y_raw = omega_body[1] + np.random.normal(0.0, self.sigma_gyro)
        gyro_z_raw = omega_body[2] + np.random.normal(0.0, self.sigma_gyro)

        # D. Barometer Sensor: absolute altitude, noise std dev = 0.10m
        baro_z_raw = z_gt + np.random.normal(0.0, self.sigma_baro)

        # E. Altitude Sensor (Lidar): distance to ground beneath drone, noise std dev = 0.015m
        alt_gt = z_gt - get_terrain_height(x_gt, y_gt)
        alt_raw = alt_gt + np.random.normal(0.0, self.sigma_alt)

        # F. Front Distance Sensor: casts ray forward. Direction rotated by yaw to match heading
        heading = np.array([math.cos(yaw_gt), math.sin(yaw_gt), 0.0])
        # Position offset: slightly forward from center of drone
        ray_origin = pos_gt + heading * 0.15
        front_dist_gt = mujoco.mj_ray(model, data, ray_origin, heading, None, True, self.drone_body_id, None)
        if front_dist_gt < 0.0 or front_dist_gt > 10.0:
            front_dist_gt = 10.0  # Max sensor range limit
        
        front_raw = front_dist_gt + np.random.normal(0.0, self.sigma_front)

        # ============================================================
        #  2. Implement Estimation Filters
        # ============================================================

        # Rotate raw body acceleration back to estimated world frame using quaternion (for altitude prediction)
        # Using ground-truth quaternion here since we're analyzing altitude and position filtering
        acc_world_est = rotate_vector_by_quat(raw_acc, quat_gt) - np.array([0.0, 0.0, 9.81])

        # ─── A. EMA Filter (α = 0.15) ──────────────────────────────────
        self.ema_x = self.ema_alpha * gps_x + (1.0 - self.ema_alpha) * self.ema_x
        self.ema_y = self.ema_alpha * gps_y + (1.0 - self.ema_alpha) * self.ema_y
        self.ema_z = self.ema_alpha * baro_z_raw + (1.0 - self.ema_alpha) * self.ema_z
        self.ema_alt = self.ema_alpha * alt_raw + (1.0 - self.ema_alpha) * self.ema_alt
        self.ema_front = self.ema_alpha * front_raw + (1.0 - self.ema_alpha) * self.ema_front

        # ─── B. Complementary Altitude (Z) Filter ──────────────────────
        # Propagate state using vertical proper acceleration
        self.comp_vz += acc_world_est[2] * self.dt
        comp_z_pred = self.comp_z + self.comp_vz * self.dt + 0.5 * acc_world_est[2] * self.dt**2
        # Complementary update
        self.comp_z = self.comp_z_alpha * comp_z_pred + (1.0 - self.comp_z_alpha) * baro_z_raw
        # Correct vertical velocity to damp integration drift
        self.comp_vz += (1.0 - self.comp_z_alpha) * (baro_z_raw - comp_z_pred) / self.dt

        # ─── C. Complementary Attitude (Roll/Pitch) Filter ─────────────
        # Accelerometer estimate (assuming gravity vector is dominant acceleration)
        acc_roll_est = math.atan2(acc_y_raw, acc_z_raw)
        acc_pitch_est = math.atan2(-acc_x_raw, math.sqrt(acc_y_raw**2 + acc_z_raw**2))
        
        # Gyroscope raw integration
        self.gyro_roll_integrated += gyro_x_raw * self.dt
        self.gyro_pitch_integrated += gyro_y_raw * self.dt

        # Complementary update roll / pitch
        self.comp_roll = self.comp_att_alpha * (self.comp_roll + gyro_x_raw * self.dt) + (1.0 - self.comp_att_alpha) * acc_roll_est
        self.comp_pitch = self.comp_att_alpha * (self.comp_pitch + gyro_y_raw * self.dt) + (1.0 - self.comp_att_alpha) * acc_pitch_est

        # ─── D. Kalman Filters (X, Y, Z) ────────────────────────────────
        # Predict states with estimated world accelerations as control inputs
        self.kf_x.predict(acc_world_est[0], self.dt)
        self.kf_y.predict(acc_world_est[1], self.dt)
        self.kf_z.predict(acc_world_est[2], self.dt)

        # Update when GPS or Barometer measurements arrive
        if self.step_count % self.gps_update_period == 0:
            self.kf_x.update(gps_x)
            self.kf_y.update(gps_y)
        self.kf_z.update(baro_z_raw)

        # ============================================================
        #  3. Save Logs for Performance Comparison
        # ============================================================
        self.log_time.append(t)
        # Ground Truths
        self.log_gt_x.append(x_gt); self.log_gt_y.append(y_gt); self.log_gt_z.append(z_gt)
        self.log_gt_vx.append(vx_gt); self.log_gt_vy.append(vy_gt); self.log_gt_vz.append(vz_gt)
        self.log_gt_roll.append(roll_gt); self.log_gt_pitch.append(pitch_gt); self.log_gt_yaw.append(yaw_gt)
        self.log_gt_alt.append(alt_gt); self.log_gt_front.append(front_dist_gt)
        # Raw noisy
        self.log_raw_gps_x.append(gps_x); self.log_raw_gps_y.append(gps_y); self.log_raw_gps_z.append(gps_z)
        self.log_raw_baro_z.append(baro_z_raw)
        self.log_raw_acc_x.append(acc_x_raw); self.log_raw_acc_y.append(acc_y_raw); self.log_raw_acc_z.append(acc_z_raw)
        self.log_raw_gyro_x.append(gyro_x_raw); self.log_raw_gyro_y.append(gyro_y_raw); self.log_raw_gyro_z.append(gyro_z_raw)
        self.log_raw_alt.append(alt_raw); self.log_raw_front.append(front_raw)
        # Filtered values
        self.log_filt_ema_x.append(self.ema_x); self.log_filt_ema_y.append(self.ema_y); self.log_filt_ema_z.append(self.ema_z)
        self.log_filt_comp_z.append(self.comp_z)
        self.log_filt_comp_roll.append(self.comp_roll); self.log_filt_comp_pitch.append(self.comp_pitch)
        self.log_filt_gyro_roll.append(self.gyro_roll_integrated); self.log_filt_gyro_pitch.append(self.gyro_pitch_integrated)
        self.log_filt_acc_roll.append(acc_roll_est); self.log_filt_acc_pitch.append(acc_pitch_est)
        self.log_filt_kf_x.append(self.kf_x.x[0]); self.log_filt_kf_y.append(self.kf_y.x[0]); self.log_filt_kf_z.append(self.kf_z.x[0])
        self.log_filt_ema_alt.append(self.ema_alt); self.log_filt_ema_front.append(self.ema_front)

    # ============================================================
    #  Filter Comparison Metrics & Graph Generation
    # ============================================================

    def estimate_delay(self, filtered, ground_truth):
        f = np.array(filtered) - np.mean(filtered)
        gt = np.array(ground_truth) - np.mean(ground_truth)
        std_f = np.std(f)
        std_gt = np.std(gt)
        if std_f < 1e-6 or std_gt < 1e-6:
            return 0.0
        f /= std_f
        gt /= std_gt
        corr = np.correlate(f, gt, mode='full')
        lags = np.arange(-len(gt) + 1, len(f))
        idx = np.argmax(corr)
        lag = lags[idx]
        return max(0.0, lag * self.dt)

    def generate_plots_and_report(self):
        # Make sure we have logged data
        if not self.log_time:
            print("[SENSORS] No data logged to generate reports.")
            return

        print("\n[SENSORS] Generating filter comparison graphs and computing metrics...")

        time_arr = np.array(self.log_time)
        
        # Base arrays
        gt_z_base = np.array(self.log_gt_z)
        raw_z_base = np.array(self.log_raw_baro_z)
        ema_z_base = np.array(self.log_filt_ema_z)
        comp_z_base = np.array(self.log_filt_comp_z)
        kf_z_base = np.array(self.log_filt_kf_z)

        gt_x_base = np.array(self.log_gt_x)
        raw_x_base = np.array(self.log_raw_gps_x)
        ema_x_base = np.array(self.log_filt_ema_x)
        kf_x_base = np.array(self.log_filt_kf_x)

        gt_y_base = np.array(self.log_gt_y)
        raw_y_base = np.array(self.log_raw_gps_y)
        ema_y_base = np.array(self.log_filt_ema_y)
        kf_y_base = np.array(self.log_filt_kf_y)

        gt_roll_base = np.array(self.log_gt_roll)
        acc_roll_base = np.array(self.log_filt_acc_roll)
        gyro_roll_base = np.array(self.log_filt_gyro_roll)
        comp_roll_base = np.array(self.log_filt_comp_roll)

        gt_alt_base = np.array(self.log_gt_alt)
        raw_alt_base = np.array(self.log_raw_alt)
        ema_alt_base = np.array(self.log_filt_ema_alt)

        gt_front_base = np.array(self.log_gt_front)
        raw_front_base = np.array(self.log_raw_front)
        ema_front_base = np.array(self.log_filt_ema_front)

        def smooth_signal(x, window_len=51):
            x = np.array(x)
            n = len(x)
            if n < 5:
                return x
            w = min(window_len, n)
            if w % 2 == 0:
                w = max(3, w - 1)
            if w < 5:
                return x
            try:
                from scipy.signal import savgol_filter
                return savgol_filter(x, w, polyorder=2)
            except Exception:
                kernel = np.ones(w) / w
                return np.convolve(x, kernel, mode='same')

        for is_noiseless in [False, True]:
            if is_noiseless:
                plot_path = "sensor_filtering_noiseless.png"
                # Smooth all noisy/filtered arrays
                gt_z = gt_z_base
                raw_z = smooth_signal(raw_z_base)
                ema_z = smooth_signal(ema_z_base)
                comp_z = smooth_signal(comp_z_base)
                kf_z = smooth_signal(kf_z_base)

                gt_x = gt_x_base
                raw_x = smooth_signal(raw_x_base)
                ema_x = smooth_signal(ema_x_base)
                kf_x = smooth_signal(kf_x_base)

                gt_y = gt_y_base
                raw_y = smooth_signal(raw_y_base)
                ema_y = smooth_signal(ema_y_base)
                kf_y = smooth_signal(kf_y_base)

                gt_roll = gt_roll_base
                acc_roll = smooth_signal(acc_roll_base)
                gyro_roll = smooth_signal(gyro_roll_base)
                comp_roll = smooth_signal(comp_roll_base)

                gt_alt = gt_alt_base
                raw_alt = smooth_signal(raw_alt_base)
                ema_alt = smooth_signal(ema_alt_base)

                gt_front = gt_front_base
                raw_front = smooth_signal(raw_front_base)
                ema_front = smooth_signal(ema_front_base)
            else:
                plot_path = "sensor_filtering_analysis.png"
                gt_z = gt_z_base
                raw_z = raw_z_base
                ema_z = ema_z_base
                comp_z = comp_z_base
                kf_z = kf_z_base

                gt_x = gt_x_base
                raw_x = raw_x_base
                ema_x = ema_x_base
                kf_x = kf_x_base

                gt_y = gt_y_base
                raw_y = raw_y_base
                ema_y = ema_y_base
                kf_y = kf_y_base

                gt_roll = gt_roll_base
                acc_roll = acc_roll_base
                gyro_roll = gyro_roll_base
                comp_roll = comp_roll_base

                gt_alt = gt_alt_base
                raw_alt = raw_alt_base
                ema_alt = ema_alt_base

                gt_front = gt_front_base
                raw_front = raw_front_base
                ema_front = ema_front_base

            # ─── Compute Comparison Metrics ─────────────────────────────────────
            metrics = {}
            
            def calc_metrics(raw, filtered, gt):
                err_raw = raw - gt
                err_filt = filtered - gt
                rmse_raw = np.sqrt(np.mean(err_raw**2))
                rmse_filt = np.sqrt(np.mean(err_filt**2))
                mae_raw = np.mean(np.abs(err_raw))
                mae_filt = np.mean(np.abs(err_filt))
                std_raw = np.std(err_raw)
                std_filt = np.std(err_filt)
                noise_red = (1.0 - (std_filt / std_raw)) * 100.0 if std_raw > 1e-6 else 0.0
                delay = self.estimate_delay(filtered, gt)
                return rmse_filt, noise_red, mae_filt, delay

            # A. Altitude (Z) Filters
            metrics["Z_Raw"] = (np.sqrt(np.mean((raw_z - gt_z)**2)), 0.0, np.mean(np.abs(raw_z - gt_z)), 0.0)
            metrics["Z_EMA"] = calc_metrics(raw_z, ema_z, gt_z)
            metrics["Z_Comp"] = calc_metrics(raw_z, comp_z, gt_z)
            metrics["Z_Kalman"] = calc_metrics(raw_z, kf_z, gt_z)

            # B. Position X Filters
            metrics["X_Raw"] = (np.sqrt(np.mean((raw_x - gt_x)**2)), 0.0, np.mean(np.abs(raw_x - gt_x)), 0.0)
            metrics["X_EMA"] = calc_metrics(raw_x, ema_x, gt_x)
            metrics["X_Kalman"] = calc_metrics(raw_x, kf_x, gt_x)

            # C. Roll Angle estimation Filters
            metrics["Roll_Acc"] = (np.sqrt(np.mean((acc_roll - gt_roll)**2)), 0.0, np.mean(np.abs(acc_roll - gt_roll)), 0.0)
            metrics["Roll_Gyro"] = (np.sqrt(np.mean((gyro_roll - gt_roll)**2)), 0.0, np.mean(np.abs(gyro_roll - gt_roll)), 0.0)
            metrics["Roll_Comp"] = calc_metrics(acc_roll, comp_roll, gt_roll)

            # Only print table to console for the actual analysis run (False)
            if not is_noiseless:
                print("\n" + "="*60)
                print("          SENSOR FILTERING SUITE COMPARATIVE ANALYSIS")
                print("="*60)
                print(f"{'State & Filter':<20} | {'RMSE (m/rad)':<12} | {'Noise Red. (%)':<15} | {'Delay (s)':<10}")
                print("-"*60)
                for key in ["Z_Raw", "Z_EMA", "Z_Comp", "Z_Kalman", "X_Raw", "X_EMA", "X_Kalman", "Roll_Acc", "Roll_Gyro", "Roll_Comp"]:
                    r, nr, mae, d = metrics[key]
                    print(f"{key:<20} | {r:.4f}       | {nr:.2f}%          | {d:.4f}")
                print("="*60)

            # ─── Plot Figures ──────────────────────────────────────────────────
            plt.style.use('default')
            fig = plt.figure(figsize=(15, 10))
            title_text = "Software Sensor Simulation & Filter Comparison (Noiseless)" if is_noiseless else "Software Sensor Simulation & Filter Comparison"
            fig.suptitle(title_text, fontsize=16, color='black', fontweight='bold')

            # Subplot 1: Altitude (Z Position) Tracking
            ax1 = plt.subplot(3, 2, 1)
            ax1.plot(time_arr, gt_z, label='Ground Truth', color='black', linewidth=2)
            if is_noiseless:
                ax1.plot(time_arr, raw_z, label='Raw (Baro)', color='red', alpha=0.5, linewidth=1.5)
            else:
                ax1.plot(time_arr, raw_z, label='Raw (Baro)', color='red', alpha=0.3, linestyle='None', marker='.', markersize=2)
            ax1.plot(time_arr, ema_z, label='EMA Filter', color='orange', alpha=0.8)
            ax1.plot(time_arr, comp_z, label='Complementary Filter', color='purple', alpha=0.8)
            ax1.plot(time_arr, kf_z, label='Kalman Filter', color='blue', alpha=0.9, linewidth=1.5)
            ax1.set_title("Altitude (Z) Position Tracking", color='black')
            ax1.set_ylabel("Z Position (m)")
            ax1.grid(color='#e0e0e0')
            ax1.legend(fontsize=6, loc='upper right', framealpha=0.7, handlelength=1.0, borderpad=0.3, labelspacing=0.3)

            # Subplot 2: Attitude estimation (Roll)
            ax2 = plt.subplot(3, 2, 2)
            ax2.plot(time_arr, np.degrees(gt_roll), label='Ground Truth', color='black', linewidth=2)
            ax2.plot(time_arr, np.degrees(acc_roll), label='Raw (Acc estimate)', color='red', alpha=0.3)
            ax2.plot(time_arr, np.degrees(gyro_roll), label='Gyro integration (drift)', color='orange', alpha=0.5)
            ax2.plot(time_arr, np.degrees(comp_roll), label='Complementary Filter', color='blue', alpha=0.9, linewidth=1.5)
            ax2.set_title("Attitude (Roll) Estimation", color='black')
            ax2.set_ylabel("Roll Angle (deg)")
            ax2.grid(color='#e0e0e0')
            ax2.legend(fontsize=6, loc='upper right', framealpha=0.7, handlelength=1.0, borderpad=0.3, labelspacing=0.3)

            # Subplot 3: 2D Horizontal Position Tracking (X vs Y)
            ax3 = plt.subplot(3, 2, 3)
            ax3.plot(gt_x, gt_y, label='Ground Truth', color='black', linewidth=2)
            if is_noiseless:
                ax3.plot(raw_x, raw_y, label='Raw (GPS)', color='red', alpha=0.5, linewidth=1.5)
            else:
                ax3.scatter(raw_x, raw_y, label='Raw (GPS)', color='red', s=1, alpha=0.3)
            ax3.plot(ema_x, ema_y, label='EMA Filter', color='orange', alpha=0.7)
            ax3.plot(kf_x, kf_y, label='Kalman Filter', color='blue', alpha=0.9, linewidth=1.5)
            ax3.set_title("Horizontal Trajectory Tracking (X vs Y)", color='black')
            ax3.set_xlabel("X Position (m)")
            ax3.set_ylabel("Y Position (m)")
            ax3.grid(color='#e0e0e0')
            ax3.legend(fontsize=6, loc='lower right', framealpha=0.7, handlelength=1.0, borderpad=0.3, labelspacing=0.3)

            # Subplot 4: Rangefinders (Altimeter & Front sensor)
            ax4 = plt.subplot(3, 2, 4)
            ax4.plot(time_arr, gt_alt, label='Ground Truth Alt', color='black', linewidth=2)
            ax4.plot(time_arr, raw_alt, label='Raw Alt Sensor', color='red', alpha=0.3)
            ax4.plot(time_arr, ema_alt, label='EMA Alt Sensor', color='orange', alpha=0.8)
            ax4.plot(time_arr, gt_front, label='Ground Truth Front', color='purple', linewidth=1.5)
            ax4.plot(time_arr, ema_front, label='EMA Front Sensor', color='blue', alpha=0.8)
            ax4.set_title("Rangefinders & Distance Sensors", color='black')
            ax4.set_ylabel("Distance (m)")
            ax4.grid(color='#e0e0e0')
            ax4.legend(fontsize=6, loc='upper right', framealpha=0.7, handlelength=1.0, borderpad=0.3, labelspacing=0.3)

            # Subplot 5: Bar Chart comparing RMSE and Delay for Z filters
            ax5 = plt.subplot(3, 2, 5)
            filter_names = ['Raw (Baro)', 'EMA', 'Complementary', 'Kalman']
            rmse_vals = [metrics["Z_Raw"][0], metrics["Z_EMA"][0], metrics["Z_Comp"][0], metrics["Z_Kalman"][0]]
            delay_vals = [metrics["Z_Raw"][3], metrics["Z_EMA"][3], metrics["Z_Comp"][3], metrics["Z_Kalman"][3]]

            x = np.arange(len(filter_names))
            width = 0.35

            rects1 = ax5.bar(x - width/2, rmse_vals, width, label='RMSE (m)', color='red')
            rects2 = ax5.bar(x + width/2, delay_vals, width, label='Delay (s)', color='blue')
            ax5.set_title("Filter Performance Metrics (Z Axis)", color='black')
            ax5.set_xticks(x)
            ax5.set_xticklabels(filter_names)
            ax5.grid(color='#e0e0e0', linestyle='--')
            ax5.legend(fontsize=6, loc='upper right', framealpha=0.7, handlelength=1.0, borderpad=0.3, labelspacing=0.3)

            # Subplot 6: Summary Table
            ax6 = plt.subplot(3, 2, 6)
            ax6.axis('off')
            
            table_data = [
                ["Filter/Sensor", "RMSE (m/rad)", "Noise Red. (%)", "Tracking Error (MAE)", "Delay (s)"],
                ["Z Baro (Raw)", f"{metrics['Z_Raw'][0]:.4f}", "0.00%", f"{metrics['Z_Raw'][2]:.4f}", "0.000"],
                ["Z EMA", f"{metrics['Z_EMA'][0]:.4f}", f"{metrics['Z_EMA'][1]:.2f}%", f"{metrics['Z_EMA'][2]:.4f}", f"{metrics['Z_EMA'][3]:.3f}"],
                ["Z Complementary", f"{metrics['Z_Comp'][0]:.4f}", f"{metrics['Z_Comp'][1]:.2f}%", f"{metrics['Z_Comp'][2]:.4f}", f"{metrics['Z_Comp'][3]:.3f}"],
                ["Z Kalman", f"{metrics['Z_Kalman'][0]:.4f}", f"{metrics['Z_Kalman'][1]:.2f}%", f"{metrics['Z_Kalman'][2]:.4f}", f"{metrics['Z_Kalman'][3]:.3f}"],
                ["X GPS (Raw)", f"{metrics['X_Raw'][0]:.4f}", "0.00%", f"{metrics['X_Raw'][2]:.4f}", "0.000"],
                ["X EMA", f"{metrics['X_EMA'][0]:.4f}", f"{metrics['X_EMA'][1]:.2f}%", f"{metrics['X_EMA'][2]:.4f}", f"{metrics['X_EMA'][3]:.3f}"],
                ["X Kalman", f"{metrics['X_Kalman'][0]:.4f}", f"{metrics['X_Kalman'][1]:.2f}%", f"{metrics['X_Kalman'][2]:.4f}", f"{metrics['X_Kalman'][3]:.3f}"],
                ["Roll Acc (Raw)", f"{metrics['Roll_Acc'][0]:.4f}", "0.00%", f"{metrics['Roll_Acc'][2]:.4f}", "0.000"],
                ["Roll Comp", f"{metrics['Roll_Comp'][0]:.4f}", f"{metrics['Roll_Comp'][1]:.2f}%", f"{metrics['Roll_Comp'][2]:.4f}", f"{metrics['Roll_Comp'][3]:.3f}"]
            ]

            table = ax6.table(cellText=table_data, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(7)
            table.scale(1.1, 1.25)
            
            # Color table cells
            for i in range(len(table_data)):
                for j in range(5):
                    cell = table[i, j]
                    if i == 0:
                        cell.set_facecolor('#f0f0f0')
                        cell.set_text_props(color='black', fontweight='bold')
                    else:
                        cell.set_facecolor('white')
                        cell.set_text_props(color='black')

            plt.tight_layout()
            plt.savefig(plot_path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"[SENSORS] Comparative analysis plot saved to {os.path.abspath(plot_path)}")

            # Save metrics to json file for the report generator (only for the actual noisy run)
            if not is_noiseless:
                try:
                    import json
                    metrics_dict = {
                        "Z_Raw": list(metrics["Z_Raw"]),
                        "Z_EMA": list(metrics["Z_EMA"]),
                        "Z_Comp": list(metrics["Z_Comp"]),
                        "Z_Kalman": list(metrics["Z_Kalman"]),
                        "X_Raw": list(metrics["X_Raw"]),
                        "X_EMA": list(metrics["X_EMA"]),
                        "X_Kalman": list(metrics["X_Kalman"]),
                        "Roll_Acc": list(metrics["Roll_Acc"]),
                        "Roll_Gyro": list(metrics["Roll_Gyro"]),
                        "Roll_Comp": list(metrics["Roll_Comp"])
                    }
                    with open("sensor_metrics.json", "w") as f:
                        json.dump(metrics_dict, f, indent=4)
                    print(f"[SENSORS] Saved metrics to {os.path.abspath('sensor_metrics.json')}\n")
                except Exception as e:
                    print(f"[SENSORS] Error saving metrics to json: {e}\n")
