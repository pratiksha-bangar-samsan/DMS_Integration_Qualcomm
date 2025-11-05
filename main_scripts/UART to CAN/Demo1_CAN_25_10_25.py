import cv2
import time
import numpy as np
import os
import argparse
import mediapipe as mp
from scipy.spatial import distance as dist
from playsound import playsound
import threading
import configparser
import logging
import subprocess
from datetime import datetime

# --- CAN LIBRARY IMPORT ---
try:
    import can
    # Set logging level for the can library to suppress verbose output
    logging.getLogger('can').setLevel(logging.WARNING)
except ImportError:
    print("Warning: 'python-can' library not found. CAN functionality will be disabled.")
    can = None
# --------------------------

# ==================== Functions to be used in the Class ======================
def eye_aspect_ratio(eye_landmarks):
    """Calculates the Eye Aspect Ratio (EAR)."""
    p2_p6 = dist.euclidean(np.array([eye_landmarks[1].x, eye_landmarks[1].y]), np.array([eye_landmarks[5].x, eye_landmarks[5].y]))
    p3_p5 = dist.euclidean(np.array([eye_landmarks[2].x, eye_landmarks[2].y]), np.array([eye_landmarks[4].x, eye_landmarks[4].y]))
    p1_p4 = dist.euclidean(np.array([eye_landmarks[0].x, eye_landmarks[0].y]), np.array([eye_landmarks[3].x, eye_landmarks[3].y]))
    return (p2_p6 + p3_p5) / (2.0 * p1_p4)

def mouth_aspect_ratio(landmarks, MOUTH_INNER_TOP, MOUTH_INNER_BOTTOM, MOUTH_OUTER_HORIZONTAL):
    """Calculates the Mouth Aspect Ratio (MAR)."""
    A = dist.euclidean(np.array([landmarks[MOUTH_INNER_TOP[0]].x, landmarks[MOUTH_INNER_TOP[0]].y]),
                       np.array([landmarks[MOUTH_INNER_BOTTOM[0]].x, landmarks[MOUTH_INNER_BOTTOM[0]].y]))
    B = dist.euclidean(np.array([landmarks[MOUTH_INNER_TOP[1]].x, landmarks[MOUTH_INNER_TOP[1]].y]),
                       np.array([landmarks[MOUTH_INNER_BOTTOM[1]].x, landmarks[MOUTH_INNER_BOTTOM[1]].y]))
    C = dist.euclidean(np.array([landmarks[MOUTH_INNER_TOP[2]].x, landmarks[MOUTH_INNER_TOP[2]].y]),
                       np.array([landmarks[MOUTH_INNER_BOTTOM[2]].x, landmarks[MOUTH_INNER_BOTTOM[2]].y]))
    D = dist.euclidean(np.array([landmarks[MOUTH_OUTER_HORIZONTAL[0]].x, landmarks[MOUTH_OUTER_HORIZONTAL[0]].y]),
                       np.array([landmarks[MOUTH_OUTER_HORIZONTAL[1]].x, landmarks[MOUTH_OUTER_HORIZONTAL[1]].y]))
    mar = (A + B + C) / (3.0 * D)
    return mar

# ==================== EdgetensorDMMMonitor Class ======================
class EdgetensorDMMMonitor:
    def __init__(self, roi_position, config_file='config.ini'):
        self.config = self.load_config(config_file)

        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Screen layout and thresholds from config
        self.video_width = 800
        self.panel_width = 490
        self.display_height = 750
        self.video_height = self.display_height
        self.display_width = self.video_width + self.panel_width

        # Load thresholds and timers from config
        self.EAR_THRESHOLD = self.config.getfloat('THRESHOLDS', 'ear_threshold')
        self.YAWN_THRESHOLD = self.config.getfloat('THRESHOLDS', 'yawn_threshold')
        self.BRIGHTNESS_THRESHOLD = self.config.getfloat('THRESHOLDS', 'brightness_threshold')
        self.COLOR_VAR_THRESHOLD = self.config.getfloat('THRESHOLDS', 'color_var_threshold')
        self.LEFT_THRESHOLD = self.config.getfloat('THRESHOLDS', 'left_threshold')
        self.RIGHT_THRESHOLD = self.config.getfloat('THRESHOLDS', 'right_threshold')
        self.UP_THRESHOLD = self.config.getfloat('THRESHOLDS', 'up_threshold')
        self.DOWN_THRESHOLD = self.config.getfloat('THRESHOLDS', 'down_threshold')

        self.YAWN_TIME_THRESHOLD = self.config.getfloat('TIMERS', 'yawn_time')
        self.FATIGUE_TIME_THRESHOLD = self.config.getfloat('TIMERS', 'fatigue_time')
        self.DISTRACTION_TIME_THRESHOLD = self.config.getfloat('TIMERS', 'distraction_time')
        self.FACE_OBSTRUCTION_TIME = self.config.getfloat('TIMERS', 'face_obstruction_time')
        self.CAMERA_OBSTRUCTION_TIME = self.config.getfloat('TIMERS', 'camera_obstruction_time')
        self.SMOKING_TIME = self.config.getfloat('TIMERS', 'smoking_time')
        self.PHONE_USE_TIME = self.config.getfloat('TIMERS', 'phone_use_time')
        self.EATING_DRINKING_TIME = self.config.getfloat('TIMERS', 'eating_drinking_time')
        self.ALERT_DELAY_SECONDS = self.config.getfloat('TIMERS', 'alert_delay_seconds')

        # Facial landmark indices
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.MOUTH_INNER_TOP = [13, 308, 324]
        self.MOUTH_INNER_BOTTOM = [14, 84, 314]
        self.MOUTH_OUTER_HORIZONTAL = [61, 291]
        self.NOSE_LANDMARKS = [4, 6, 195]

        # MediaPipe FaceMesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.roi_position = roi_position
        self.roi_x1, self.roi_y1, self.roi_x2, self.roi_y2 = self.get_roi_coordinates(roi_position)

        self.icons_path = "/home/jetson/DMS_new/Detection/icons"
        self.audio_path = "/home/jetson/obstruction_gpu/audio_alert" # New Audio Path
        self.icons = {}
        self.load_icons()

        # Voice alerts mapping
        self.voice_alerts = {
            'yawning': 'yawning.mp3',
            'fatigue': 'fatigue_HoldSterring.mp3',
            'camera_obstruction': 'obstruction.mp3',
            'face_obstruction': 'face_obstruction.mp3',
            'smoking': 'smoking.mp3',
            'phone_uses': 'phone_uses.mp3',
            'eating_drinking': 'eating_drinking.mp3',
            'seatbelt_off': 'seatbelt_off.mp3',

        }

        self.events = {
            'smoking': {'active': False, 'timer': 0.0, 'threshold': self.SMOKING_TIME, 'alert_time': 0.0},
            'yawning': {'active': False, 'timer': 0.0, 'threshold': self.YAWN_TIME_THRESHOLD, 'alert_time': 0.0},
            'eating_drinking': {'active': False, 'timer': 0.0, 'threshold': self.EATING_DRINKING_TIME, 'alert_time': 0.0},
            'phone_uses': {'active': False, 'timer': 0.0, 'threshold': self.PHONE_USE_TIME, 'alert_time': 0.0},
            'wearing_mask': {'active': False, 'timer': 0.0, 'threshold': None, 'alert_time': 0.0},
            'seatbelt_off': {'active': False, 'timer': 0.0, 'threshold': None, 'alert_time': 0.0},
            'wearing_seatbelt': {'active': False, 'timer': 0.0, 'threshold': None, 'alert_time': 0.0},
            'fatigue': {'active': False, 'timer': 0.0, 'threshold': self.FATIGUE_TIME_THRESHOLD, 'alert_time': 0.0},
            'distraction': {'active': False, 'timer': 0.0, 'threshold': self.DISTRACTION_TIME_THRESHOLD, 'alert_time': 0.0},
            'camera_obstruction': {'active': False, 'timer': 0.0, 'threshold': self.CAMERA_OBSTRUCTION_TIME, 'alert_time': 0.0},
            'face_obstruction': {'active': False, 'timer': 0.0, 'threshold': self.FACE_OBSTRUCTION_TIME, 'alert_time': 0.0},
            'left_eye_closed': {'active': False, 'timer': 0.0, 'threshold': None, 'alert_time': 0.0},
            'right_eye_closed': {'active': False, 'timer': 0.0, 'threshold': None, 'alert_time': 0.0},
        }

        self.direction_timers = {'left': 0.0, 'right': 0.0, 'up': 0.0, 'down': 0.0}
        self.distraction_start_time = None
        self.fatigue_start_time = None
        self.yawning_start_time = None
        self.smoking_start_time = None
        self.phone_use_start_time = None
        self.eating_drinking_start_time = None
        self.camera_obstructed_start_time = None
        self.face_obstructed_start_time = None

        self.gaze_text = "No Face"
        self.metrics = {'fps': 0, 'frame_processing_time': 0, 'faces_detected': 0}
        self.head_pose_yaw = 0.0
        self.head_pose_pitch = 0.0
        self.head_pose_roll = 0.0
        self.last_time = time.time()
        self.face_not_detected = False

        # --- CAN Bus Initialization (Configurable: PCAN, SocketCAN, or CAN-to-UART SLCAN/Serial) ---
        self.can_bus = None
        self.can_enabled = False
        self.can_interface = 'auto'
        self.serial_port = None
        self.baudrate = 115200
        self.serial_handle = None  # For UART/SLCAN persistent connection

        if can is not None:
            # Read CAN config
            can_interface = 'auto'
            serial_port = None
            baudrate = 115200  # Default to working baudrate from your test code
            try:
                if self.config and self.config.has_section('CAN'):
                    can_interface = self.config.get('CAN', 'can_interface', fallback='auto').lower()
                    serial_port = self.config.get('CAN', 'serial_port', fallback=None)
                    baudrate = self.config.getint('CAN', 'baudrate', fallback=115200)
            except Exception:
                can_interface = 'auto'
                serial_port = None
                baudrate = 115200
            # Store for use in send_can_message
            self.can_interface = can_interface
            self.serial_port = serial_port
            self.baudrate = baudrate

            # Helper for UART: send 'S6' to set CAN speed, then 'O' to open CAN channel (SLCAN protocol)
            def slcan_open(serial_path):
                import serial as pyserial
                try:
                    self.serial_handle = pyserial.Serial(serial_path, baudrate=self.baudrate, timeout=1)
                    print(f"Opened serial port {serial_path} at {self.baudrate} baud.")
                    
                    # Set CAN bus speed (matching your working code)
                    self.serial_handle.write(b'S6\r')  # SLCAN: S6 = 500kbit/sec CAN bus
                    self.serial_handle.flush()
                    time.sleep(0.1)
                    resp = self.serial_handle.read(10)
                    print(f"Sent S6 (CAN speed). Response: {resp}")
                    
                    # Open CAN bus (matching your working code)
                    self.serial_handle.write(b'O\r')  # SLCAN: O = open CAN bus
                    self.serial_handle.flush()
                    time.sleep(0.1)
                    resp = self.serial_handle.read(10)
                    print(f"Sent O (open CAN bus). Response: {resp}")
                    
                except Exception as e:
                    print(f"Failed to send 'S6'/'O' to {serial_path}: {e}")

            # Initialize CAN bus only once to avoid repeated open/close cycles
            can_initialized = False
            
            # Initialize CAN bus - UART/SLCAN only (PCAN functionality removed)
            try:
                if self.can_interface == 'uart':
                    # UART/SLCAN initialization with "O" command activation
                    slcan_channel = self.serial_port or '/dev/ttyTHS1'
                    slcan_open(slcan_channel)  # This sends S6 and O commands
                    try:
                        self.can_bus = can.interface.Bus(channel=slcan_channel, interface='slcan', bitrate=self.baudrate)
                        self.can_enabled = True
                        can_initialized = True
                        print(f"CAN Bus initialized using SLCAN on {slcan_channel} at {self.baudrate} baud.")
                    except Exception:
                        self.can_bus = can.interface.Bus(channel=slcan_channel, interface='serial', bitrate=self.baudrate)
                        self.can_enabled = True
                        can_initialized = True
                        print(f"CAN Bus initialized using serial backend on {slcan_channel} at {self.baudrate} baud.")
                elif self.can_interface == 'socketcan':
                    # SocketCAN initialization (no SLCAN commands needed)
                    self.can_bus = can.interface.Bus(channel='can0', interface='socketcan')
                    self.can_enabled = True
                    can_initialized = True
                    print("CAN Bus initialized using SocketCAN (can0).")
                else:  # auto mode: try SocketCAN first, then UART/SLCAN
                    try:
                        self.can_bus = can.interface.Bus(channel='can0', interface='socketcan')
                        self.can_enabled = True
                        can_initialized = True
                        print("CAN Bus initialized using SocketCAN (can0).")
                    except Exception as e_socket:
                        # Fallback to UART/SLCAN with "O" command activation
                        slcan_channel = self.serial_port or '/dev/ttyTHS1'
                        slcan_open(slcan_channel)  # This sends S6 and O commands
                        try:
                            self.can_bus = can.interface.Bus(channel=slcan_channel, interface='slcan', bitrate=self.baudrate)
                            self.can_enabled = True
                            can_initialized = True
                            print(f"CAN Bus initialized using SLCAN on {slcan_channel} at {self.baudrate} baud.")
                        except Exception:
                            try:
                                self.can_bus = can.interface.Bus(channel=slcan_channel, interface='serial', bitrate=self.baudrate)
                                self.can_enabled = True
                                can_initialized = True
                                print(f"CAN Bus initialized using serial backend on {slcan_channel} at {self.baudrate} baud.")
                            except Exception as e_slcan:
                                print(f"Error initializing CAN bus (SocketCAN: {e_socket}; Serial/SLCAN: {e_slcan}). CAN functionality disabled.")
                                self.can_enabled = False
                                can_initialized = False
            except Exception as e:
                print(f"Error initializing CAN bus (forced interface {self.can_interface}): {e}")
                self.can_enabled = False
        else:
            print("Warning: 'python-can' library not found. CAN functionality will be disabled.")
        # ----------------------------------------------------

    def load_config(self, config_file):
        config = configparser.ConfigParser()
        if not os.path.exists(config_file):
            print(f"Error: Config file '{config_file}' not found.")
            exit(1)
        config.read(config_file)
        return config
    
    def get_roi_coordinates(self, roi_position):
        try:
            section_name = f'ROI_{roi_position}'
            if not self.config.has_section(section_name):
                section_name = f"ROI_{self.config.get('DEFAULT', 'active_roi')}"

            x1 = self.config.getint(section_name, 'x1')
            y1 = self.config.getint(section_name, 'y1')
            x2 = self.config.getint(section_name, 'x2')
            y2 = self.config.getint(section_name, 'y2')
            return x1, y1, x2, y2
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            print(f"Error reading ROI configuration: {e}. Using hardcoded default.")
            return 200, 150, 600, 600

    def play_alert(self, event_name):
        """Plays the audio alert for a given event in a separate thread."""
        def play_sound_async():
            if event_name in self.voice_alerts:
                audio_file = os.path.join(self.audio_path, self.voice_alerts[event_name])
                if os.path.exists(audio_file):
                    try:
                        playsound(audio_file)
                    except Exception as e:
                        print(f"Error playing sound for {event_name}: {e}")
                else:
                    print(f"Warning: Audio file not found for event '{event_name}' at {audio_file}")
            else:
                print(f"Warning: No voice alert defined for event '{event_name}'.")

        thread = threading.Thread(target=play_sound_async)
        thread.daemon = True
        thread.start()

    def load_icons(self):
        icon_mappings = {
            'smoking': ['Smoking/smoke.jpg'], 'yawning': ['yawning/yawn.png'],
            'eating_drinking': ['eating_drinking/food.jpg'], 'phone_uses': ['phone use/phone.png'],
            'wearing_mask': ['mask/mask.png'], 'seatbelt_off': ['noseatbelt/nobelt.jpg'],
            'wearing_seatbelt': ['seatbelt/seatbelt.jpg'], 'distraction': ['distraction/distraction.png'],
            'camera_obstruction': ['camera_obstraction/obstraction.jpeg'],
            'face_obstruction': ['face_obstruction/face.png'],
            'fatigue': ['fatigue/drowsy.png'],
        }
        icon_size = (60, 60)
        for icon_name, possible_paths in icon_mappings.items():
            icon_loaded = False
            for path in possible_paths:
                full_path = os.path.join(self.icons_path, path)
                if os.path.exists(full_path):
                    try:
                        icon = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
                        if icon is not None:
                            if len(icon.shape) == 2:
                                icon = cv2.cvtColor(icon, cv2.COLOR_GRAY2BGR)
                            icon = cv2.resize(icon, icon_size)
                            self.icons[icon_name] = icon
                            icon_loaded = True
                            break
                    except Exception as e:
                        print(f"Error loading icon {path}: {e}")
            if not icon_loaded:
                print(f"Warning: Could not load icon for {icon_name}")
                placeholder = np.zeros((*icon_size, 3), dtype=np.uint8)
                placeholder[:] = (128, 128, 128)
                cv2.putText(placeholder, icon_name[:3].upper(), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                self.icons[icon_name] = placeholder

    def draw_icon(self, frame, icon_name, x, y, bg_color):
        if icon_name in self.icons:
            icon_original = self.icons[icon_name]
            h, w = icon_original.shape[:2]
            icon_to_draw = icon_original.copy()
            if bg_color == (0, 255, 0):
                if icon_to_draw.shape[2] == 4:
                    mask = icon_to_draw[:, :, 3] > 0
                    green_overlay = np.zeros_like(icon_to_draw, dtype=np.uint8)
                    green_overlay[mask] = (0, 255, 0, 255)
                    icon_to_draw = cv2.addWeighted(icon_to_draw, 0.5, green_overlay, 0.5, 0)
                else:
                    gray = cv2.cvtColor(icon_to_draw, cv2.COLOR_BGR2GRAY)
                    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
                    icon_to_draw[mask > 0] = (0, 255, 0)
            cv2.rectangle(frame, (x - 5, y - 5), (x + w + 5, y + h + 5), bg_color, -1)
            cv2.rectangle(frame, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 0, 0), 1)
            frame_h, frame_w = frame.shape[:2]
            if y + h > frame_h or x + w > frame_w:
                pass
            if len(icon_to_draw.shape) == 3 and icon_to_draw.shape[2] == 4:
                alpha_channel = icon_to_draw[:, :, 3] / 255.0
                for c in range(3):
                    frame[y:y + h, x:x + w, c] = frame[y:y + h, x:x + w, c] * (1 - alpha_channel) + icon_to_draw[:, :, c] * alpha_channel
            else:
                if len(icon_to_draw.shape) == 2:
                    icon_to_draw = cv2.cvtColor(icon_to_draw, cv2.COLOR_GRAY2BGR)
                frame[y:y + h, x:x + w] = icon_to_draw

    def detect_eye_closure(self, landmarks):
        left_eye_landmarks = [landmarks[i] for i in self.LEFT_EYE]
        right_eye_landmarks = [landmarks[i] for i in self.RIGHT_EYE]
        left_ear = eye_aspect_ratio(left_eye_landmarks)
        right_ear = eye_aspect_ratio(right_eye_landmarks)
        left_eye_closed = left_ear < self.EAR_THRESHOLD
        right_eye_closed = right_ear < self.EAR_THRESHOLD
        return left_eye_closed, right_eye_closed

    def detect_yawn(self, landmarks):
        mar = mouth_aspect_ratio(landmarks, self.MOUTH_INNER_TOP, self.MOUTH_INNER_BOTTOM, self.MOUTH_OUTER_HORIZONTAL)
        yawn_detected = mar > self.YAWN_THRESHOLD
        yawn_confidence = mar / (self.YAWN_THRESHOLD * 1.5) if yawn_detected else 0
        yawn_confidence = min(1.0, yawn_confidence)
        return yawn_detected, yawn_confidence

    def detect_head_pose(self, landmarks):
        image_points = np.array([
            (landmarks[33].x * self.video_width, landmarks[33].y * self.video_height),
            (landmarks[263].x * self.video_width, landmarks[263].y * self.video_height),
            (landmarks[1].x * self.video_width, landmarks[1].y * self.video_height),
            (landmarks[61].x * self.video_width, landmarks[61].y * self.video_height),
            (landmarks[291].x * self.video_width, landmarks[291].y * self.video_height),
            (landmarks[199].x * self.video_width, landmarks[199].y * self.video_height)
        ], dtype="double")
        model_points = np.array([
            (0.0, 0.0, 0.0), (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0), (0.0, 0.0, -330.0)
        ])
        focal_length = self.video_width
        center = (self.video_width / 2, self.video_height / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
        dist_coeffs = np.zeros((4, 1))
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if success:
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
            singular = sy < 1e-6
            if not singular:
                pitch = np.arctan2(-rotation_matrix[2, 0], sy)
                yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
                roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            else:
                pitch = np.arctan2(-rotation_matrix[2, 0], sy)
                yaw = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                roll = 0.0
            yaw_deg = np.degrees(yaw)
            pitch_deg = np.degrees(pitch)
            roll_deg = np.degrees(roll)

            distraction_detected = False
            direction = None
            if yaw_deg > self.RIGHT_THRESHOLD:
                distraction_detected = True
                direction = 'right'
            elif yaw_deg < -self.LEFT_THRESHOLD:
                distraction_detected = True
                direction = 'left'
            elif pitch_deg > self.UP_THRESHOLD:
                distraction_detected = True
                direction = 'up'
            elif pitch_deg < -self.DOWN_THRESHOLD:
                distraction_detected = True
                direction = 'down'

            return distraction_detected, direction, yaw_deg, pitch_deg, roll_deg
        return False, None, 0.0, 0.0, 0.0

    def check_camera_obstruction(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_intensity = np.mean(gray_frame)
        color_variance = np.std(frame)
        is_dark_obstructed = mean_intensity < self.BRIGHTNESS_THRESHOLD
        is_uniform_obstructed = color_variance < self.COLOR_VAR_THRESHOLD
        return is_dark_obstructed or is_uniform_obstructed

    def update_event_states(self, yawn_detected, distraction_detected_from_head_pose, head_direction, left_eye_closed, right_eye_closed, is_camera_obstructed, faces_detected):
        delta_time = 1 / self.metrics['fps'] if self.metrics['fps'] > 0 else 0
        current_time = time.time()

        # Handle Camera Obstruction
        if is_camera_obstructed:
            if self.camera_obstructed_start_time is None:
                self.camera_obstructed_start_time = current_time
            self.events['camera_obstruction']['timer'] = current_time - self.camera_obstructed_start_time
            if self.events['camera_obstruction']['timer'] >= self.CAMERA_OBSTRUCTION_TIME:
                self.events['camera_obstruction']['active'] = True
                # --- ADDED: Print Event Message ---
                print("Camera is obstructed")
                # ----------------------------------
                if (current_time - self.events['camera_obstruction']['alert_time']) > self.ALERT_DELAY_SECONDS:
                    self.play_alert('camera_obstruction')
                    self.events['camera_obstruction']['alert_time'] = current_time
            for event_name in self.events:
                if event_name not in ['camera_obstruction', 'face_obstruction']:
                    self.events[event_name]['active'] = False
                    self.events[event_name]['timer'] = 0.0
            return
        else:
            self.camera_obstructed_start_time = None
            self.events['camera_obstruction']['active'] = False
            self.events['camera_obstruction']['timer'] = 0.0

        # Handle Face Obstruction
        if faces_detected == 0 and not is_camera_obstructed:
            if self.face_obstructed_start_time is None:
                self.face_obstructed_start_time = current_time
            self.events['face_obstruction']['timer'] = current_time - self.face_obstructed_start_time
            if self.events['face_obstruction']['timer'] >= self.FACE_OBSTRUCTION_TIME:
                self.events['face_obstruction']['active'] = True
                # --- ADDED: Print Event Message ---
                print("Face obstruction detected")
                # ----------------------------------
                if (current_time - self.events['face_obstruction']['alert_time']) > self.ALERT_DELAY_SECONDS:
                    self.play_alert('face_obstruction')
                    self.events['face_obstruction']['alert_time'] = current_time
            for event_name in self.events:
                if event_name not in ['camera_obstruction', 'face_obstruction']:
                    self.events[event_name]['active'] = False
                    self.events[event_name]['timer'] = 0.0
            return
        else:
            self.face_obstructed_start_time = None
            self.events['face_obstruction']['active'] = False
            self.events['face_obstruction']['timer'] = 0.0

        # Handle Distraction (Head Pose)
        is_distracted = distraction_detected_from_head_pose
        if is_distracted:
            # Note: The original code didn't have a timer/threshold check for distraction here,
            # but it did set the active flag and check the alert delay. Keeping the original logic.
            self.events['distraction']['active'] = True 
            if (current_time - self.events['distraction']['alert_time']) > self.ALERT_DELAY_SECONDS:
                # print("Distraction detected") # You can uncomment this if you need a distraction printout
                # self.play_alert('distraction') # Alert is played in the original code but logic is spread out
                self.events['distraction']['alert_time'] = current_time
            for direction in self.direction_timers:
                if direction == head_direction:
                    self.direction_timers[direction] += delta_time
                else:
                    self.direction_timers[direction] = 0.0
        else:
            self.distraction_start_time = None
            self.events['distraction']['active'] = False
            self.events['distraction']['timer'] = 0.0
            for direction in self.direction_timers:
                self.direction_timers[direction] = 0.0

        # Handle Yawning
        if yawn_detected:
            if self.yawning_start_time is None:
                self.yawning_start_time = current_time
            self.events['yawning']['timer'] = current_time - self.yawning_start_time
            if self.events['yawning']['timer'] >= self.YAWN_TIME_THRESHOLD:
                self.events['yawning']['active'] = True
                # --- ADDED: Print Event Message ---
                print("Yawning detected")
                # ----------------------------------
                if (current_time - self.events['yawning']['alert_time']) > self.ALERT_DELAY_SECONDS:
                    self.play_alert('yawning')
                    self.events['yawning']['alert_time'] = current_time
        else:
            self.yawning_start_time = None
            self.events['yawning']['active'] = False
            self.events['yawning']['timer'] = 0.0

        # Handle Fatigue
        if left_eye_closed and right_eye_closed:
            if self.fatigue_start_time is None:
                self.fatigue_start_time = current_time
            closed_time = current_time - self.fatigue_start_time
            self.events['fatigue']['timer'] = closed_time
            if closed_time >= self.FATIGUE_TIME_THRESHOLD:
                self.events['fatigue']['active'] = True
                # --- ADDED: Print Event Message ---
                print("Fatigue detected")
                # ----------------------------------
                if (current_time - self.events['fatigue']['alert_time']) > self.ALERT_DELAY_SECONDS:
                    self.play_alert('fatigue')
                    self.events['fatigue']['alert_time'] = current_time
        else:
            self.fatigue_start_time = None
            self.events['fatigue']['active'] = False
            self.events['fatigue']['timer'] = 0.0

        self.events['left_eye_closed']['active'] = left_eye_closed
        self.events['left_eye_closed']['timer'] = (self.events['left_eye_closed']['timer'] + delta_time) if left_eye_closed else 0.0
        self.events['right_eye_closed']['active'] = right_eye_closed
        self.events['right_eye_closed']['timer'] = (self.events['right_eye_closed']['timer'] + delta_time) if right_eye_closed else 0.0

    # Reset other timers if a major event is active
        if self.events['fatigue']['active'] or self.events['yawning']['active'] or self.events['distraction']['active']:
            self.events['smoking']['active'] = False
            self.events['smoking']['timer'] = 0.0
            self.smoking_start_time = None

            self.events['phone_uses']['active'] = False
            self.events['phone_uses']['timer'] = 0.0
            self.phone_use_start_time = None

            self.events['eating_drinking']['active'] = False
            self.events['eating_drinking']['timer'] = 0.0
            self.eating_drinking_start_time = None

            self.events['wearing_mask']['active'] = False
            self.events['wearing_mask']['timer'] = 0.0

    def draw_dms_events_panel(self, frame):
        panel_x, panel_y, events_panel_height = self.video_width, 0, 500
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + self.panel_width, panel_y + events_panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + self.panel_width, panel_y + events_panel_height), (0, 255, 0), 2)
        label_text = "DMS Events"
        (text_w, _), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        label_x = panel_x + (self.panel_width - text_w) // 2
        cv2.putText(frame, label_text, (label_x, panel_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        events_list = [
            ('smoking', 'Smoking'), ('yawning', 'Yawning'), ('camera_obstruction', 'Cam Obsc'),
            ('face_obstruction', 'Face Obsc'),('fatigue', 'Fatigue'),('distraction', 'Distraction'), ('phone_uses', 'Phone'),
            ('wearing_mask', 'Mask'),('seatbelt_off', 'S_OFF'), ('wearing_seatbelt', 'S_ON'),
            ('eating_drinking', 'Eating/Drinking'),
        ]
        icon_start_y, icon_size, row_spacing, icon_spacing_x = panel_y + 80, 60, 92, 90
        for i, (icon_name, label) in enumerate(events_list):
            row, col = i // 5, i % 5
            icon_x, icon_y = panel_x + 20 + col * icon_spacing_x, icon_start_y + row * row_spacing
            is_active = self.events.get(icon_name, {'active': False})['active']
            bg_color = (0, 255, 0) if is_active else (220, 220, 220)
            label_offset_y = icon_size + 15
            (text_w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            label_x = icon_x + (icon_size - text_w) // 2
            cv2.putText(frame, label, (label_x, icon_y + label_offset_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            self.draw_icon(frame, icon_name, icon_x, icon_y, bg_color)

    def draw_dms_output_panel(self, frame):
        panel_x, panel_y = self.video_width, 350
        panel_height = self.display_height - panel_y
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + self.panel_width, panel_y + panel_height), (255, 255, 255), -1)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + self.panel_width, panel_y + panel_height), (0, 255, 0), 2)
        cv2.putText(frame, "DMS Output Parameters:", (panel_x + 10, panel_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y_offset = 60

        obstruction_text = ""
        if self.events['camera_obstruction']['active']:
            obstruction_text = "Camera is Obstructed"
        elif self.events['face_obstruction']['active']:
            obstruction_text = "Face is Obstructed"

        params = [
            f"Camera FPS: {self.metrics['fps']:.1f}",
            f"Distraction_Looking_left: {self.direction_timers['left']:.1f} secs",
            f"Distraction_Looking_right: {self.direction_timers['right']:.1f} secs",
            f"Distraction_Looking_up: {self.direction_timers['up']:.1f} secs",
            f"Distraction_Looking_down: {self.direction_timers['down']:.1f} secs",
            f"Head Pose (Yaw): {self.head_pose_yaw:.1f} deg",
            f"Head Pose (Pitch): {self.head_pose_pitch:.1f} deg",
            f"Head Pose (Roll): {self.head_pose_roll:.1f} deg",
            f"Fatigue: {self.events['fatigue']['timer']:.1f} secs",
            f"Yawning: {self.events['yawning']['timer']:.1f} secs",
            f"Phone Use: {self.events['phone_uses']['timer']:.1f} secs",
            f"Eating/Drinking: {self.events['eating_drinking']['timer']:.1f} secs",
            f"Smoking: {self.events['smoking']['timer']:.1f} secs",
            f"Mask: {self.events['wearing_mask']['timer']:.1f} secs",
            f"Left Eye Closed: {self.events['left_eye_closed']['timer']:.1f} secs",
            f"Right Eye Closed: {self.events['right_eye_closed']['timer']:.1f} secs",
        ]

        if obstruction_text:
            params.append(f"Obstruction: {obstruction_text}")

        for param in params:
            cv2.putText(frame, param, (panel_x + 10, panel_y + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            y_offset += 20

    # ==================== CAN SIGNALING METHODS (MODIFIED) ======================

    def package_can_data(self):
        """
        Creates a single 8-byte payload for the consolidated CAN ID 0x307 based on general status.
        This is used when camera obstruction is NOT active.
        """

        # --- 1. Data[0]: DMS Event Flags (Bit-Packed) ---
        event_byte = 0
        if self.events['fatigue']['active']:
            event_byte |= 0b00000001  # Bit 0 (1)
        if self.events['yawning']['active']:
            event_byte |= 0b00000010  # Bit 1 (2)
        if self.events['face_obstruction']['active']:
            event_byte |= 0b00000100  # Bit 2 (4)
        if self.events['camera_obstruction']['active']:
            event_byte |= 0b00001000  # Bit 3 (8)

        # --- 2. Data[1] & Data[2]: Head Pose Metrics (Signed 1-degree resolution) ---
        yaw_int = int(max(-127, min(127, round(self.head_pose_yaw))))
        pitch_int = int(max(-127, min(127, round(self.head_pose_pitch))))

        # --- 3. Data[3] & Data[4]: Timer Metrics (Unsigned 0.1s resolution) ---
        timer_fatigue_int = int(max(0, min(255, round(self.events['fatigue']['timer'], 1) * 10)))
        timer_yawning_int = int(max(0, min(255, round(self.events['yawning']['timer'], 1) * 10)))

        # --- 4. Assemble the 8-Byte Payload ---
        can_payload = [
            event_byte,             # Data[0]: Event Flags (Fatigue, Yawning, Face/Cam Obstr)
            yaw_int & 0xFF,         # Data[1]: Yaw Angle (signed, 1 deg/bit)
            pitch_int & 0xFF,       # Data[2]: Pitch Angle (signed, 1 deg/bit)
            timer_fatigue_int,      # Data[3]: Fatigue Timer (0.1s/bit)
            timer_yawning_int,      # Data[4]: Yawning Timer (0.1s/bit)
            0x00,                   # Data[5]: Reserved
            0x00,                   # Data[6]: Reserved
            0x00                    # Data[7]: Reserved
        ]

        return {
            'DMS_STATUS': {'id': 0x307, 'data': can_payload}
        }

    def send_can_message(self, message_data):
        """Sends a single CAN message if the bus is enabled. For CAN-to-UART, sends SLCAN string over serial in t<ID><DLC><DATA> format. For PCAN/SocketCAN, uses python-can as before."""
        if not self.can_enabled or not self.can_bus:
            return

        can_interface = getattr(self, 'can_interface', 'auto')
        serial_port = getattr(self, 'serial_port', None)
        baudrate = getattr(self, 'baudrate', 5000000)

        # --- SLCAN Bus Enable: Use persistent serial handle (matching your working code) ---
        if can_interface == 'uart' and serial_port and message_data['id'] == 0x307:
            try:
                import serial as pyserial
                if self.serial_handle is None:
                    print("Error: Serial handle not initialized. SLCAN bus not open.")
                    return
                can_id = message_data['id']
                data = message_data['data']
                dlc = len(data)
                can_id_str = f"{can_id:03X}"  # 3-digit hex, e.g. '307'
                data_str = ''.join(f"{b:02X}" for b in data)  # 2-digit hex for each byte
                slcan_str = f"t{can_id_str}{dlc}{data_str}\r"
                self.serial_handle.write(slcan_str.encode('ascii'))
                self.serial_handle.flush()
                time.sleep(0.3)  # Increased delay for CAN-to-UART module to prevent signal overlap
                print(f"Sent CAN frame: {slcan_str.strip()}")
            except Exception as e:
                print(f"Error sending SLCAN message to UART: {e}")
            return

        # Otherwise, use python-can for PCAN/SocketCAN
        if not self.can_bus:
            return
        try:
            import can
            msg = can.Message(
                arbitration_id=message_data['id'],
                data=message_data['data'],
                is_extended_id=False
            )
            self.can_bus.send(msg)
            print(f"Sent CAN frame: ID={message_data['id']:03X} Data={message_data['data']}")
        except Exception as e:
            print(f"Error sending CAN frame: {e}")

    def cleanup_can(self):
        """Properly shuts down all CAN interfaces (PCAN, SocketCAN, UART/SLCAN, Serial) to avoid bus errors."""
        # Only shutdown CAN bus after all processing is done (matching your working code)
        if getattr(self, 'can_interface', None) == 'uart' and getattr(self, 'serial_port', None):
            try:
                import serial as pyserial
                if self.serial_handle:
                    self.serial_handle.write(b'C\r')  # SLCAN: 'C' + CR to close CAN channel
                    self.serial_handle.flush()
                    time.sleep(0.1)  # Matching your working code timing
                    resp = self.serial_handle.read(10)
                    print(f"Sent C (close CAN bus). Response: {resp}")
                    self.serial_handle.close()
                    self.serial_handle = None
                    print("Serial port closed.")
                else:
                    print("Serial handle not open, cannot send 'C'.")
            except Exception as e:
                print(f"Failed to send 'C' to {self.serial_port}: {e}")
        # PCAN, SocketCAN, SLCAN, Serial: python-can bus shutdown
        if getattr(self, 'can_bus', None):
            try:
                # Only shutdown if bus is open
                if hasattr(self.can_bus, 'is_open'):
                    if self.can_bus.is_open:
                        self.can_bus.shutdown()
                        print("python-can bus shutdown called.")
                else:
                    self.can_bus.shutdown()
                    print("python-can bus shutdown called.")
            except Exception as e:
                print(f"Error shutting down python-can bus: {e}")

    def run(self):
        print("Starting Samsan DMS Monitor...")
        if self.can_enabled:
            print("CAN Bus communication is ACTIVE on 'can0', sending ID 0x307.")
        else:
            print("CAN Bus functionality is disabled.")

        while self.cap.isOpened():
            current_time = time.time()
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 1. Image Preprocessing and ROI
            frame_resized = cv2.resize(frame, (self.video_width, self.video_height))
            rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # 2. Camera Obstruction Check (before face detection)
            is_camera_obstructed = self.check_camera_obstruction(frame_resized)
            
            # 3. Face Mesh Detection
            results = self.face_mesh.process(rgb_frame)
            
            yawn_detected = False
            distraction_detected_from_head_pose = False
            head_direction = None
            left_eye_closed = False
            right_eye_closed = False
            self.metrics['faces_detected'] = 0

            # Process detected face landmarks
            if results.multi_face_landmarks:
                self.metrics['faces_detected'] = len(results.multi_face_landmarks)
                landmarks = results.multi_face_landmarks[0].landmark
                
                # Eye Closure and Fatigue
                left_eye_closed, right_eye_closed = self.detect_eye_closure(landmarks)

                # Yawning
                yawn_detected, _ = self.detect_yawn(landmarks)

                # Head Pose and Distraction
                (distraction_detected_from_head_pose, head_direction, self.head_pose_yaw, self.head_pose_pitch, self.head_pose_roll) = self.detect_head_pose(landmarks)

                # Draw landmarks (optional)
                for face_landmarks in results.multi_face_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame_resized,
                        face_landmarks,
                        self.mp_face_mesh.FACEMESH_CONTOURS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                        self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1)
                    )

            # 4. Update Event States and Timers
            self.update_event_states(yawn_detected, distraction_detected_from_head_pose, head_direction, left_eye_closed, right_eye_closed, is_camera_obstructed, self.metrics['faces_detected'])

            # 5. Rendering Panels
            display_frame = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
            display_frame[0:self.video_height, 0:self.video_width] = frame_resized
            self.draw_dms_events_panel(display_frame)
            self.draw_dms_output_panel(display_frame)

            # 6. CAN Message Sending Logic (CORRECTED AND CONSOLIDATED)
            if self.can_enabled:
                can_messages = {}
                
                # Default/General DMS Status (packaged in package_can_data)
                can_messages.update(self.package_can_data())

                # Overwrite/Add specific messages if a major event is active, as per custom logic in the original file
                if self.events['camera_obstruction']['active']:
                    # Requested CAN data frame: 307h 50h 8 D 00h 00h 00h 00h 67h 00h 00h 00h
                    # Interpreting the 8-byte payload as [0x00, 0x00, 0x00, 0x00, 0x67, 0x00, 0x00, 0x00]
                    camera_obstruction_data = [
                        0x00, 0x00, 0x00, 0x00, 0x67, 0x00, 0x00, 0x00
                    ]
                    can_messages['DMS_STATUS'] = { # Overwrite general status with specific obstruction message
                        'id': 0x307, 
                        'data': camera_obstruction_data
                    }
                
                elif self.events['fatigue']['active']:
                    # Pre-frame message from original file's logic
                    pre_frame_data=[
                        0x00, 0x00, 0x00, 0x00, 0x11, 0x00, 0x00, 0x00
                    ]
                    pre_frame = {
                        'id': 0x307, 
                        'data': pre_frame_data
                    }
                    self.send_can_message(pre_frame)
                    
                    # Additional delay for CAN-to-UART module between multiple frames
                    time.sleep(0.2)

                    # Fatigue message from original file's logic
                    fatigue_data = [
                        0x00, 0x00, 0x00, 0x00, 0xF0, 0x00, 0x00, 0x00
                    ]
                    can_messages['DMS_STATUS'] = { # Overwrite with fatigue message
                        'id': 0x307, 
                        'data': fatigue_data
                    }
                
                elif self.events['face_obstruction']['active']:
                    # Face obstruction message from original file's logic
                    face_obstruction_data = [
                        0x00, 0x00, 0x67, 0x12, 0x00, 0x00, 0x00, 0x00
                    ]
                    can_messages['DMS_STATUS'] = { # Overwrite with face obstruction message
                        'id': 0x307, 
                        'data': face_obstruction_data
                    }
                    
                # Send all (potentially overwritten) packaged messages for ID 0x307
                for msg_name, msg_data in can_messages.items():
                    self.send_can_message(msg_data)

            # 7. Calculate FPS
            frame_time = current_time - self.last_time
            self.last_time = current_time
            self.metrics['frame_processing_time'] = frame_time * 1000
            self.metrics['fps'] = 1.0 / frame_time if frame_time > 0 else 0
            
            # 8. Display
            cv2.imshow("Edgetensor DMS Monitor", display_frame)

            if cv2.waitKey(5) & 0xFF == 27: # ESC key to exit
                break

        self.cap.release()
        cv2.destroyAllWindows()
        # Only shutdown CAN bus after all event processing and display is complete
        self.cleanup_can()  # Ensure CAN bus is properly closed
        # Add a final print to confirm script end
        print("DMS Monitor finished. CAN bus shutdown complete.")

# ==================== MAIN EXECUTION BLOCK ======================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Edgetensor DMS Monitor for Driver Fatigue and Distraction")
    parser.add_argument('--roi', type=str, default='default', help='ROI configuration name from config.ini')
    args = parser.parse_args()

    monitor = EdgetensorDMMMonitor(roi_position=args.roi)
    monitor.run()
