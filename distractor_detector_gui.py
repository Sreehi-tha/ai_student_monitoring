import cv2
import numpy as np
import dlib
import time
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
from collections import deque
import threading
import queue
import pygame
from datetime import datetime, timedelta
import warnings
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from PIL import Image, ImageTk
import os
warnings.filterwarnings('ignore')

class DistractorDetectorGUI:
    def __init__(self):
        # Initialize the main window
        self.root = tk.Tk()
        self.root.title("🎓 Student Distractor Detector")
        self.root.geometry("1000x700")
        self.root.configure(bg='#1a1a2e')
        
        # Make window non-resizable for consistent layout
        self.root.resizable(False, False)
        
        # Initialize detection variables
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = None
        self.cap = None
        self.is_running = False
        self.video_thread = None
        
        # Detection parameters (using original logic)
        self.EAR_THRESHOLD = 0.25
        self.CONSECUTIVE_FRAMES = 20
        self.MAR_THRESHOLD = 0.7
        self.TALKING_FRAMES = 5
        self.GAZE_THRESHOLD = 0.2
        self.GAZE_FRAMES = 15
        self.HEAD_POSE_YAW_THRESHOLD = 15
        self.HEAD_POSE_PITCH_THRESHOLD = 15
        self.EYE_GAZE_THRESHOLD = 0.3
        self.FACE_DISPLACEMENT_THRESHOLD = 0.2
        
        # Counters
        self.drowsy_counter = 0
        self.talking_counter = 0
        self.gaze_counter = 0
        
        # Alert status
        self.drowsy_alert = False
        self.talking_alert = False
        self.gaze_alert = False
        
        # Data logging
        self.start_time = time.time()
        self.log_data = {
            'minutes': [],
            'drowsy_events': [],
            'talking_events': [],
            'gaze_events': [],
            'total_distractions': []
        }
        
        self.current_minute_data = {
            'drowsy': 0,
            'talking': 0,
            'gaze': 0
        }
        self.last_minute_log = time.time()
        
        # Eye landmarks indices
        self.LEFT_EYE = list(range(36, 42))
        self.RIGHT_EYE = list(range(42, 48))
        self.MOUTH = list(range(48, 68))
        
        # Initialize pygame for audio
        pygame.mixer.init()
        
        # Initialize GUI
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the modern GUI interface"""
        # Create modern style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure custom colors
        style.configure('Modern.TFrame', background='#1a1a2e')
        style.configure('Title.TLabel', background='#1a1a2e', foreground='#ffffff', 
                       font=('Arial', 24, 'bold'))
        style.configure('Subtitle.TLabel', background='#1a1a2e', foreground='#a0a0a0', 
                       font=('Arial', 12))
        style.configure('Success.TButton', background='#00ff88', foreground='#000000',
                       font=('Arial', 12, 'bold'), padding=10)
        style.configure('Danger.TButton', background='#ff4757', foreground='#ffffff',
                       font=('Arial', 12, 'bold'), padding=10)
        style.configure('Info.TButton', background='#3742fa', foreground='#ffffff',
                       font=('Arial', 12, 'bold'), padding=10)
        
        # Main container
        main_container = tk.Frame(self.root, bg='#1a1a2e')
        main_container.pack(fill=tk.BOTH, expand=True, padx=30, pady=30)
        
        # Header section
        header_frame = tk.Frame(main_container, bg='#1a1a2e')
        header_frame.pack(fill=tk.X, pady=(0, 30))
        
        # Title with gradient-like effect
        title_frame = tk.Frame(header_frame, bg='#1a1a2e')
        title_frame.pack()
        
        title_label = tk.Label(title_frame, text="🎓 Student Distractor Detector", 
                              font=('Arial', 28, 'bold'), bg='#1a1a2e', fg='#00ff88')
        title_label.pack()
        
        subtitle_label = tk.Label(title_frame, text="AI-Powered Focus Monitoring System", 
                                 font=('Arial', 14), bg='#1a1a2e', fg='#a0a0a0')
        subtitle_label.pack(pady=(5, 0))
        
        # Status indicator
        self.status_frame = tk.Frame(header_frame, bg='#1a1a2e')
        self.status_frame.pack(pady=(20, 0))
        
        self.status_indicator = tk.Label(self.status_frame, text="●", 
                                        font=('Arial', 20), bg='#1a1a2e', fg='#ff6b6b')
        self.status_indicator.pack(side=tk.LEFT)
        
        self.status_label = tk.Label(self.status_frame, text="Ready to Start", 
                                   font=('Arial', 14, 'bold'), bg='#1a1a2e', fg='#ffffff')
        self.status_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Control buttons section
        control_frame = tk.Frame(main_container, bg='#1a1a2e')
        control_frame.pack(fill=tk.X, pady=(0, 30))
        
        # Buttons with modern styling
        button_container = tk.Frame(control_frame, bg='#1a1a2e')
        button_container.pack()
        
        # Start button
        self.start_btn = tk.Button(button_container, text="🎬 Start Detection", 
                                  command=self.start_detection,
                                  bg='#00ff88', fg='#000000', font=('Arial', 14, 'bold'),
                                  relief='flat', padx=30, pady=15, cursor='hand2',
                                  activebackground='#00e676', activeforeground='#000000')
        self.start_btn.pack(side=tk.LEFT, padx=(0, 20))
        
        # Stop button
        self.stop_btn = tk.Button(button_container, text="⏹️ Stop Detection", 
                                 command=self.stop_detection, state=tk.DISABLED,
                                 bg='#ff4757', fg='#ffffff', font=('Arial', 14, 'bold'),
                                 relief='flat', padx=30, pady=15, cursor='hand2',
                                 activebackground='#ff3838', activeforeground='#ffffff')
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 20))
        
        # Generate graphs button
        self.graph_btn = tk.Button(button_container, text="📊 View Analytics", 
                                  command=self.generate_graphs,
                                  bg='#3742fa', fg='#ffffff', font=('Arial', 14, 'bold'),
                                  relief='flat', padx=30, pady=15, cursor='hand2',
                                  activebackground='#2f3542', activeforeground='#ffffff')
        self.graph_btn.pack(side=tk.LEFT, padx=(0, 20))
        
        # Summary button
        self.summary_btn = tk.Button(button_container, text="📋 Summary Report", 
                                    command=self.show_summary,
                                    bg='#ff6b6b', fg='#ffffff', font=('Arial', 14, 'bold'),
                                    relief='flat', padx=30, pady=15, cursor='hand2',
                                    activebackground='#ff5252', activeforeground='#ffffff')
        self.summary_btn.pack(side=tk.LEFT)
        
        # Video display section
        video_frame = tk.Frame(main_container, bg='#16213e', relief='ridge', bd=2)
        video_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Video header
        video_header = tk.Frame(video_frame, bg='#16213e', height=40)
        video_header.pack(fill=tk.X)
        video_header.pack_propagate(False)
        
        video_title = tk.Label(video_header, text="📹 Live Video Feed", 
                              font=('Arial', 16, 'bold'), bg='#16213e', fg='#ffffff')
        video_title.pack(side=tk.LEFT, padx=15, pady=10)
        
        # Session timer
        self.timer_label = tk.Label(video_header, text="00:00", 
                                   font=('Arial', 16, 'bold'), bg='#16213e', fg='#00ff88')
        self.timer_label.pack(side=tk.RIGHT, padx=15, pady=10)
        
        # Video display
        self.video_label = tk.Label(video_frame, text="Camera feed will appear here\nClick 'Start Detection' to begin", 
                                   bg='#0f0f23', fg='#a0a0a0', font=('Arial', 16),
                                   width=80, height=20)
        self.video_label.pack(expand=True, fill=tk.BOTH, padx=20, pady=(0, 20))
        
        # Footer
        footer_frame = tk.Frame(main_container, bg='#1a1a2e')
        footer_frame.pack(fill=tk.X)
        
        footer_label = tk.Label(footer_frame, text="💡 Tip: Maintain good posture and look directly at the camera for accurate detection", 
                               font=('Arial', 10), bg='#1a1a2e', fg='#a0a0a0')
        footer_label.pack()
        
        # Start GUI update timer
        self.update_gui()
        
    def start_detection(self):
        """Start the detection process"""
        # Check for landmark file
        if not os.path.exists('shape_predictor_68_face_landmarks.dat'):
            messagebox.showerror("Missing File", 
                               "shape_predictor_68_face_landmarks.dat not found!\n\n"
                               "Please download it from:\n"
                               "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2\n\n"
                               "Extract and place it in the same directory as this script.")
            return
            
        try:
            self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            if not self.cap.isOpened():
                messagebox.showerror("Camera Error", "Could not open camera!\nPlease check if camera is connected and not in use.")
                return
                
            self.is_running = True
            self.start_time = time.time()
            self.last_minute_log = time.time()
            
            # Reset data
            self.log_data = {
                'minutes': [],
                'drowsy_events': [],
                'talking_events': [],
                'gaze_events': [],
                'total_distractions': []
            }
            self.current_minute_data = {'drowsy': 0, 'talking': 0, 'gaze': 0}
            
            # Start video thread
            self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
            self.video_thread.start()
            
            # Update UI
            self.start_btn.config(state=tk.DISABLED, bg='#cccccc')
            self.stop_btn.config(state=tk.NORMAL, bg='#ff4757')
            self.status_indicator.config(fg='#00ff88')
            self.status_label.config(text="Detection Active")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start detection:\n{str(e)}")
            
    def stop_detection(self):
        """Stop the detection process"""
        self.is_running = False
        
        if self.cap:
            self.cap.release()
            
        # Update UI
        self.start_btn.config(state=tk.NORMAL, bg='#00ff88')
        self.stop_btn.config(state=tk.DISABLED, bg='#cccccc')
        self.status_indicator.config(fg='#ff6b6b')
        self.status_label.config(text="Stopped")
        self.video_label.config(image='', text="Detection stopped\nClick 'Start Detection' to begin again")
        
    def show_summary(self):
        """Show summary report"""
        self.generate_final_report()
        
    def update_gui(self):
        """Update GUI elements"""
        if hasattr(self, 'start_time') and self.is_running:
            # Update session time
            elapsed = int(time.time() - self.start_time)
            minutes = elapsed // 60
            seconds = elapsed % 60
            self.timer_label.config(text=f"{minutes:02d}:{seconds:02d}")
        
        # Schedule next update
        self.root.after(1000, self.update_gui)
        
    # ============ ORIGINAL DETECTION LOGIC ============
    
    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear
    
    def mouth_aspect_ratio(self, mouth):
        A = dist.euclidean(mouth[2], mouth[10])
        B = dist.euclidean(mouth[4], mouth[8])
        C = dist.euclidean(mouth[0], mouth[6])
        mar = (A + B) / (2.0 * C)
        return mar
    
    def get_head_pose(self, landmarks, frame_shape):
        model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ])
        
        image_points = np.array([
            landmarks[30],
            landmarks[8],
            landmarks[36],
            landmarks[45],
            landmarks[48],
            landmarks[54]
        ], dtype="double")
        
        height, width = frame_shape[:2]
        focal_length = width
        center = (width/2, height/2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")
        
        dist_coeffs = np.zeros((4,1))
        
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs)
        
        if success:
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            sy = np.sqrt(rotation_matrix[0,0] * rotation_matrix[0,0] + 
                        rotation_matrix[1,0] * rotation_matrix[1,0])
            
            singular = sy < 1e-6
            
            if not singular:
                x = np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2])
                y = np.arctan2(-rotation_matrix[2,0], sy)
                z = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])
            else:
                x = np.arctan2(-rotation_matrix[1,2], rotation_matrix[1,1])
                y = np.arctan2(-rotation_matrix[2,0], sy)
                z = 0
            
            pitch = np.degrees(x)
            yaw = np.degrees(y)
            roll = np.degrees(z)
            
            return pitch, yaw, roll
        
        return 0, 0, 0
    
    def get_eye_gaze_ratio(self, eye_landmarks):
        left_corner = eye_landmarks[0]
        right_corner = eye_landmarks[3]
        top_point = eye_landmarks[1]
        bottom_point = eye_landmarks[5] if len(eye_landmarks) > 5 else eye_landmarks[4]
        
        eye_center = np.mean(eye_landmarks, axis=0)
        eye_width = np.linalg.norm(right_corner - left_corner)
        
        if eye_width == 0:
            return 0
        
        horizontal_ratio = (eye_center[0] - left_corner[0]) / eye_width
        gaze_ratio = (horizontal_ratio - 0.5) * 2
        
        return abs(gaze_ratio)
    
    def is_looking_away(self, landmarks, frame_shape):
        pitch, yaw, roll = self.get_head_pose(landmarks, frame_shape)
        
        significant_yaw = abs(yaw) > self.HEAD_POSE_YAW_THRESHOLD
        significant_pitch = abs(pitch) > self.HEAD_POSE_PITCH_THRESHOLD
        head_looking_away = significant_yaw or significant_pitch
        
        left_eye = landmarks[self.LEFT_EYE]
        right_eye = landmarks[self.RIGHT_EYE]
        
        left_gaze_ratio = self.get_eye_gaze_ratio(left_eye)
        right_gaze_ratio = self.get_eye_gaze_ratio(right_eye)
        avg_gaze_ratio = (left_gaze_ratio + right_gaze_ratio) / 2
        
        extreme_eye_movement = avg_gaze_ratio > self.EYE_GAZE_THRESHOLD
        
        face_center = np.mean(landmarks, axis=0)
        frame_center = np.array([frame_shape[1]/2, frame_shape[0]/2])
        face_displacement = np.linalg.norm(face_center - frame_center)
        
        frame_diagonal = np.sqrt(frame_shape[0]**2 + frame_shape[1]**2)
        normalized_displacement = face_displacement / frame_diagonal
        
        significant_displacement = normalized_displacement > self.FACE_DISPLACEMENT_THRESHOLD
        
        looking_away = head_looking_away or (extreme_eye_movement and significant_displacement)
        
        confidence_score = 0
        if significant_yaw:
            confidence_score += 2
        if significant_pitch:
            confidence_score += 1.5
        if extreme_eye_movement:
            confidence_score += 1
        if significant_displacement:
            confidence_score += 0.5
            
        final_looking_away = confidence_score >= 2.0
        
        return final_looking_away, pitch, yaw, avg_gaze_ratio
    
    def generate_beep_sound(self, frequency, duration, volume=0.5):
        try:
            sample_rate = 22050
            frames = int(duration * sample_rate)
            arr = np.zeros(frames)
            
            for i in range(frames):
                arr[i] = volume * np.sin(2 * np.pi * frequency * i / sample_rate)
            
            arr = (arr * 32767).astype(np.int16)
            stereo_arr = np.zeros((frames, 2), dtype=np.int16)
            stereo_arr[:, 0] = arr
            stereo_arr[:, 1] = arr
            
            sound = pygame.sndarray.make_sound(stereo_arr)
            sound.play()
            pygame.time.wait(int(duration * 1000))
            
        except Exception as e:
            print(f"Audio error: {e}")
            print("\a")
    
    def play_alert_sound(self, alert_type):
        print(f"🔊 PLAYING {alert_type.upper()} ALERT SOUND!")
        try:
            if alert_type == "drowsy":
                for i in range(3):
                    self.generate_beep_sound(400, 0.3, 0.7)
                    if i < 2:
                        time.sleep(0.1)
            elif alert_type == "talking":
                for i in range(2):
                    self.generate_beep_sound(600, 0.2, 0.6)
                    if i < 1:
                        time.sleep(0.1)
            elif alert_type == "gaze":
                self.generate_beep_sound(800, 0.5, 0.8)
        except Exception as e:
            print(f"Alert sound error: {e}")
            try:
                import winsound
                if alert_type == "drowsy":
                    for i in range(3):
                        winsound.Beep(400, 300)
                elif alert_type == "talking":
                    for i in range(2):
                        winsound.Beep(600, 200)
                elif alert_type == "gaze":
                    winsound.Beep(800, 500)
            except:
                if alert_type == "drowsy":
                    print("\a" * 3)
                elif alert_type == "talking":
                    print("\a" * 2)
                elif alert_type == "gaze":
                    print("\a" * 1)
    
    def log_minute_data(self):
        current_time = time.time()
        if current_time - self.last_minute_log >= 60:
            minutes_elapsed = len(self.log_data['minutes']) + 1
            
            print(f"📊 LOGGING MINUTE {minutes_elapsed} DATA:")
            print(f"   Drowsy events: {self.current_minute_data['drowsy']}")
            print(f"   Talking events: {self.current_minute_data['talking']}")
            print(f"   Gaze events: {self.current_minute_data['gaze']}")
            
            self.log_data['minutes'].append(minutes_elapsed)
            self.log_data['drowsy_events'].append(self.current_minute_data['drowsy'])
            self.log_data['talking_events'].append(self.current_minute_data['talking'])
            self.log_data['gaze_events'].append(self.current_minute_data['gaze'])
            
            total = (self.current_minute_data['drowsy'] + 
                    self.current_minute_data['talking'] + 
                    self.current_minute_data['gaze'])
            self.log_data['total_distractions'].append(total)
            
            self.current_minute_data = {'drowsy': 0, 'talking': 0, 'gaze': 0}
            self.last_minute_log = current_time
    
    def video_loop(self):
        """Main video processing loop"""
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = self.process_frame(frame)
            self.log_minute_data()
            
            # Convert frame for GUI display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_pil = frame_pil.resize((640, 380), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(frame_pil)
            
            # Update video label
            self.video_label.config(image=photo, text="")
            self.video_label.image = photo  # Keep a reference
            
        # Clean up
        if self.cap:
            self.cap.release()
    
    def process_frame(self, frame):
        """Process a single frame for all distraction types"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        for face in faces:
            landmarks = self.predictor(gray, face)
            landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
            
            # Eye aspect ratio calculation
            left_eye = landmarks[self.LEFT_EYE]
            right_eye = landmarks[self.RIGHT_EYE]
            left_ear = self.eye_aspect_ratio(left_eye)
            right_ear = self.eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            
            # Drowsiness detection
            if ear < self.EAR_THRESHOLD:
                self.drowsy_counter += 1
                if self.drowsy_counter >= self.CONSECUTIVE_FRAMES:
                    if not self.drowsy_alert:
                        self.drowsy_alert = True
                        self.current_minute_data['drowsy'] += 1
                        print(f"🚨 DROWSY ALERT TRIGGERED! Event #{self.current_minute_data['drowsy']} this minute")
                        self.play_alert_sound("drowsy")
                    cv2.putText(frame, "DROWSY!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.8, (0, 0, 255), 2)
            else:
                self.drowsy_counter = 0
                self.drowsy_alert = False
            
            # Talking detection
            mouth = landmarks[self.MOUTH]
            mar = self.mouth_aspect_ratio(mouth)
            
            if mar > self.MAR_THRESHOLD:
                self.talking_counter += 1
                if self.talking_counter >= self.TALKING_FRAMES:
                    if not self.talking_alert:
                        self.talking_alert = True
                        self.current_minute_data['talking'] += 1
                        print(f"🚨 TALKING ALERT TRIGGERED! Event #{self.current_minute_data['talking']} this minute")
                        self.play_alert_sound("talking")
                    cv2.putText(frame, "TALKING!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.8, (0, 255, 255), 2)
            else:
                self.talking_counter = 0
                self.talking_alert = False
            
            # Gaze detection
            looking_away, pitch, yaw, gaze_ratio = self.is_looking_away(landmarks, frame.shape)
            
            if looking_away:
                self.gaze_counter += 1
                print(f"LOOKING AWAY DETECTED - Counter: {self.gaze_counter}/{self.GAZE_FRAMES}")
                if self.gaze_counter >= self.GAZE_FRAMES:
                    if not self.gaze_alert:
                        self.gaze_alert = True
                        self.current_minute_data['gaze'] += 1
                        print(f"🚨 GAZE ALERT TRIGGERED! Event #{self.current_minute_data['gaze']} this minute")
                        self.play_alert_sound("gaze")
                    cv2.putText(frame, "LOOKING AWAY!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.8, (255, 0, 0), 2)
            else:
                if self.gaze_counter > 0:
                    print(f"Looking away counter reset from {self.gaze_counter}")
                self.gaze_counter = 0
                self.gaze_alert = False
            
            # Draw eye and mouth contours
            cv2.drawContours(frame, [cv2.convexHull(left_eye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(right_eye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (255, 0, 0), 1)
            
            # Display metrics
            cv2.putText(frame, f"EAR: {ear:.2f}", (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (450, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
            
            # Status indicator
            status = "FOCUSED" if not any([self.drowsy_alert, self.talking_alert, self.gaze_alert]) else "DISTRACTED"
            status_color = (0, 255, 0) if status == "FOCUSED" else (0, 0, 255)
            cv2.putText(frame, f"Status: {status}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, status_color, 2)
        
        # Display session info
        elapsed_minutes = int((time.time() - self.start_time) / 60)
        cv2.putText(frame, f"Session: {elapsed_minutes} min", (10, frame.shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def generate_graphs(self):
        """Generate comprehensive graphs for distraction analysis"""
        if len(self.log_data['minutes']) == 0:
            messagebox.showinfo("No Data", "No data available for graphs.\nRun detection for at least one minute.")
            return
        
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle('📊 Student Distraction Analysis Report', fontsize=16, fontweight='bold', color='white')
        
        # Plot 1: Individual distraction types over time
        plt.subplot(2, 2, 1)
        plt.plot(self.log_data['minutes'], self.log_data['drowsy_events'], 
                'r-o', label='Drowsiness', linewidth=2, markersize=4)
        plt.plot(self.log_data['minutes'], self.log_data['talking_events'], 
                'b-s', label='Talking', linewidth=2, markersize=4)
        plt.plot(self.log_data['minutes'], self.log_data['gaze_events'], 
                'g-^', label='Looking Away', linewidth=2, markersize=4)
        plt.xlabel('Time (Minutes)')
        plt.ylabel('Distraction Events')
        plt.title('Distraction Types Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Total distractions over time
        plt.subplot(2, 2, 2)
        plt.plot(self.log_data['minutes'], self.log_data['total_distractions'], 
                'purple', linewidth=3, marker='o', markersize=5)
        plt.fill_between(self.log_data['minutes'], self.log_data['total_distractions'], 
                        alpha=0.3, color='purple')
        plt.xlabel('Time (Minutes)')
        plt.ylabel('Total Distractions')
        plt.title('Total Distractions per Minute')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Distraction distribution (pie chart)
        plt.subplot(2, 2, 3)
        total_drowsy = sum(self.log_data['drowsy_events'])
        total_talking = sum(self.log_data['talking_events'])
        total_gaze = sum(self.log_data['gaze_events'])
        
        if total_drowsy + total_talking + total_gaze > 0:
            labels = ['Drowsiness', 'Talking', 'Looking Away']
            sizes = [total_drowsy, total_talking, total_gaze]
            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.title('Distraction Distribution')
        
        # Plot 4: Moving average of distractions
        plt.subplot(2, 2, 4)
        if len(self.log_data['total_distractions']) >= 3:
            window_size = min(3, len(self.log_data['total_distractions']))
            moving_avg = np.convolve(self.log_data['total_distractions'], 
                                   np.ones(window_size)/window_size, mode='valid')
            plt.plot(self.log_data['minutes'][:len(moving_avg)], moving_avg, 
                    'orange', linewidth=3, marker='s', markersize=4)
            plt.xlabel('Time (Minutes)')
            plt.ylabel('Moving Average')
            plt.title(f'{window_size}-Minute Moving Average of Distractions')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_final_report(self):
        """Generate a final session report"""
        current_time = time.time()
        session_duration = current_time - self.start_time
        
        # Include current minute data if session ends mid-minute
        if (session_duration % 60 > 10) and (  
            self.current_minute_data['drowsy'] > 0 or 
            self.current_minute_data['talking'] > 0 or 
            self.current_minute_data['gaze'] > 0):
            
            print("📊 Including current incomplete minute in final report...")
            minutes_elapsed = len(self.log_data['minutes']) + 1
            
            self.log_data['minutes'].append(minutes_elapsed)
            self.log_data['drowsy_events'].append(self.current_minute_data['drowsy'])
            self.log_data['talking_events'].append(self.current_minute_data['talking'])
            self.log_data['gaze_events'].append(self.current_minute_data['gaze'])
            
            total = (self.current_minute_data['drowsy'] + 
                    self.current_minute_data['talking'] + 
                    self.current_minute_data['gaze'])
            self.log_data['total_distractions'].append(total)
        
        total_minutes = max(1, len(self.log_data['minutes']))
        total_drowsy = sum(self.log_data['drowsy_events'])
        total_talking = sum(self.log_data['talking_events'])
        total_gaze = sum(self.log_data['gaze_events'])
        total_distractions = total_drowsy + total_talking + total_gaze
        
        # Create modern report window
        report_window = tk.Toplevel(self.root)
        report_window.title("📋 Session Summary Report")
        report_window.geometry("700x600")
        report_window.configure(bg='#1a1a2e')
        report_window.resizable(False, False)
        
        # Make it modal
        report_window.transient(self.root)
        report_window.grab_set()
        
        # Report header
        header_frame = tk.Frame(report_window, bg='#1a1a2e', pady=20)
        header_frame.pack(fill=tk.X)
        
        title_label = tk.Label(header_frame, text="📋 SESSION SUMMARY REPORT", 
                              font=('Arial', 20, 'bold'), bg='#1a1a2e', fg='#00ff88')
        title_label.pack()
        
        # Stats cards
        stats_frame = tk.Frame(report_window, bg='#1a1a2e')
        stats_frame.pack(fill=tk.X, padx=30, pady=(0, 20))
        
        # Duration card
        duration_card = tk.Frame(stats_frame, bg='#16213e', relief='ridge', bd=1)
        duration_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        tk.Label(duration_card, text="⏱️", font=('Arial', 24), bg='#16213e', fg='#4ecdc4').pack(pady=(10, 0))
        tk.Label(duration_card, text=f"{int(session_duration/60)}:{int(session_duration%60):02d}", 
                font=('Arial', 18, 'bold'), bg='#16213e', fg='#ffffff').pack()
        tk.Label(duration_card, text="Duration", font=('Arial', 10), bg='#16213e', fg='#a0a0a0').pack(pady=(0, 10))
        
        # Total distractions card
        total_card = tk.Frame(stats_frame, bg='#16213e', relief='ridge', bd=1)
        total_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 5))
        
        tk.Label(total_card, text="📊", font=('Arial', 24), bg='#16213e', fg='#ff6b6b').pack(pady=(10, 0))
        tk.Label(total_card, text=str(total_distractions), 
                font=('Arial', 18, 'bold'), bg='#16213e', fg='#ffffff').pack()
        tk.Label(total_card, text="Total Events", font=('Arial', 10), bg='#16213e', fg='#a0a0a0').pack(pady=(0, 10))
        
        # Attention score card
        attention_score = max(0, 100 - (total_distractions / total_minutes * 10)) if total_minutes > 0 else 100
        score_card = tk.Frame(stats_frame, bg='#16213e', relief='ridge', bd=1)
        score_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        tk.Label(score_card, text="🎯", font=('Arial', 24), bg='#16213e', fg='#45b7d1').pack(pady=(10, 0))
        tk.Label(score_card, text=f"{attention_score:.0f}/100", 
                font=('Arial', 18, 'bold'), bg='#16213e', fg='#ffffff').pack()
        tk.Label(score_card, text="Attention Score", font=('Arial', 10), bg='#16213e', fg='#a0a0a0').pack(pady=(0, 10))
        
        # Detailed breakdown
        breakdown_frame = tk.Frame(report_window, bg='#16213e', relief='ridge', bd=1)
        breakdown_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=(0, 20))
        
        breakdown_title = tk.Label(breakdown_frame, text="📈 Detailed Breakdown", 
                                  font=('Arial', 16, 'bold'), bg='#16213e', fg='#ffffff')
        breakdown_title.pack(pady=(15, 10))
        
        # Create breakdown content
        breakdown_content = tk.Frame(breakdown_frame, bg='#16213e')
        breakdown_content.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 15))
        
        breakdown_items = [
            ("😴 Drowsiness Events:", total_drowsy, "#ff6b6b"),
            ("🗣️ Talking Events:", total_talking, "#4ecdc4"),
            ("👀 Looking Away Events:", total_gaze, "#45b7d1"),
            ("📊 Average per Minute:", f"{total_distractions/total_minutes:.1f}", "#ffffff")
        ]
        
        for i, (label, value, color) in enumerate(breakdown_items):
            item_frame = tk.Frame(breakdown_content, bg='#16213e')
            item_frame.pack(fill=tk.X, pady=5)
            
            tk.Label(item_frame, text=label, font=('Arial', 12), bg='#16213e', fg='#a0a0a0').pack(side=tk.LEFT)
            tk.Label(item_frame, text=str(value), font=('Arial', 12, 'bold'), bg='#16213e', fg=color).pack(side=tk.RIGHT)
        
        # Recommendations
        recommendations_frame = tk.Frame(report_window, bg='#16213e', relief='ridge', bd=1)
        recommendations_frame.pack(fill=tk.X, padx=30, pady=(0, 20))
        
        rec_title = tk.Label(recommendations_frame, text="💡 Recommendations", 
                            font=('Arial', 16, 'bold'), bg='#16213e', fg='#ffffff')
        rec_title.pack(pady=(15, 10))
        
        # Generate recommendations based on primary distraction type
        if total_drowsy > total_talking and total_drowsy > total_gaze:
            recommendation = "• Take regular breaks every 25-30 minutes\n• Ensure adequate sleep (7-9 hours)\n• Maintain proper lighting in study area"
        elif total_talking > total_gaze:
            recommendation = "• Find a quieter study environment\n• Use noise-canceling headphones\n• Set specific times for breaks and conversations"
        else:
            recommendation = "• Minimize visual distractions in study area\n• Practice focus techniques like Pomodoro\n• Position yourself to face away from distracting elements"
        
        if attention_score >= 80:
            recommendation += "\n• Excellent attention! Keep up the good work."
        elif attention_score >= 60:
            recommendation += "\n• Good attention with room for improvement."
        else:
            recommendation += "\n• Consider implementing better study habits."
        
        rec_text = tk.Label(recommendations_frame, text=recommendation, 
                           font=('Arial', 10), bg='#16213e', fg='#a0a0a0', 
                           justify=tk.LEFT, wraplength=600)
        rec_text.pack(padx=20, pady=(0, 15))
        
        # Buttons
        button_frame = tk.Frame(report_window, bg='#1a1a2e')
        button_frame.pack(fill=tk.X, padx=30, pady=(0, 20))
        
        # Modern button styling
        analytics_btn = tk.Button(button_frame, text="📊 View Analytics", 
                                 command=self.generate_graphs,
                                 bg='#3742fa', fg='#ffffff', font=('Arial', 12, 'bold'),
                                 relief='flat', padx=20, pady=10, cursor='hand2')
        analytics_btn.pack(side=tk.LEFT, padx=(0, 15))
        
        save_btn = tk.Button(button_frame, text="💾 Save Report", 
                            command=lambda: self.save_report(report_window, session_duration, 
                                                           total_distractions, total_drowsy, 
                                                           total_talking, total_gaze, attention_score),
                            bg='#00ff88', fg='#000000', font=('Arial', 12, 'bold'),
                            relief='flat', padx=20, pady=10, cursor='hand2')
        save_btn.pack(side=tk.LEFT, padx=(0, 15))
        
        close_btn = tk.Button(button_frame, text="✖️ Close", 
                             command=report_window.destroy,
                             bg='#ff4757', fg='#ffffff', font=('Arial', 12, 'bold'),
                             relief='flat', padx=20, pady=10, cursor='hand2')
        close_btn.pack(side=tk.RIGHT)
    
    def save_report(self, parent_window, session_duration, total_distractions, 
                   total_drowsy, total_talking, total_gaze, attention_score):
        """Save report to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"distraction_report_{timestamp}.txt"
            
            report_content = f"""
{'='*60}
STUDENT DISTRACTOR DETECTOR - SESSION REPORT
{'='*60}
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

SESSION SUMMARY:
- Duration: {int(session_duration/60)} minutes {int(session_duration%60)} seconds
- Total Distraction Events: {total_distractions}
- Attention Score: {attention_score:.1f}/100

DETAILED BREAKDOWN:
- Drowsiness Events: {total_drowsy}
- Talking Events: {total_talking}
- Looking Away Events: {total_gaze}

MINUTE-BY-MINUTE DATA:
Minutes: {self.log_data['minutes']}
Drowsy Events: {self.log_data['drowsy_events']}
Talking Events: {self.log_data['talking_events']}
Gaze Events: {self.log_data['gaze_events']}
Total per Minute: {self.log_data['total_distractions']}

PERFORMANCE ANALYSIS:
"""
            
            if attention_score >= 80:
                report_content += "Excellent focus and attention throughout the session."
            elif attention_score >= 60:
                report_content += "Good attention with some areas for improvement."
            else:
                report_content += "Significant distractions detected. Consider environmental changes."
            
            report_content += f"\n{'='*60}\n"
            
            with open(filename, 'w') as f:
                f.write(report_content)
            
            messagebox.showinfo("Success", f"Report saved successfully!\nFile: {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save report:\n{str(e)}")
    
    def run(self):
        """Start the GUI application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        """Handle application closing"""
        if self.is_running:
            result = messagebox.askyesno("Confirm Exit", 
                                       "Detection is still running. Stop detection and exit?")
            if result:
                self.stop_detection()
                self.root.destroy()
        else:
            self.root.destroy()

# Run the application
if __name__ == "__main__":
    try:
        app = DistractorDetectorGUI()
        app.run()
    except Exception as e:
        print(f"Error starting application: {e}")
        messagebox.showerror("Error", f"Failed to start application:\n{str(e)}")