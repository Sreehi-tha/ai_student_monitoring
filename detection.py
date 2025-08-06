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
warnings.filterwarnings('ignore')

class DistractorDetector:
    def __init__(self):
        # Initialize face detector and landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        
        # Eye aspect ratio thresholds
        self.EAR_THRESHOLD = 0.25
        self.CONSECUTIVE_FRAMES = 20
        
        # Mouth aspect ratio threshold for talking detection
        self.MAR_THRESHOLD = 0.7
        self.TALKING_FRAMES = 5
        
        # Gaze detection parameters - Reduced frames for more reliable detection
        self.GAZE_THRESHOLD = 0.2  # Lower = more sensitive
        self.GAZE_FRAMES = 15  # Reduced from 20 to 15 for easier triggering
        self.HEAD_POSE_YAW_THRESHOLD = 15  # degrees - strict for left/right
        self.HEAD_POSE_PITCH_THRESHOLD = 15  # degrees - strict for up/down
        self.EYE_GAZE_THRESHOLD = 0.3  # Lower threshold for eye movement
        self.FACE_DISPLACEMENT_THRESHOLD = 0.2  # Lower threshold for displacement
        
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
        
        # Minute-based tracking
        self.current_minute_data = {
            'drowsy': 0,
            'talking': 0,
            'gaze': 0
        }
        self.last_minute_log = time.time()
        
        # Initialize pygame for audio alerts
        pygame.mixer.init()
        
        # Eye landmarks indices
        self.LEFT_EYE = list(range(36, 42))
        self.RIGHT_EYE = list(range(42, 48))
        self.MOUTH = list(range(48, 68))
        
    def eye_aspect_ratio(self, eye):
        """Calculate eye aspect ratio"""
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear
    
    def mouth_aspect_ratio(self, mouth):
        """Calculate mouth aspect ratio for talking detection"""
        A = dist.euclidean(mouth[2], mouth[10])  # 50, 58
        B = dist.euclidean(mouth[4], mouth[8])   # 52, 56
        C = dist.euclidean(mouth[0], mouth[6])   # 48, 54
        mar = (A + B) / (2.0 * C)
        return mar
    
    def get_head_pose(self, landmarks, frame_shape):
        """Calculate head pose angles (pitch, yaw, roll)"""
        # 3D model points for head pose estimation
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])
        
        # 2D image points from facial landmarks
        image_points = np.array([
            landmarks[30],     # Nose tip
            landmarks[8],      # Chin
            landmarks[36],     # Left eye left corner
            landmarks[45],     # Right eye right corner
            landmarks[48],     # Left mouth corner
            landmarks[54]      # Right mouth corner
        ], dtype="double")
        
        # Camera internals
        height, width = frame_shape[:2]
        focal_length = width
        center = (width/2, height/2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")
        
        # Assuming no lens distortion
        dist_coeffs = np.zeros((4,1))
        
        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs)
        
        if success:
            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            
            # Calculate Euler angles
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
            
            # Convert to degrees
            pitch = np.degrees(x)
            yaw = np.degrees(y)
            roll = np.degrees(z)
            
            return pitch, yaw, roll
        
        return 0, 0, 0
    
    def get_eye_gaze_ratio(self, eye_landmarks):
        """Calculate eye gaze ratio based on pupil position relative to eye corners"""
        # Get eye corner points
        left_corner = eye_landmarks[0]
        right_corner = eye_landmarks[3]
        
        # Get top and bottom points
        top_point = eye_landmarks[1]
        bottom_point = eye_landmarks[5] if len(eye_landmarks) > 5 else eye_landmarks[4]
        
        # Calculate eye center
        eye_center = np.mean(eye_landmarks, axis=0)
        
        # Calculate eye width and height
        eye_width = np.linalg.norm(right_corner - left_corner)
        eye_height = np.linalg.norm(top_point - bottom_point)
        
        if eye_width == 0:
            return 0
        
        # Calculate horizontal position ratio
        horizontal_ratio = (eye_center[0] - left_corner[0]) / eye_width
        
        # Normalize to [-1, 1] where 0 is center
        gaze_ratio = (horizontal_ratio - 0.5) * 2
        
        return abs(gaze_ratio)
    
    def is_looking_away(self, landmarks, frame_shape):
        """Improved gaze detection with more lenient thresholds"""
        # Method 1: Head pose estimation with separate thresholds for yaw and pitch
        pitch, yaw, roll = self.get_head_pose(landmarks, frame_shape)
        
        # More lenient head pose detection - only significant turns
        significant_yaw = abs(yaw) > self.HEAD_POSE_YAW_THRESHOLD  # Left/right turn
        significant_pitch = abs(pitch) > self.HEAD_POSE_PITCH_THRESHOLD  # Up/down turn
        head_looking_away = significant_yaw or significant_pitch
        
        # Method 2: Eye gaze ratio - only for extreme eye movements
        left_eye = landmarks[self.LEFT_EYE]
        right_eye = landmarks[self.RIGHT_EYE]
        
        left_gaze_ratio = self.get_eye_gaze_ratio(left_eye)
        right_gaze_ratio = self.get_eye_gaze_ratio(right_eye)
        avg_gaze_ratio = (left_gaze_ratio + right_gaze_ratio) / 2
        
        extreme_eye_movement = avg_gaze_ratio > self.EYE_GAZE_THRESHOLD
        
        # Method 3: Face position - only for significant displacement
        face_center = np.mean(landmarks, axis=0)
        frame_center = np.array([frame_shape[1]/2, frame_shape[0]/2])
        face_displacement = np.linalg.norm(face_center - frame_center)
        
        # Normalize by frame diagonal
        frame_diagonal = np.sqrt(frame_shape[0]**2 + frame_shape[1]**2)
        normalized_displacement = face_displacement / frame_diagonal
        
        significant_displacement = normalized_displacement > self.FACE_DISPLACEMENT_THRESHOLD
        
        # More conservative combination logic:
        # Only trigger if there's significant head movement OR
        # both extreme eye movement AND significant displacement
        looking_away = head_looking_away or (extreme_eye_movement and significant_displacement)
        
        # Additional check: Only consider it looking away if multiple indicators agree
        # This reduces false positives significantly
        confidence_score = 0
        if significant_yaw:
            confidence_score += 2  # Yaw is most reliable
        if significant_pitch:
            confidence_score += 1.5
        if extreme_eye_movement:
            confidence_score += 1
        if significant_displacement:
            confidence_score += 0.5
            
        # Only trigger if confidence is high enough
        final_looking_away = confidence_score >= 2.0
        
        return final_looking_away, pitch, yaw, avg_gaze_ratio
    
    def get_gaze_direction(self, landmarks, frame_shape):
        """Legacy method - kept for compatibility"""
        looking_away, pitch, yaw, gaze_ratio = self.is_looking_away(landmarks, frame_shape)
        return gaze_ratio if looking_away else 0
    
    def generate_beep_sound(self, frequency, duration, volume=0.5):
        """Generate a beep sound with specified frequency and duration"""
        try:
            sample_rate = 22050
            frames = int(duration * sample_rate)
            arr = np.zeros(frames)
            
            for i in range(frames):
                arr[i] = volume * np.sin(2 * np.pi * frequency * i / sample_rate)
            
            # Convert to 16-bit integers
            arr = (arr * 32767).astype(np.int16)
            
            # Create stereo sound
            stereo_arr = np.zeros((frames, 2), dtype=np.int16)
            stereo_arr[:, 0] = arr
            stereo_arr[:, 1] = arr
            
            # Play sound
            sound = pygame.sndarray.make_sound(stereo_arr)
            sound.play()
            pygame.time.wait(int(duration * 1000))  # Wait for sound to finish
            
        except Exception as e:
            print(f"Audio error: {e}")
            # Fallback to system beep
            print("\a")
    
    def play_alert_sound(self, alert_type):
        """Play different alert sounds for different distractions with fallback"""
        print(f"🔊 PLAYING {alert_type.upper()} ALERT SOUND!")
        try:
            if alert_type == "drowsy":
                # Low frequency alarm for drowsiness - 3 beeps
                for i in range(3):
                    self.generate_beep_sound(400, 0.3, 0.7)
                    if i < 2:  # Don't wait after last beep
                        time.sleep(0.1)
            elif alert_type == "talking":
                # Medium frequency alarm for talking - 2 beeps
                for i in range(2):
                    self.generate_beep_sound(600, 0.2, 0.6)
                    if i < 1:  # Don't wait after last beep
                        time.sleep(0.1)
            elif alert_type == "gaze":
                # High frequency alarm for looking away - 1 long beep
                self.generate_beep_sound(800, 0.5, 0.8)
                print("🚨 GAZE ALERT SOUND PLAYED! 🚨")
        except Exception as e:
            print(f"Alert sound error: {e}")
            # Enhanced fallback system beeps
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
                # Final fallback
                if alert_type == "drowsy":
                    print("\a" * 3)
                elif alert_type == "talking":
                    print("\a" * 2)
                elif alert_type == "gaze":
                    print("\a" * 1)
    
    def log_minute_data(self):
        """Log data every minute with debugging - fixed duplicate logging"""
        current_time = time.time()
        if current_time - self.last_minute_log >= 60:  # Every minute
            minutes_elapsed = len(self.log_data['minutes']) + 1  # Use array length instead of time calculation
            
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
            
            print(f"   Total distractions this minute: {total}")
            print(f"   Total logged minutes so far: {len(self.log_data['minutes'])}")
            
            # Reset minute counters
            self.current_minute_data = {'drowsy': 0, 'talking': 0, 'gaze': 0}
            self.last_minute_log = current_time
    
    def generate_graphs(self):
        """Generate comprehensive graphs for distraction analysis"""
        if len(self.log_data['minutes']) == 0:
            return
        
        plt.figure(figsize=(15, 10))
        
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
            colors = ['red', 'blue', 'green']
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
    
    def process_frame(self, frame):
        """Process a single frame for all distraction types"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        status_text = []
        
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
                    status_text.append("DROWSINESS DETECTED!")
                    cv2.putText(frame, "DROWSY!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 0, 255), 2)
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
                    status_text.append("TALKING DETECTED!")
                    cv2.putText(frame, "TALKING!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 255, 255), 2)
            else:
                self.talking_counter = 0
                self.talking_alert = False
            
            # Gaze detection - Strict detection with debug output
            looking_away, pitch, yaw, gaze_ratio = self.is_looking_away(landmarks, frame.shape)
            
            if looking_away:
                self.gaze_counter += 1
                print(f"LOOKING AWAY DETECTED - Counter: {self.gaze_counter}/{self.GAZE_FRAMES}")
                if self.gaze_counter >= self.GAZE_FRAMES:
                    if not self.gaze_alert:
                        self.gaze_alert = True
                        self.current_minute_data['gaze'] += 1
                        print(f"🚨 GAZE ALERT TRIGGERED! Event #{self.current_minute_data['gaze']} this minute")
                        # Play alert sound in main thread to ensure it works
                        self.play_alert_sound("gaze")
                    status_text.append("LOOKING AWAY!")
                    cv2.putText(frame, "LOOKING AWAY!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (255, 0, 0), 2)
            else:
                if self.gaze_counter > 0:
                    print(f"Looking away counter reset from {self.gaze_counter}")
                self.gaze_counter = 0
                self.gaze_alert = False
            
            # Draw eye contours
            cv2.drawContours(frame, [cv2.convexHull(left_eye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(right_eye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (255, 0, 0), 1)
            
            # Display enhanced metrics with strict status indicators
            cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
            
            # Color-coded head pose indicators with strict thresholds
            yaw_color = (0, 0, 255) if abs(yaw) > self.HEAD_POSE_YAW_THRESHOLD else (0, 255, 0)
            pitch_color = (0, 0, 255) if abs(pitch) > self.HEAD_POSE_PITCH_THRESHOLD else (0, 255, 0)
            gaze_color = (0, 0, 255) if gaze_ratio > self.EYE_GAZE_THRESHOLD else (0, 255, 0)
            
            cv2.putText(frame, f"Head Yaw: {yaw:.1f}°", (300, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, yaw_color, 2)
            cv2.putText(frame, f"Head Pitch: {pitch:.1f}°", (300, 120), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, pitch_color, 2)
            cv2.putText(frame, f"Eye Gaze: {gaze_ratio:.2f}", (300, 150), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, gaze_color, 2)
            
            # Debug info for looking away detection with better counter display
            if looking_away:
                cv2.putText(frame, f"LOOKING AWAY: {self.gaze_counter}/{self.GAZE_FRAMES}", 
                           (300, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                # Add progress bar visualization
                progress = self.gaze_counter / self.GAZE_FRAMES
                bar_width = 100
                bar_filled = int(bar_width * progress)
                cv2.rectangle(frame, (300, 190), (300 + bar_width, 200), (100, 100, 100), -1)
                cv2.rectangle(frame, (300, 190), (300 + bar_filled, 200), (0, 0, 255), -1)
            
            # Show current detection status
            status = "FOCUSED" if not looking_away else "DISTRACTED"
            status_color = (0, 255, 0) if not looking_away else (0, 0, 255)
            cv2.putText(frame, f"Status: {status}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, status_color, 2)
        
        # Display session info
        elapsed_minutes = int((time.time() - self.start_time) / 60)
        cv2.putText(frame, f"Session: {elapsed_minutes} min", (10, frame.shape[0] - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display current minute stats
        cv2.putText(frame, f"This min - D:{self.current_minute_data['drowsy']} "
                          f"T:{self.current_minute_data['talking']} "
                          f"G:{self.current_minute_data['gaze']}", 
                   (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def run(self):
        """Main execution loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Distractor Detector Started!")
        print("Press 'q' to quit, 'g' to generate graphs")
        print("Detecting: Drowsiness, Talking, Looking Away")
        print("Audio alerts enabled - Different beep patterns for each distraction type")
        print("  - Drowsiness: 3 low-pitched beeps (400Hz)")
        print("  - Talking: 2 medium-pitched beeps (600Hz)")
        print("  - Looking Away: 1 high-pitched beep (800Hz)")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = self.process_frame(frame)
            self.log_minute_data()
            
            cv2.imshow('Enhanced Distractor Detector', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('g'):
                self.generate_graphs()
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Generate final report
        self.generate_final_report()
    
    def generate_final_report(self):
        """Generate a final session report with current minute data included - fixed duplicates"""
        # Include current minute data if session ends mid-minute
        current_time = time.time()
        session_duration = current_time - self.start_time
        
        # Only add incomplete minute if it has data and hasn't been logged yet
        if (session_duration % 60 > 10) and (  # At least 10 seconds into the minute
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
        
        total_minutes = max(1, len(self.log_data['minutes']))  # Use actual logged minutes
        total_drowsy = sum(self.log_data['drowsy_events'])
        total_talking = sum(self.log_data['talking_events'])
        total_gaze = sum(self.log_data['gaze_events'])
        total_distractions = total_drowsy + total_talking + total_gaze
        
        print("\n" + "="*50)
        print("SESSION SUMMARY REPORT")
        print("="*50)
        print(f"Total Session Duration: {int(session_duration/60)} minutes {int(session_duration%60)} seconds")
        print(f"Data Points Collected: {len(self.log_data['minutes'])}")
        print(f"Total Distractions: {total_distractions}")
        print(f"  - Drowsiness Events: {total_drowsy}")
        print(f"  - Talking Events: {total_talking}")
        print(f"  - Looking Away Events: {total_gaze}")
        
        if total_minutes > 0:
            print(f"Average Distractions per Minute: {total_distractions/total_minutes:.2f}")
        
        # Attention score (100 - distraction percentage)
        if total_minutes > 0:
            attention_score = max(0, 100 - (total_distractions / total_minutes * 10))
            print(f"Attention Score: {attention_score:.1f}/100")
        
        # Debug info
        print("\nDebug Info:")
        print(f"Minutes logged: {self.log_data['minutes']}")
        print(f"Drowsy events: {self.log_data['drowsy_events']}")
        print(f"Talking events: {self.log_data['talking_events']}")
        print(f"Gaze events: {self.log_data['gaze_events']}")
        print("="*50)
        
        # Generate final graphs if we have data
        if len(self.log_data['minutes']) > 0:
            self.generate_graphs()
        else:
            print("⚠️ No minute data available for graphs (session too short)")

# Usage
if __name__ == "__main__":
    # Note: You need to download shape_predictor_68_face_landmarks.dat
    # from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    
    try:
        detector = DistractorDetector()
        detector.run()
    except FileNotFoundError:
        print("Error: shape_predictor_68_face_landmarks.dat not found!")
        print("Please download it from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("Extract and place it in the same directory as this script.")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have all required libraries installed:")
        print("pip install opencv-python dlib numpy matplotlib scipy pygame")