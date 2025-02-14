import cv2
import mediapipe as mp
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

class TouchlessZoom:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0  
        )

        temp_cap = cv2.VideoCapture(0)
        self.frame_width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        temp_cap.release()

        self.scale_factor = 1.0
        self.min_scale = 0.05  
        self.max_scale = 150.0  
        self.prev_distance = None
        self.zoom_sensitivity = 200.0 
        self.smoothing_factor = 0.08  
        self.zoom_history = []
        self.history_size = 8 
        self.is_zooming = True
        self.ema_alpha = 0.15 
        self.prev_scale = 1.0
        self.target_scale = 1.0

        try:
           
            image_paths = [
                r'C:\Users\Admin\Desktop\sample.jpg',
                r'C:\Users\Admin\Desktop\opencv 2\sample.jpg',
                'sample.jpg'
            ]
            
            self.original_image = None
            for path in image_paths:
                img = cv2.imread(path)
                if img is not None:
                    self.original_image = img
                    print(f"Successfully loaded image from: {path}")
                    break
            
            if self.original_image is None:
                print("No image found, creating a test pattern...")
                self.original_image = np.zeros((800, 1200, 3), dtype=np.uint8)
                
                cv2.rectangle(self.original_image, (100, 100), (1100, 700), (0, 255, 0), 2)
                cv2.circle(self.original_image, (600, 400), 200, (0, 0, 255), -1)
                cv2.putText(self.original_image, 'Test Image', (450, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                cv2.putText(self.original_image, 'Move fingers to zoom', (400, 700), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                for x in range(0, 1200, 50):
                    cv2.line(self.original_image, (x, 0), (x, 800), (128, 128, 128), 1)
                for y in range(0, 800, 50):
                    cv2.line(self.original_image, (0, y), (1200, y), (128, 128, 128), 1)
                
                print("Created test pattern image")
            
            self.orig_height, self.orig_width = self.original_image.shape[:2]
            print(f"Image size: {self.orig_width} x {self.orig_height}")
            
        except Exception as e:
            print("Error during image initialization:", str(e))
            raise

    def calculate_finger_distance(self, hand_landmarks):
        """Calculate distance between thumb and index finger with additional smoothing"""
        thumb_tip = np.array([
            hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].x,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].y
        ])
        index_tip = np.array([
            hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y
        ])
        
        current_distance = np.linalg.norm(thumb_tip - index_tip)
        if hasattr(self, 'prev_raw_distance'):
            smooth_factor = 0.7  
            current_distance = (smooth_factor * self.prev_raw_distance + 
                              (1 - smooth_factor) * current_distance)
        
        self.prev_raw_distance = current_distance
        return current_distance

    def is_fist_closed(self, hand_landmarks):
        """Detect if hand is in a fist position"""
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        
        thumb_base = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_CMC]
        index_base = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        middle_base = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        
        thumb_height = thumb_tip.y - thumb_base.y
        index_height = index_tip.y - index_base.y
        middle_height = middle_tip.y - middle_base.y
        
        threshold = 0.05  
        return thumb_height > threshold and index_height > threshold and middle_height > threshold

    def update_zoom(self, distance):
        """Update zoom level based on finger distance with enhanced smoothing"""
        if not self.is_zooming:
            return

        if self.prev_distance is not None:
            distance_change = distance - self.prev_distance
            self.target_scale = self.scale_factor + (distance_change * self.zoom_sensitivity)
            
            self.target_scale = (self.target_scale * self.smoothing_factor + 
                               self.prev_scale * (1 - self.smoothing_factor))
            
            self.zoom_history.append(self.target_scale)
            if len(self.zoom_history) > self.history_size:
                self.zoom_history.pop(0)
            
            weights = np.linspace(1, 2, len(self.zoom_history))  
            weighted_avg = np.average(self.zoom_history, weights=weights)
            
            self.scale_factor = (self.ema_alpha * weighted_avg + 
                               (1 - self.ema_alpha) * self.scale_factor)
            
            self.scale_factor = np.clip(self.scale_factor, self.min_scale, self.max_scale)
            
            self.prev_scale = self.scale_factor
        
        self.prev_distance = distance

    def zoom_image(self):
        """Apply zoom to the image"""
        height, width = self.original_image.shape[:2]
        new_height = int(height * self.scale_factor)
        new_width = int(width * self.scale_factor)
        
        start_y = max(0, (new_height - height) // 2)
        start_x = max(0, (new_width - width) // 2)
        
        resized = cv2.resize(self.original_image, (new_width, new_height),
                           interpolation=cv2.INTER_LINEAR)
        
        if self.scale_factor > 1.0:
            zoomed = resized[start_y:start_y + height, start_x:start_x + width]
        else:
            pad_y = (height - new_height) // 2
            pad_x = (width - new_width) // 2
            zoomed = np.zeros((height, width, 3), dtype=np.uint8)
            zoomed[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = resized
            
        return zoomed

    def run(self):
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Could not open camera")
            
            print("Camera opened successfully! Press 'q' to quit, 'r' to reset zoom.")
            print("Instructions:")
            print("1. Move fingers apart/together to zoom in/out")
            print("2. Make a fist to lock zoom level")
            print("3. Press 'r' to reset zoom")
            print("4. Press 'q' to quit")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_rgb.flags.writeable = False  
                    results = self.hands.process(frame_rgb)
                    frame_rgb.flags.writeable = True

                    if results.multi_hand_landmarks:
                        hand_landmarks = results.multi_hand_landmarks[0]
                        
                        try:
                            if self.is_fist_closed(hand_landmarks):
                                if self.is_zooming:
                                    print("Zoom locked")
                                self.is_zooming = False
                            else:
                                if not self.is_zooming:
                                    print("Zoom unlocked")
                                self.is_zooming = True
                                distance = self.calculate_finger_distance(hand_landmarks)
                                self.update_zoom(distance)
                        except Exception as e:
                            logging.debug("Error processing hand landmarks: %s", str(e))
                            continue

                    zoomed_image = self.zoom_image()

                    cv2.imshow('Zoomed Image (Press q to quit)', zoomed_image)

                except Exception as e:
                    logging.debug("Error in main loop: %s", str(e))
                    continue

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  
                    print("Quitting...")
                    break
                elif key == ord('r'):  
                    self.scale_factor = 1.0
                    self.prev_distance = None
                    self.zoom_history = []
                    self.is_zooming = True
                    print("Zoom reset to 1.0x")

        except Exception as e:
            print("Fatal error:", str(e))
        
        finally:
            print("Cleaning up...")
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    zoom_controller = TouchlessZoom()
    zoom_controller.run()
