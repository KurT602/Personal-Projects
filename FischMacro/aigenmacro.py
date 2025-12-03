main_roi1 = {
            'left': 392,
            'top': 747,
            'width': 1110,
            'height': 40
        }

main_roi = {
            'left': 510,
            'top': 800,
            'width': 950,
            'height': 40
        }

import cv2
import numpy as np
import pyautogui
import mss
import time

class OptimizedFishingBarDetector:
    def __init__(self):
        # Define the region of interest (ROI) where the fishing bar appears
        self.roi = main_roi
        # Initialize MSS for fast screen capture
        self.sct = mss.mss()
        
        # Cache for optimization
        self.frame_skip = 2  # Process every Nth frame
        self.frame_count = 0
        
    def capture_screen_fast(self):
        """Fast screen capture using MSS (much faster than pyautogui)"""
        monitor = {
            'left': self.roi['left'],
            'top': self.roi['top'],
            'width': self.roi['width'],
            'height': self.roi['height']
        }
        
        # Capture and convert to numpy array
        screenshot = self.sct.grab(monitor)
        frame = np.array(screenshot)
        
        # Convert BGRA to BGR
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    
    def detect_bar_position_optimized(self, frame):
        """
        Detect the vertical gray bar (indicator) within the white track
        """
        # Downsample for faster processing (optional)
        small_frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
        
        # Convert to grayscale for better detection of the gray vertical bar
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # The vertical bar appears to be darker than the white background
        # Use edge detection to find the vertical bar
        edges = cv2.Canny(gray, 50, 150)
        
        # Find vertical lines using contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_candidate = None
        best_score = 0
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter for vertical bar characteristics:
            # - Should be taller than it is wide (aspect ratio)
            # - Should span most of the height of the ROI
            aspect_ratio = h / (w + 1)  # +1 to avoid division by zero
            
            if aspect_ratio > 2 and h > small_frame.shape[0] * 0.5:  # Vertical and tall enough
                score = h  # Prefer taller bars
                if score > best_score:
                    best_score = score
                    best_candidate = (x, y, w, h)
        
        if best_candidate:
            x, y, w, h = best_candidate
            # Scale back to original coordinates
            center_x = (x + w // 2) * 2
            return center_x, (x*2, y*2, w*2, h*2)
        
        return None, None
    
    def preview_region(self, duration=5):
        """Static preview showing single screenshot for specified duration"""
        print(f"Showing preview for {duration} seconds...")
        print(f"ROI: left={self.roi['left']}, top={self.roi['top']}, "
              f"width={self.roi['width']}, height={self.roi['height']}")
        
        window_name = 'ROI Preview - Press Q to close'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Full screen monitor config
        full_monitor = self.sct.monitors[1]  # Primary monitor
        
        # Capture ONE screenshot
        screenshot = self.sct.grab(full_monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        # Draw rectangle
        cv2.rectangle(
            frame,
            (self.roi['left'], self.roi['top']),
            (self.roi['left'] + self.roi['width'], 
             self.roi['top'] + self.roi['height']),
            (0, 255, 0),
            3
        )
        
        # Add text
        cv2.putText(
            frame,
            'Detection Region',
            (self.roi['left'], self.roi['top'] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Resize if needed
        if frame.shape[1] > 1920:
            scale = 1920 / frame.shape[1]
            new_size = (int(frame.shape[1] * scale), int(frame.shape[0] * scale))
            frame = cv2.resize(frame, new_size)
        
        # Show the static image
        cv2.imshow(window_name, frame)
        
        # Wait for duration or 'q' key
        start_time = time.time()
        while time.time() - start_time < duration:
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        print("Preview closed.")
    
    def track_bar_movement(self, duration=10, target_fps=30):
        """
        Optimized tracking with frame rate control
        """
        print(f"Starting optimized detection for {duration} seconds...")
        start_time = time.time()
        positions = []
        frame_time = 1.0 / target_fps
        last_frame = None
        
        while time.time() - start_time < duration:
            loop_start = time.time()
            
            # Fast screen capture
            frame = self.capture_screen_fast()
            last_frame = frame.copy()  # Save the last frame
            
            # Detect position
            center_x, bbox = self.detect_bar_position_optimized(frame)
            
            if center_x is not None:
                positions.append({
                    'time': time.time() - start_time,
                    'x_position': center_x
                })
                print(f"Bar at x={center_x}")
            
            # Frame rate limiting
            elapsed = time.time() - loop_start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
        
        # Show final visualization
        if positions and last_frame is not None:
            self.show_detection_results(last_frame, positions)
        
        return positions
    
    def show_detection_results(self, frame, positions):
        """
        Show a screenshot with detected bar positions marked
        """
        print("\n=== Showing Detection Results ===")
        
        # Get the last detected position
        last_pos = positions[-1]['x_position']
        
        # Get average position
        avg_pos = int(np.mean([p['x_position'] for p in positions]))
        
        # Draw the last detected position (green line)
        cv2.line(frame, (last_pos, 0), (last_pos, frame.shape[0]), (0, 255, 0), 3)
        cv2.putText(frame, f"Last: {last_pos}", (last_pos + 5, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw the average position (blue line)
        cv2.line(frame, (avg_pos, 0), (avg_pos, frame.shape[0]), (255, 0, 0), 2)
        cv2.putText(frame, f"Avg: {avg_pos}", (avg_pos + 5, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Draw range rectangle (showing min to max positions)
        min_pos = min(p['x_position'] for p in positions)
        max_pos = max(p['x_position'] for p in positions)
        cv2.rectangle(frame, (min_pos, 0), (max_pos, frame.shape[0]), (0, 255, 255), 2)
        cv2.putText(frame, f"Range: {min_pos}-{max_pos}", (10, frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Show the result
        window_name = 'Detection Results - Press any key to close'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, frame)
        cv2.waitKey(0)  # Wait for any key press
        cv2.destroyAllWindows()
        print("Results window closed.")
    
    def calibrate_colors(self):
        """Helper to find HSV color ranges"""
        print("Click on the fishing bar to get HSV values. Press 'q' to quit.")
        
        window_name = 'Calibration - Click on bar'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                hsv_value = param[y, x]
                print(f"HSV at ({x}, {y}): {hsv_value}")
        
        while True:
            frame = self.capture_screen_fast()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            cv2.imshow(window_name, frame)
            cv2.setMouseCallback(window_name, mouse_callback, hsv)
            
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
    
    def cleanup(self):
        """Clean up resources"""
        self.sct.close()


# ALTERNATIVE: Ultra-lightweight pixel sampling method
class LightweightBarDetector:
    """
    Even lighter approach: scan for the dark vertical bar
    This is MUCH faster but less accurate
    """
    def __init__(self):
        self.roi = main_roi
        self.sct = mss.mss()
        
    def find_vertical_bar(self):
        """
        Scan horizontally to find the vertical gray bar
        Uses edge detection for better accuracy
        """
        monitor = {
            'left': self.roi['left'],
            'top': self.roi['top'],
            'width': self.roi['width'],
            'height': self.roi['height']
        }
        
        screenshot = self.sct.grab(monitor)
        img = np.array(screenshot)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        
        # Use edge detection to find vertical edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Sum edges vertically to find x-positions with most vertical edges
        vertical_sums = np.sum(edges, axis=0)
        
        # Find peaks in the vertical sums (these are likely vertical bars)
        # Use a threshold to filter noise
        threshold = np.max(vertical_sums) * 0.5
        
        if np.max(vertical_sums) > 0:
            # Find all positions above threshold
            peaks = np.where(vertical_sums > threshold)[0]
            
            if len(peaks) > 0:
                # Find groups of consecutive peaks (actual bars)
                groups = []
                current_group = [peaks[0]]
                
                for i in range(1, len(peaks)):
                    if peaks[i] - peaks[i-1] <= 5:  # Within 5 pixels = same bar
                        current_group.append(peaks[i])
                    else:
                        groups.append(current_group)
                        current_group = [peaks[i]]
                groups.append(current_group)
                
                # Find the most prominent group (tallest vertical bar)
                best_group = max(groups, key=lambda g: np.sum(vertical_sums[g]))
                
                # Return center of the group
                return int(np.mean(best_group))
        
        return None
    
    def track_lightweight(self, duration=10):
        """Ultra-fast tracking using pixel scanning"""
        start_time = time.time()
        positions = []
        last_frame = None
        
        monitor = {
            'left': self.roi['left'],
            'top': self.roi['top'],
            'width': self.roi['width'],
            'height': self.roi['height']
        }
        
        while time.time() - start_time < duration:
            screenshot = self.sct.grab(monitor)
            last_frame = np.array(screenshot)
            last_frame = cv2.cvtColor(last_frame, cv2.COLOR_BGRA2BGR)
            
            x_pos = self.find_vertical_bar()
            if x_pos is not None:
                positions.append({
                    'time': time.time() - start_time,
                    'x_position': x_pos
                })
                print(f"Vertical bar detected at x={x_pos}")
            
            time.sleep(0.01)  # 100 FPS sampling rate
        
        # Show final visualization
        if positions and last_frame is not None:
            self.show_detection_results(last_frame, positions)
        
        return positions
    
    def show_detection_results(self, frame, positions):
        """Show detection results visualization"""
        print("\n=== Showing Detection Results ===")
        
        last_pos = positions[-1]['x_position']
        avg_pos = int(np.mean([p['x_position'] for p in positions]))
        
        # Draw lines and labels
        cv2.line(frame, (last_pos, 0), (last_pos, frame.shape[0]), (0, 255, 0), 3)
        cv2.putText(frame, f"Last: {last_pos}", (last_pos + 5, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.line(frame, (avg_pos, 0), (avg_pos, frame.shape[0]), (255, 0, 0), 2)
        cv2.putText(frame, f"Avg: {avg_pos}", (avg_pos + 5, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        min_pos = min(p['x_position'] for p in positions)
        max_pos = max(p['x_position'] for p in positions)
        cv2.rectangle(frame, (min_pos, 0), (max_pos, frame.shape[0]), (0, 255, 255), 2)
        cv2.putText(frame, f"Range: {min_pos}-{max_pos}", (10, frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        window_name = 'Detection Results - Press any key to close'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Results window closed.")
    
    def cleanup(self):
        self.sct.close()
    
    def preview_region(self):
        """Preview using the optimized detector's method"""
        # Create temporary optimized detector for preview
        temp_detector = OptimizedFishingBarDetector()
        temp_detector.roi = self.roi
        temp_detector.preview_region(duration=5)


# Example usage
if __name__ == "__main__":
    print("Choose detector type:")
    print("1. Optimized detector (good balance)")
    print("2. Lightweight detector (fastest, less accurate)")
    print("3. Calibrate colors (find HSV values)")
    
    choice = input("Enter choice (1, 2, or 3): ").strip()
    
    if choice == "3":
        # Calibration mode
        detector = OptimizedFishingBarDetector()
        
        print("\n=== Color Calibration Mode ===")
        print("First, let's preview the region...")
        time.sleep(2)
        detector.preview_region(duration=5)
        
        print("\nNow entering calibration mode...")
        print("Click on the vertical bar to get its HSV values")
        print("Press 'Q' when done")
        time.sleep(2)
        detector.calibrate_colors()
        
        print("\nCalibration complete!")
        print("Update the color ranges in detect_bar_position_optimized() with the HSV values you found.")
        detector.cleanup()
        
    elif choice == "2":
        detector = LightweightBarDetector()
        
        # Show preview for lightweight detector too
        print("\n=== Preview Region ===")
        print("Showing ROI in 2 seconds...")
        time.sleep(2)
        detector.preview_region()
        
        print("\n=== Starting Lightweight Detection ===")
        time.sleep(2)
        positions = detector.track_lightweight(duration=10)
        
        # Analysis
        if positions:
            print(f"\n--- Results ---")
            print(f"Positions recorded: {len(positions)}")
            print(f"Range: {min(p['x_position'] for p in positions)} to {max(p['x_position'] for p in positions)}")
        
        detector.cleanup()
        
    else:
        detector = OptimizedFishingBarDetector()
        
        print("\n=== Preview Region ===")
        print("Showing ROI in 2 seconds...")
        time.sleep(2)
        detector.preview_region(duration=5)
        
        print("\n=== Track Movement ===")
        time.sleep(2)
        positions = detector.track_bar_movement(duration=10, target_fps=30)
        
        # Analysis
        if positions:
            print(f"\n--- Results ---")
            print(f"Positions recorded: {len(positions)}")
            print(f"Range: {min(p['x_position'] for p in positions)} to {max(p['x_position'] for p in positions)}")
        
        detector.cleanup()