import cv2, mss, pyautogui
import numpy as np
from PIL import Image
from time import time, sleep

main_roi = {
    'left': 580,
    'top': 910,
    'width': 765,
    'height': 20
}

class FischMacro:
    def __init__(self):
        self.roi = main_roi
        self.sct = mss.mss()
        
        # Tracking variables
        self.bar_pos = 0
        self.bar_width = 0
        self.fish_pos = 0
        self.start_time = time()
        self.frame_count = 0
        
        # Tolerance for "good enough" positioning (pixels)
        self.tolerance = 15
        
    def capture_screen_fast(self):
        """Fast screen capture using MSS"""
        monitor = {
            'left': self.roi['left'],
            'top': self.roi['top'],
            'width': self.roi['width'],
            'height': self.roi['height']
        }
        screenshot = self.sct.grab(monitor)
        frame = np.array(screenshot)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    
    def find_bar(self, gray_array):
        """Find the controllable bar (usually white/bright rectangle)"""
        _, binary = cv2.threshold(gray_array, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # Find the largest contour (the bar)
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            
            self.bar_pos = x + (w / 2)  # Center of bar
            self.bar_width = w
            
            return x, y, w, h
        
        return None
    
    def find_fish(self, gray_array):
        """Find the fish indicator (vertical line)"""
        edges = cv2.Canny(gray_array, 50, 150)
        vertical_sums = np.sum(edges, axis=0)
        
        if np.max(vertical_sums) == 0:
            return None
        
        threshold = np.max(vertical_sums) * 0.55
        peaks = np.where(vertical_sums > threshold)[0]
        
        if len(peaks) == 0:
            return None
        
        # Group consecutive peaks
        groups = []
        current_group = [peaks[0]]
        
        for i in range(1, len(peaks)):
            if peaks[i] - peaks[i-1] <= 10:
                current_group.append(peaks[i])
            else:
                groups.append(current_group)
                current_group = [peaks[i]]
        groups.append(current_group)
        
        # Get most prominent group
        best_group = max(groups, key=lambda g: np.sum(vertical_sums[g]))
        self.fish_pos = int(np.mean(best_group))
        
        return self.fish_pos
    
    def is_fish_in_bar(self, bar_rect):
        """Check if fish is within the bar boundaries"""
        if bar_rect is None:
            return False
        
        x, y, w, h = bar_rect
        bar_left = x + self.tolerance  # Add tolerance
        bar_right = x + w - self.tolerance
        
        return bar_left <= self.fish_pos <= bar_right
    
    def should_hold_mouse(self, bar_rect):
        """Determine if we should hold mouse button (move bar up)"""
        if bar_rect is None or self.fish_pos is None:
            return False
        
        x, y, w, h = bar_rect
        bar_center = x + (w / 2)
        
        # If fish is to the RIGHT of bar center, hold mouse (bar moves right)
        # If fish is to the LEFT of bar center, release mouse (bar moves left)
        return self.fish_pos > bar_center + self.tolerance
    
    def run(self, duration=10, show_debug=True):
        """Main tracking loop"""
        print(f"Starting macro for {duration} seconds...")
        start = time()
        
        try:
            while time() - start < duration:
                loop_start = time()
                
                # Capture and convert to grayscale
                frame = self.capture_screen_fast()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Find bar and fish
                bar_rect = self.find_bar(gray)
                fish_pos = self.find_fish(gray)
                
                # Control logic
                if bar_rect and fish_pos is not None:
                    in_bar = self.is_fish_in_bar(bar_rect)
                    
                    if in_bar:
                        # Fish is in the bar - we're good!
                        pyautogui.mouseUp()
                        status = "✓ IN BAR"
                    else:
                        # Fish is outside - need to move
                        if self.should_hold_mouse(bar_rect):
                            pyautogui.mouseDown()
                            status = "→ MOVING RIGHT"
                        else:
                            pyautogui.mouseUp()
                            status = "← MOVING LEFT"
                    
                    if show_debug:
                        x, y, w, h = bar_rect
                        bar_center = x + (w / 2)
                        print(f"{status} | Bar: {bar_center:.1f} | Fish: {fish_pos} | Diff: {fish_pos - bar_center:.1f}")
                
                self.frame_count += 1
                
                # Maintain frame rate (~60 FPS)
                elapsed = time() - loop_start
                if elapsed < 0.016:
                    sleep(0.016 - elapsed)
        
        finally:
            # Always release mouse button when done
            pyautogui.mouseUp()
            
            # Print stats
            total_time = time() - start
            fps = self.frame_count / total_time
            print(f"\n=== Session Complete ===")
            print(f"Duration: {total_time:.2f}s")
            print(f"Frames: {self.frame_count}")
            print(f"Average FPS: {fps:.1f}")
    
    def preview_detection(self):
        """Show a single frame with detection visualization"""
        # frame = self.capture_screen_fast()
        img = Image.open("dw.png")
        frame = np.array(img)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find bar and fish
        bar_rect = self.find_bar(gray)
        fish_pos = self.find_fish(gray)
        
        # Draw on frame
        vis_frame = frame.copy()
        
        if bar_rect:
            x, y, w, h = bar_rect
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(vis_frame, "BAR", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if fish_pos is not None:
            cv2.line(vis_frame, (fish_pos, 0), (fish_pos, vis_frame.shape[0]), (0, 0, 255), 2)
            cv2.putText(vis_frame, "FISH", (fish_pos + 5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Show info
        in_bar = self.is_fish_in_bar(bar_rect) if bar_rect else False
        color = (0, 255, 0) if in_bar else (0, 0, 255)
        status = "IN BAR" if in_bar else "OUT OF BAR"
        cv2.putText(vis_frame, status, (10, vis_frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.imshow("Detection Preview", vis_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def cleanup(self):
        """Clean up resources"""
        self.sct.close()
        pyautogui.mouseUp()


if __name__ == "__main__":
    macro = FischMacro()
    
    print("=== Fisch Macro ===")
    print("1. Preview detection")
    print("2. Run macro")
    
    choice = input("Choose (1 or 2): ").strip()
    
    if choice == "1":
        print("Capturing preview in 2 seconds...")
        sleep(2)
        macro.preview_detection()
    else:
        print("Starting macro in 3 seconds...")
        sleep(3)
        macro.run(duration=15, show_debug=True)
    
    macro.cleanup()
    print("Done!")