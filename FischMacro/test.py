import cv2, mss, pyautogui
import numpy as np
from PIL import Image
from time import time, sleep
from collections import deque

main_roi = {
            'left': 580,
            'top': 910,
            'width': 765,
            'height': 20
        }

img = Image.open("unn.png")

class Finder:
    def __init__(self):
          # Define the region of interest (ROI) where the fishing bar appears
        self.roi = main_roi
        # Initialize MSS for fast screen capture
        self.sct = mss.mss()
        
        # # If your array is already grayscale, this step might be skipped
        self.start_time = time()
        self.end_time = 0
        # color_array = np.asarray(img)

        self.contour_center = 0
        self.best_group = 0
        self.position_history = deque(maxlen=10)  # Keep last 10 positions
        self.time_history = deque(maxlen=10)

    def find_bar(self):
        ret, self.binary_array = cv2.threshold(self.gray_array, 30, 225, cv2.THRESH_TOZERO)
        # 127 is the threshold value, 255 is the max value, THRESH_BINARY creates binary output

        contours, hierarchy = cv2.findContours(self.binary_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_w = 0
        if len(contours) > 0:
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > best_w:
                    self.bar = cv2.boundingRect(contour)
                    best_w = w
                    self.contour_center = x + (w / 2)
                    # self.contour_center = w
            
            # print("Best:",best_w,self.contour_center)

            # Create a blank image to draw contours on
            self.contour_image = np.zeros_like(self.color_array)
            cv2.drawContours(self.contour_image, contours, -1, (255, 255, 255), 2) # Draw all contours in white, thickness 2
        else:
            print("no contours found")

        self.end_time = time()

        # self.preview(0)
        return int(self.contour_center)

    def find_fish(self):
        self.edges = cv2.Canny(self.gray_array, 50, 150)
        vertical_sums = np.sum(self.edges, axis=0)
        threshold = np.max(vertical_sums) * 0.55
        peaks = np.where(vertical_sums > threshold)[0]
        # print(peaks)

        if len(peaks) > 0:
            # Find groups of consecutive peaks (actual bars)
            groups = []
            current_group = [peaks[0]]
            
            for i in range(1, len(peaks)):
                if peaks[i] - peaks[i-1] <= 15:  # Within 5 pixels = same bar
                    current_group.append(peaks[i])
                else:
                    groups.append(current_group)
                    current_group = [peaks[i]]
            groups.append(current_group)
            
            # Find the most prominent group (tallest vertical bar)
            self.best_group = max(groups, key=lambda g: np.sum(vertical_sums[g]))
        else:
            return 0
        self.end_time = time()

        # self.preview(1)
        return int(np.mean(self.best_group))

    def calculate_vel_accel(self):
        if len(self.position_history) < 2:
            return 0,0
        
        recent_positions = list(self.position_history)[-5:]
        recent_times = list(self.time_history)[-5:]

        if len(recent_positions) < 2:
            return 0,0
        
        # Calculate delta time and delta pos
        velocities = []
        vel_times = []
        accelerations = []

        for i in range(1, len(recent_positions)):
            dt = recent_times[i] - recent_times[i-1]
            if dt > 0:
                dx = recent_positions[i] - recent_positions[i-1]
                velocity = dx/dt
                velocities.append(velocity)
                vel_times.append(recent_times[i])
        
        avg_vel = np.mean(velocities) if velocities else 0
        
        if len(velocities) >= 2:
            for i in range(1, len(velocities)):
                dt = vel_times[i] - vel_times[i-1]
                if dt > 0:
                    dv = velocities[i] - velocities[i-1]
                    accelerations.append(dv / dt)
            
            if accelerations:
                    return avg_vel, np.mean(accelerations)
            
        return avg_vel,0

    def predict_future_position(self,timeahead=0.1):
        """
        Predict where the bar will be in 'time_ahead' seconds
        Uses current position, velocity, and acceleration
        """
        if len(self.position_history) < 2:
            return None
        
        current_pos = self.position_history[-1]
        velocity, acceleration = self.calculate_vel_accel()
        # print(f"{velocity:6.1f} {acceleration:6.1f}")

        # Physics equation: position = pos + velocity*t + 0.5*acceleration*t^2
        predicted_pos = current_pos + velocity * timeahead + 0.5 * acceleration * (timeahead ** 2)
        predicted_pos = max(0, min(self.roi['width'], predicted_pos))
        
        return round(predicted_pos)

    def move_bar(self,duration=0.01):

        start = time()

        while time() - start < duration:
            self.color_array = self.capture_screen_fast()
            self.gray_array = cv2.cvtColor(self.color_array, cv2.COLOR_BGR2GRAY)

            bar = self.find_bar()
            fish = self.find_fish()

            if fish == 0:
                continue

            current_time = time()
            self.position_history.append(bar)
            self.time_history.append(current_time)

            cur_pos = self.position_history[-1]
            prev_pos = self.position_history[-2]
            dx = cur_pos - prev_pos

            nextPos = self.predict_future_position()
            velocity, acceleration = self.calculate_vel_accel()
            # print("Right") if velocity > 0 else print("Left")
            # print(f"Pos: {bar} Next Pos: {nextPos} Fish: {fish}")
            print(velocity)
            # self.dev_vis_vel(velocity)

            if fish > main_roi['width'] - self.bar[2]:
                pyautogui.mouseDown()
            elif fish < self.bar[2]:
                pyautogui.mouseUp()
            else:
                if not (self.bar[0] + main_roi['left']) + (self.bar[2]/2.67) <= fish <= (self.bar[2] + main_roi['left']) - (self.bar[2]/2.67):
                    if fish > bar:
                        # if velocity < 400:
                        pyautogui.mouseDown()
                        # if fish > (main_roi['left'] + main_roi['width'] - (self.bar[2]/2)):
                        #     pyautogui.mouseDown()
                            # print("###")
                            # pass
                    if fish < bar:
                        pyautogui.mouseUp()
                        if velocity < -200 and not (fish<self.bar[2]/2):
                            pyautogui.click()
                        # print("---")
                        # pass
                
                # pyautogui.moveTo(main_roi['left'] + fish, main_roi['top'])

            # print(f"bar: {bar} and fish: {fish}")

            # self.dev_vis_fb(fish,bar)
            
        
        # self.preview()
        print(self.position_history)

    def dev_vis_fb(self, fish, bar):
        pos = int((fish/main_roi['width'])*100)
        bah = int((bar/main_roi['width'])*100)
        prin = []
        for i in range(100):
            prin.append("-")
        prin[pos] = "|"
        prin[bah] = "#"
        prin = "".join(prin)
        print(prin)

    def dev_vis_vel(self,velocity):
        pos = int(abs(velocity)/100)
        prin = []
        for i in range(100):
            prin.append("-")
        prin[pos] = "|"
        prin = "".join(prin)
        print(prin)

    def preview(self):
        region_prev = self.capture_screen()
        # region_prev = np.array(img)
        region_prev = cv2.cvtColor(region_prev, cv2.COLOR_BGRA2BGR)

        cv2.rectangle(
                    region_prev,
                    (main_roi['left'], main_roi['top']),
                    (main_roi['left'] + main_roi['width'], 
                    main_roi['top'] + main_roi['height']),
                    (0, 0, 225),
                    1
                )
        
        # try:
        cv2.line(
            region_prev,
            (main_roi['left'] + int(self.contour_center), main_roi['top']),
            (main_roi['left'] + int(self.contour_center), main_roi['top'] + main_roi['height']),
            (0,225,0),
            4
        )
        cv2.line(
            region_prev,
            (main_roi['left'] + int(np.mean(self.best_group)), main_roi['top']),
            (main_roi['left'] + int(np.mean(self.best_group)),main_roi['top'] + main_roi['height']),
            (225,0,0),
            2
        )
        cv2.imshow("Detected Edges", self.edges)
        cv2.imshow("Detected Contours", self.contour_image)
        cv2.imshow("Binary Array", self.binary_array)
        # except Exception as e:
        #     print(e)

        cv2.imshow("Original Array", self.color_array)
        cv2.imshow("Region",region_prev)
        print("FPS:", int(1/(self.end_time - self.start_time)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
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
    
    def capture_screen(self):
        """Fast screen capture using MSS (much faster than pyautogui)"""
        monitor = {
            'left': self.roi['left']-100,
            'top': self.roi['top']-100,
            'width': self.roi['width']+200,
            'height': self.roi['height']+200
        }
        
        # Capture and convert to numpy array
        screenshot = self.sct.grab(monitor)
        frame = np.array(screenshot)
        
        # Convert BGRA to BGR
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def capture_region(self):
        bbox = (
            self.roi['left'],
            self.roi['top'],
            (self.roi['left'] + self.roi['width']),
            (self.roi['top'] + self.roi['height'])
        )

        try:
            crop = img.crop(bbox)

            region = np.asarray(crop)
        except Exception as e:
            print(e)
            return None

        return cv2.cvtColor(region, cv2.COLOR_BGRA2BGR)

if __name__ == "__main__":
    print("capturing in 3 seconds")
    sleep(1)
    print("2")
    sleep(1)
    print("1")
    sleep(1)
    finder = Finder()
    # finder.contour_method()
    finder.move_bar(duration=60)
    pyautogui.mouseUp()