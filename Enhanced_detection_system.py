import cv2
import numpy as np

# Global variables
roi_x, roi_y, roi_width, roi_height = 100, 100, 200, 200
roi_shape = 'rectangle'  # Default ROI shape
frame_count = 0
frame_rate = 0
sum_roi = 0
avg_sum_roi = 0
max_pixel_value = 0
accumulated_frames = 100  # Number of frames to accumulate
resizing_roi = False
moving_roi = False

# Initialize moving_roi
moving_roi = False

# Resize the ROI
def resize_roi(image, x_factor, y_factor):
    global roi_width, roi_height
    new_width = int(roi_width * x_factor)
    new_height = int(roi_height * y_factor)
    roi_width = max(1, new_width)
    roi_height = max(1, new_height)

# Move the ROI
def move_roi(image, x_offset, y_offset):
    global roi_x, roi_y
    roi_x = max(0, roi_x + x_offset)
    roi_y = max(0, roi_y + y_offset)
    
# Display information in a black window
def display_info_window():
    global avg_sum_roi, sum_frame, sum_roi, frame_count, accumulated_frames, max_pixel_value, frame_rate
    info_frame = np.zeros((300, 600, 3), dtype=np.uint8)
    cap = cv2.VideoCapture(0)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    text = f'Accumulated Frames: {accumulated_frames}'
    cv2.putText(info_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    text = f'Sum of Pixels in Current Frame: {sum_roi}'
    cv2.putText(info_frame, text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    text = f'Avg. Sum of Pixels in ROI: {avg_sum_roi:.2f}'
    cv2.putText(info_frame, text, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    text = f'Max Pixel Value in ROI: {max_pixel_value}'
    cv2.putText(info_frame, text, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    text = f'Camera frame rate: {frame_rate:.2f} fps'
    cv2.putText(info_frame, text, (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Readings", info_frame)


# Main function
def main():
    global roi_x, roi_y, roi_width, roi_height, sum_roi, avg_sum_roi, moving_roi, sum_frame, frame_count, max_pixel_value, roi_shape, ellipse_major_axis, ellipse_minor_axis

    cap = cv2.VideoCapture(0)

    cv2.namedWindow("Webcam")
    cv2.namedWindow("Saturation Status", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Saturation Status", 400, 100)

    while True:
        ret, frame = cap.read()

        sum_frame = frame.copy()

        roi = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_roi)

        # Calculate sum of pixel values
        sum_roi += np.sum(roi)
        max_pixel_value = np.max(roi)
        frame_count += 1

        # Drawing rectangle, circle, or ellipse around ROI based on shape and resizing/moving status
        if resizing_roi:
            if roi_shape == 'rectangle':
                cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 0, 255), 2)
            elif roi_shape == 'circle':
                cv2.circle(frame, (roi_x + roi_width // 2, roi_y + roi_height // 2), roi_width // 2, (0, 0, 255), 2)
            elif roi_shape == 'ellipse':
                cv2.ellipse(frame, (roi_x + roi_width // 2, roi_y + roi_height // 2), (ellipse_major_axis, ellipse_minor_axis), 0, 0, 360, (0, 0, 255), 2)
        elif moving_roi:
            if roi_shape == 'rectangle':
                cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 255, 0), 2)
            elif roi_shape == 'circle':
                cv2.circle(frame, (roi_x + roi_width // 2, roi_y + roi_height // 2), roi_width // 2, (0, 255, 0), 2)
            elif roi_shape == 'ellipse':
                cv2.ellipse(frame, (roi_x + roi_width // 2, roi_y + roi_height // 2), (ellipse_major_axis, ellipse_minor_axis), 0, 0, 360, (0, 255, 0), 2)
        else:
            if roi_shape == 'rectangle':
                cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 255, 0), 2)
            elif roi_shape == 'circle':
                cv2.circle(frame, (roi_x + roi_width // 2, roi_y + roi_height // 2), roi_width // 2, (0, 255, 0), 2)
            elif roi_shape == 'ellipse':
                cv2.ellipse(frame, (roi_x + roi_width // 2, roi_y + roi_height // 2), (ellipse_major_axis, ellipse_minor_axis), 0, 0, 360, (0, 255, 0), 2)

        cv2.imshow("Webcam", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            avg_sum_roi = 0
            frame_count = 0
            sum_roi = 0
        elif key == ord('+'):
            resize_roi(frame, 1.1, 1.1)
        elif key == ord('-'):
            resize_roi(frame, 0.9, 0.9)
        elif key == ord('1'):
            roi_shape = 'rectangle'
        elif key == ord('2'):
            roi_shape = 'circle'
        elif key == ord('3'):
            roi_shape = 'ellipse'
        # Adjust major and minor axes of the ellipse
        elif key == ord('4') and roi_shape == 'ellipse':
            ellipse_major_axis = max(ellipse_major_axis - 10, 10)
        elif key == ord('5') and roi_shape == 'ellipse':
            ellipse_major_axis += 10
        elif key == ord('6') and roi_shape == 'ellipse':
            ellipse_minor_axis = max(ellipse_minor_axis - 10, 10)
        elif key == ord('7') and roi_shape == 'ellipse':
            ellipse_minor_axis += 10

        elif key == ord('m'):
            moving_roi = not moving_roi
        elif key == ord('a') and moving_roi:
            move_roi(frame, -10, 0)
        elif key == ord('d') and moving_roi:
            move_roi(frame, 10, 0)
        elif key == ord('w') and moving_roi:
            move_roi(frame, 0, -10)
        elif key == ord('s') and moving_roi:
            move_roi(frame, 0, 10)
        elif key == ord('c'):
            if frame_count >= accumulated_frames:
                avg_sum_roi = sum_roi / (frame_count * accumulated_frames)
                display_info_window()
            frame_count = 0
            sum_roi = 0
            max_pixel_value = 0

        if sum_roi > 0:
            # Display the sum of pixel values
            cv2.putText(frame, f'Sum of pixel values: {sum_roi}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Webcam", frame)

            # Check for saturation
            saturation_threshold = 200  # You can adjust this threshold
            if max_pixel_value > saturation_threshold:
                status_message = "Saturated!!!"
                status_color = (0, 0, 255)  # Red color
            else:
                status_message = "Below saturation level"
                status_color = (0, 255, 0)  # Green color
            
            # Create a frame for saturation status and display it
            status_frame = np.zeros((100, 400, 3), dtype=np.uint8)
            cv2.putText(status_frame, status_message, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            cv2.imshow("Saturation Status", status_frame)

        # Display histogram of intensity profile
        hist = cv2.calcHist([s], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, None).flatten()
        hist_image = np.zeros((256, 256, 3), dtype=np.uint8)
        max_hist_value = np.max(hist)
        for i in range(256):
            cv2.line(hist_image, (i, 256), (i, 256 - int(hist[i] * 256 / max_hist_value)), (255, 0, 0), 1)
        cv2.imshow("Intensity Profile Histogram", hist_image)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
camera_index = 0  # Use 0 for the default camera
get_camera_frame_rate(camera_index)
