import cv2
import numpy as np
from matplotlib import pyplot as plt

key = cv2.waitKey(1)
webcam = cv2.VideoCapture(0)
while True:
    try:
        check, frame = webcam.read()
        print(check)  # prints true as long as the webcam is running
        print(frame)  # prints matrix values of each framecd
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        if key == ord('s'):
            cv2.imwrite(filename='saved_img.jpg', img=frame)
            webcam.release()
            img_new = cv2.imread('saved_img.jpg', cv2.IMREAD_GRAYSCALE)
            img_new = cv2.imshow("Captured Image", img_new)
            cv2.waitKey(1650)
            cv2.destroyAllWindows()
            print("Processing image...")
            img_ = cv2.imread('saved_img.jpg', cv2.IMREAD_ANYCOLOR)
            print("Converting RGB image to grayscale...")
            gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
            print("Converted RGB image to grayscale...")
            print("Resizing image to 640x480 scale...")
            img_ = cv2.resize(gray, (640,480))
            print("Resized...")
            img_resized = cv2.imwrite(filename='saved_img-final.jpg', img=img_)
            print("Image saved!")
            imgpath = r'D:\Prajwal Chettri\Research\Waveguides\Mode profiling automation\saved_img-final.jpg'
            img = cv2.imread(imgpath, 0)

            plt.subplot(1, 2, 1)
            plt.imshow(img, cmap='gray')
            plt.title('Image')
            plt.xticks([])
            plt.yticks([])

            plt.subplot(1, 2, 2)
            hist, bin = np.histogram(img.ravel(), 256, [0, 255])
            plt.xlim([0, 255])
            plt.plot(hist)
            plt.title('Intensity Distribution')

            plt.show()

            break
        elif key == ord('q'):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break

    except(KeyboardInterrupt):
        print("Turning off camera.")
        webcam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break
