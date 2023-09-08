import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 0.5
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red
#Descargar el modelo de : https://developers.google.com/mediapipe/solutions/vision/object_detector#efficientdet-lite0_model_recommended

def visualize(frame, detection_result) -> np.ndarray:
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(frame, start_point, end_point, TEXT_COLOR, 3)

        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(frame, result_text, text_location, cv2.FONT_ITALIC,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return frame

def draw_colored_rectangle(frame, color):
    height, width, _ = frame.shape
    top_left = (width // 4, height // 4)
    bottom_right = (3 * width // 4, 3 * height // 4)

    if color == 'red':
        rectangle_color = (0, 0, 255)  # Red color in BGR
    elif color == 'blue':
        rectangle_color = (255, 0, 0)  # Blue color in BGR

    cv2.rectangle(frame, top_left, bottom_right, rectangle_color, thickness=2)

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    color_sequence = ['red', 'blue']  # Red, blue, red, blue, ...

    interval = 0.025  # Interval of 0.025 seconds

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        color = color_sequence[int(time.time() / interval) % 2]  # Alternating between red and blue
        
        draw_colored_rectangle(frame, color)
            
        cv2.imshow('Video with Rectangles', frame)
        
        if cv2.waitKey(int(1000 * interval)) & 0xFF == ord('q'):  # Display frame for interval, exit on 'q' key
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    video_path = 'cats_dogs.mp4'  # Reemplaza con la ruta de tu video
    main(video_path)
