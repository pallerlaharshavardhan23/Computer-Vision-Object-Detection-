import cv2
import numpy as np
import time

# Function to calculate real-world size in cm
def calculate_object_size(pixel_width, pixel_height, known_width_cm, known_height_cm, known_distance_cm, focal_length):
    real_width_cm = (pixel_width * known_width_cm * known_distance_cm) / focal_length
    real_height_cm = (pixel_height * known_height_cm * known_distance_cm) / focal_length
    real_length_cm = (real_width_cm**2 + real_height_cm**2)**0.5  # Diagonal length
    return real_width_cm, real_height_cm, real_length_cm

# Load YOLO model
def load_yolo_model():
    try:
        net = cv2.dnn.readNet(
            r'C:\Users\palle\OneDrive\Desktop\Object det project\yolov3.weights',
            r'C:\Users\palle\OneDrive\Desktop\Object det project\yolov3.cfg'
        )
        with open(r'C:\Users\palle\OneDrive\Desktop\Object det project\coco.names', 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        return net, classes, output_layers
    except Exception as e:
        
        print(f"Error loading YOLO model: {e}")
        exit()

# Process detections
def process_detections(net, img, output_layers, classes, known_width_cm, known_height_cm, known_distance_cm, focal_length):
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids, confidences, boxes, object_sizes = [], [], [], []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

                object_width, object_height, object_length = calculate_object_size(
                    w, h, known_width_cm, known_height_cm, known_distance_cm, focal_length
                )
                object_sizes.append((object_width, object_height, object_length))

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return boxes, class_ids, confidences, object_sizes, indexes

# Draw annotations
def draw_annotations(img, boxes, class_ids, confidences, object_sizes, indexes, classes, fps=None, mode="image"):
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            object_width, object_height, object_length = object_sizes[i]

            # Generate the label and size information
            label = f"{classes[class_ids[i]]}: {confidences[i]*100:.1f}%"
            size_info = [
                f"Width: {object_width:.1f} cm",
                f"Height: {object_height:.1f} cm",
                f"Length: {object_length:.1f} cm"
            ]

            # Draw bounding box
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Add the label above the box
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Display measurements below the box for image detection
            if mode == "image":
                text_y = y + h + 15
                for text in size_info:
                    cv2.putText(img, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    text_y += 15

            # Display measurements on the left side for video detection
            elif mode == "video":
                text_y = y
                for text in size_info:
                    text_y += 20  # Line spacing
                    cv2.putText(img, text, (x - 150, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display FPS for video detection
    if mode == "video" and fps is not None:
        cv2.putText(img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return img

# Detect objects in an image
def detect_image(net, classes, output_layers, image_path, known_width_cm, known_height_cm, known_distance_cm, focal_length):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found.")
        return

    height, width, _ = img.shape
    boxes, class_ids, confidences, object_sizes, indexes = process_detections(
        net, img, output_layers, classes, known_width_cm, known_height_cm, known_distance_cm, focal_length
    )

    annotated_img = draw_annotations(img, boxes, class_ids, confidences, object_sizes, indexes, classes, mode="image")
    cv2.namedWindow("Image Detection", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Image Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Image Detection", annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Detect objects in real-time video
def detect_video(net, classes, output_layers, known_width_cm, known_height_cm, known_distance_cm, focal_length):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Webcam not detected.")
        return

    zoom_factor = 1.0
    prev_time = 0

    cv2.namedWindow("Real-time Object Detection", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Real-time Object Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, img = cap.read()
        if not ret:
            print("Error capturing frame.")
            break

        height, width, _ = img.shape

        # Apply zoom
        center_x, center_y = width // 2, height // 2
        radius_x, radius_y = int(width / (2 * zoom_factor)), int(height / (2 * zoom_factor))
        min_x, max_x = center_x - radius_x, center_x + radius_x
        min_y, max_y = center_y - radius_y, center_y + radius_y
        img = img[min_y:max_y, min_x:max_x]
        img = cv2.resize(img, (width, height))

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        boxes, class_ids, confidences, object_sizes, indexes = process_detections(
            net, img, output_layers, classes, known_width_cm, known_height_cm, known_distance_cm, focal_length
        )

        img = draw_annotations(img, boxes, class_ids, confidences, object_sizes, indexes, classes, fps, mode="video")

        cv2.imshow("Real-time Object Detection", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('+'):  # Zoom in
            zoom_factor = min(zoom_factor + 0.1, 3.0)
        elif key == ord('-'):  # Zoom out
            zoom_factor = max(zoom_factor - 0.1, 1.0)
            # Minimize window
            cv2.setWindowProperty("Real-time Object Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    cap.release()
    cv2.destroyAllWindows()

# Main function
def main():
    net, classes, output_layers = load_yolo_model()

    print("Choose an option:")
    print("1. Detect objects in an image")
    print("2. Detect objects in real-time video")
    choice = input("Enter your choice (1/2): ").strip()

    known_width_cm = 20
    known_height_cm = 10
    known_distance_cm = 100
    focal_length = 500

    if choice == '1':
        image_path = input("Enter the image path: ").strip()
        detect_image(net, classes, output_layers, image_path, known_width_cm, known_height_cm, known_distance_cm, focal_length)
    elif choice == '2':
        detect_video(net, classes, output_layers, known_width_cm, known_height_cm, known_distance_cm, focal_length)
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
