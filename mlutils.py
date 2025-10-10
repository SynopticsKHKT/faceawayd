import cv2


def display_yolo_track(frame, tracking_results):
    """
    Draws bounding boxes and track IDs onto the frame.

    Args:
        frame (np.ndarray): The current video frame.
        tracking_results (list or iterable): The results from the tracker,
                                            e.g., a list of (bbox, track_id, class_id, confidence).
    Returns:
        np.ndarray: The frame with drawn overlays.
    """

    # Define colors for drawing (optional)
    COLOR_BOX = (0, 255, 0)  # Green BGR
    COLOR_TEXT = (255, 255, 255)  # White BGR

    # Iterate through the tracked objects
    print(tracking_results)
    for result in tracking_results:
        # NOTE: The exact way to unpack results depends heavily on your specific YOLO library/API.
        # This is a common structure for demonstration:

        # 1. Unpack data
        # Example unpacking (adjust based on your library's output format)
        x1, y1, x2, y2, track_id, class_id, conf = result

        # Convert bounding box coordinates to integers
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        track_id = int(track_id)

        # 2. Draw Bounding Box (Rectangle)
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX, 2)  # Thickness=2

        # 3. Prepare Text Label
        label = f"ID: {track_id}"
        # Optional: Add Class Name or Confidence: f"ID: {track_id} | Class: {class_name}"

        # 4. Draw Text (Track ID)
        # Position the text label just above the bounding box
        text_pos = (x1, y1 - 10 if y1 > 20 else y1 + 20)
        cv2.putText(
            frame,
            label,
            text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,  # Font Scale
            COLOR_TEXT,
            2,  # Thickness
            cv2.LINE_AA,
        )

    return frame
