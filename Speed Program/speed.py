import argparse

import cv2

from inference.models.utils import get_roboflow_model

import supervision as sv


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Program Description"
    )

    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to source video file",
        type=str,
    )
    parser.add_argument(
        "--target_width",
        default=1280,
        help="Target width for the resized video (default: 1280)",
        type=int,
    )
    parser.add_argument(
        "--target_height",
        default=720,  # Maintain aspect ratio for default target width
        help="Target height for the resized video (default: calculated based on aspect ratio)",
        type=int,
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()

    # Get the original video capture object
    cap = cv2.VideoCapture(args.source_video_path)

    # Check if video capture is successful
    if not cap.isOpened():
        print("Error opening video stream or file")
        exit()

    # Get original video width and height
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate target height based on aspect ratio (if not provided)
    if args.target_height == -1:
        aspect_ratio = original_width / original_height
        args.target_height = int(args.target_width / aspect_ratio)

    # Define resizing function
    def resize_frame(frame):
        return cv2.resize(frame, (args.target_width, args.target_height))

    model = get_roboflow_model("yolov8x-640")

    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=4)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("No more frames to process")
            break

        # Resize the frame
        resized_frame = resize_frame(frame)

        result = model.infer(resized_frame)[0]
        detections = sv.Detections.from_inference(result)

        annotated_frame = resized_frame.copy()
        annotated_frame = bounding_box_annotator.annotate(annotated_frame, detections)

        cv2.imshow("annotated_frame", annotated_frame)

        # Quit if 'q' key is pressed
        if cv2.waitKey(1) == ord("q"):
            break

    # Release capture
    cap.release()
    cv2.destroyAllWindows()
