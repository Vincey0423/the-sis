import argparse

import cv2 

from inference.models.utils import get_roboflow_model

import supervision as sv

def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Program Description"
    )

    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to source video file",
        type=str   
    
    )
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_argument()

    model = get_roboflow_model("yolov8x-640")

    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=4)
    
    frame_generator = sv.get_video_frames_generator(args.source_video_path)

    for frame in frame_generator:
        result = model.infer(frame)[0]
        detections = sv.Detections.from_inference(result)

        annotated_frame = frame.copy()
        annotated_frame = bounding_box_annotator.annotate(annotated_frame, detections)
        
        cv2.imshow("annotated_frame", annotated_frame)
        if cv2.waitKey(1) == ord("q"):
            break   

    cv2.destroyAllWindows()