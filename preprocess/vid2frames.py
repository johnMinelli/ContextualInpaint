import warnings

import cv2
import os

from tqdm import tqdm


def save_frames(video_id, video_path, output_path):
    def get_frame(frame_id):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()

        if not ret:
            warnings.warn(f"No frame at frame_id: {frame_id}, video_path: {video_path}")

        return frame

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Failed to open the video file.")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_save = range(0, frame_count, 2)

    for frame_id, frame_counter in tqdm(zip(frames_to_save, range(0, len(frames_to_save)))):
        frame = get_frame(frame_id)
        if frame is None:
            continue

        # Save the frame as an image
        frame_filename = f"{video_id}_{frame_counter}.jpg"
        frame_filepath = os.path.join(output_path, frame_filename)
        # Resize by crop
        frame_resized = frame[:, -frame.shape[0]:,:]
        frame_resized = cv2.resize(frame_resized, (512, 512))
        cv2.imwrite(frame_filepath, frame_resized)

    # Release the video capture object
    cap.release()


if __name__ == '__main__':

    input_path = "../data/verizon"
    output_path = "../data/verizon_formatted2/target"

    for root, dirs, files in os.walk(input_path):
        files = [f.replace(".mp4", "") for f in files if f.endswith(".mp4")]

        for video_id, run in enumerate(files):
            print("Video:", os.path.join(root, run))
            video_path = os.path.join(root, f"{run}.mp4")
            timestamps_file =  os.path.join(root, f"{run}.timestamps")

            save_frames(video_id, video_path, output_path)