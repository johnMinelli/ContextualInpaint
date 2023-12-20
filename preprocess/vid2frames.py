import warnings

import cv2
import csv
import os

from tqdm import tqdm


def save_frames(video_path, output_path):
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
        frame_filename = f"frame_{frame_counter}.jpg"
        frame_filepath = os.path.join(output_path, frame_filename)
        # Resize by crop
        frame_resized = frame[:, -frame.shape[0]:, :]
        frame_resized = cv2.resize(frame_resized, (512, 512))
        cv2.imwrite(frame_filepath, frame_resized)

    # Release the video capture object
    cap.release()

def save_annotated_frames(video_path, timestamps_file, annotations_file, output_path):
    def get_frame(frame_id):
        timestamp = timestamps[frame_id]

        # Calculate the adjusted timestamp relative to the video's starting timestamp in milliseconds
        adjusted_timestamp = (timestamp - video_start_timestamp) * 1000

        # Set the video's position to the adjusted timestamp
        cap.set(cv2.CAP_PROP_POS_MSEC, adjusted_timestamp)

        # Read the frame at the adjusted timestamp
        ret, frame = cap.read()

        if not ret:
            warnings.warn(f"No frame at ts: {timestamp}, frame_id: {frame_id}, video_path: {video_path}")

        return frame, timestamp

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Failed to open the video file.")

    # Read timestamps from the file
    with open(timestamps_file, 'r') as file:
        timestamps = [float(line.strip()) for line in file]

    # Get the starting timestamp of the video
    video_start_timestamp = timestamps[0]

    file_timestamp_list = []
    with open(annotations_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row

        rows = [row for row in reader if row[1].split('/')[1] in video_path]
        for row in tqdm(rows):
            participant_id, file_id, annotation_id, frame_start, frame_end, activity, chunk_id = row
            file_id = file_id.split('_')[0].replace("/", "_")
            frame_start = int(frame_start)
            name = "_".join([file_id, "act"+annotation_id, "img"+chunk_id])
            frame, ts = get_frame(frame_start)
            if frame is None: continue

            file_timestamp_list.append((name, ts))

            # Create a folder with the activity name if it doesn't exist
            activity_folder = os.path.join(output_path, activity)
            if not os.path.exists(activity_folder):
                os.makedirs(activity_folder)

            # Save the frame as an image
            frame_filename = f"{name}.jpg"
            frame_filepath = os.path.join(activity_folder, frame_filename)
            frame_resized = cv2.resize(frame, (512, 512))
            cv2.imwrite(frame_filepath, frame_resized)

    # Save the file_timestamp_list as a text file
    timestamp_file = os.path.join(output_path, "file_timestamp_list.txt")
    with open(timestamp_file, 'a') as file:
        for file_id, timestamp in file_timestamp_list:
            file.write(f"{file_id},{timestamp}\n")

    # Release the video capture object
    cap.release()


if __name__ == '__main__':

    input_path = "../data/verizon"
    output_path = "../data/verizon_formatted2/target"

    annotations_file = f"{input_path}\\tasklevel.chunks_90.csv"

    for root, dirs, files in os.walk(input_path):
        files = [f.replace(".mp4", "") for f in files if f.endswith(".mp4")]

        # with open(file_path, 'w') as file:
        #     file.write(content)

        for run in files:
            print("Video:", os.path.join(root, run))
            video_path = os.path.join(root, f"{run}.mp4")
            timestamps_file =  os.path.join(root, f"{run}.timestamps")

            # save_annotated_frames(video_path, timestamps_file, annotations_file, output_path)
            os.makedirs(os.path.join(output_path, run), exist_ok=True)
            save_frames(video_path, os.path.join(output_path, run))