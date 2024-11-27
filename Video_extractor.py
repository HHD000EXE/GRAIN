import cv2
import os
import numpy as np
import pandas as pd

frame_time = []  # frame number that indicate relative time from motor starting
exca_seq_num = []  # sequence number of excavation actions
image_name = []  # extracted image names


depth_mode = 2  # 0 for RGB frame, 1 for \Delta depth frame, 2 for raw depth frame

if depth_mode == 1 or depth_mode == 2:
    # Open the video file
    folder_path = 'RGB-D(depth video)'
    # Open the csv file contains the first frame number that the motor moves
    csv_file_path = 'video starting frame depth.csv'

else:
    # Open the video file
    folder_path = 'RGB-D(color video)'
    # Open the csv file contains the first frame number that the motor moves
    csv_file_path = 'video starting frame color.csv'
# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)
# Get a list of file names in the folder
file_names = os.listdir(folder_path)

# Create a blank mask that is the same size as the image
mask = np.ones((400, 400, 3), dtype="uint8")
# Mask the vibration of motor
points = np.array([[60, 400], [60, 180], [220, 180], [220, 270], [160, 270], [160, 400]], dtype=np.int32)
# Reshape the points in a form required by polylines
points = points.reshape((-1, 1, 2))
# Draw the polygon on the mask with white color
cv2.polylines(mask, [points], isClosed=True, color=(0, 0, 0), thickness=2)
cv2.fillPoly(mask, [points], color=(0, 0, 0))

for file_name in file_names:
    # Search for the value in the specified column
    matching_row = df[df['video_names'] == file_name]
    try:
        start_frame = matching_row['starting_frame'].iloc[0]
        frame_distance = matching_row['frame_distance'].iloc[0]
    except:
        print(file_name)
        print("File name doesn't matched")
        break


    video_path = folder_path + '/' + file_name
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was successfully opened
    if not cap.isOpened():
        print("Error opening video file")
        exit()

    # Initialize variables
    frame_count = 0
    frame_record = []
    output_frames = []


    # Read and process frames
    while True:
        # Read the next frame
        ret, frame = cap.read()
        # if file_name[-8:-5] == "dom":
        #     frame_distance = 75  # subtract images with this distance
        # else:
        #     frame_distance = 45 # subtract images with this distance

        # Check if the frame was successfully read
        if not ret:
            break

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Define the coordinates for the top-left and bottom-right corners of the ROI
        if depth_mode == 1 or depth_mode == 2:
            x1, y1 = 130, 110  # Bottom-left corner
            x2, y2 = 430, 370  # Top-right corner
        else:
            x1, y1 = 110, 50  # Bottom-left corner
            x2, y2 = 510, 450  # Top-right corner

        # Crop the ROI from the image
        frame = frame[y1:y2, x1:x2]
        # frame = cv2.flip(frame, 0)  # flip the image vertically
        frame = cv2.resize(frame, (400, 400))  # resize the image
        frame_record.append(frame)

        # Process every 4th frame
        if (frame_count-start_frame) % frame_distance == 0 and frame_count >= start_frame:
            if depth_mode == 1:
                if frame_count == start_frame:
                    frame_sub = cv2.subtract(frame, frame)
                else:
                    frame_sub = cv2.subtract(frame, frame_record[frame_count - frame_distance])
                    # Apply the mask using bitwise_and
                    # frame_sub = np.multiply(frame_sub, mask)
                # frame_sub[frame_sub > 127] = 0
                output_frames.append(frame_sub)
            else:
                output_frames.append(frame)

        # Increment frame count
        frame_count += 1

        if frame_count > start_frame + 10 * frame_distance:
            break

    # Release the video file
    cap.release()

    # Save the extracted frames to files
    for i, frame in enumerate(output_frames[:-2]):
        if depth_mode == 0:
            cv2.imwrite(f"train_images(RGB)/{file_name[:-4]}_{i}.png", frame)
        if depth_mode == 1:
            cv2.imwrite(f"train_images(D)/{file_name[:-4]}_{i}.png", frame)
        if depth_mode == 2:
            cv2.imwrite(f"train_images(RawD)/{file_name[:-4]}_{i}.png", frame)
        image_name.append(f"{file_name[:-4]}_{i}.png")
        frame_time.append(i)
        # exca_seq_num.append(np.floor(i/22.5))


# # create csv file that contains extraced images information(image_name, frame_num, frame_time, exca_seq_num)
# data = {
#     'image_names': image_name,
#     'frame_time': frame_time,
#     'sequence_number_excavation_action': exca_seq_num
# }
# train_df = pd.DataFrame(data)
# csv_file_path = 'training data phase 1.csv'
# train_df.to_csv(csv_file_path, index=False)

# # create the csv that contains all video names
# data = {
#     'video_names': file_names,
# }
# df = pd.DataFrame(data)
# csv_file_path = 'Video_names.csv'
# df.to_csv(csv_file_path, index=False)