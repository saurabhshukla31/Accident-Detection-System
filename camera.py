# import cv2
# from detection import AccidentDetectionModel
# import numpy as np
# import os
#
# model = AccidentDetectionModel("model.json", 'model_weights.keras')
# font = cv2.FONT_HERSHEY_SIMPLEX
#
# def startapplication():
#     video = cv2.VideoCapture(0)  # video = cv2.VideoCapture('cars.mp4') for videos
#
#     if not video.isOpened():
#         print("Error: Could not open camera.")
#         return
#
#     while True:
#         ret, frame = video.read()
#         if not ret or frame is None:
#             print("Error: Could not read frame. Exiting.")
#             break
#
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         roi = cv2.resize(gray_frame, (250, 250))
#
#         pred, prob = model.predict_accident(roi[np.newaxis, :, :])
#         if pred == "Accident":
#             prob = round(prob[0][0] * 100, 2)
#
#             # Uncomment to beep when alert
#             if prob > 99:
#                 os.system("say beep")
#
#             cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
#             cv2.putText(frame, f"{pred} {prob}%", (20, 30), font, 1, (255, 255, 0), 2)
#
#         if cv2.waitKey(33) & 0xFF == ord('q'):
#             break
#
#         cv2.imshow('Camera Feed', frame)
#
#     video.release()
#     cv2.destroyAllWindows()
#
# if __name__ == '__main__':
#     startapplication()
# import cv2
# from detection import AccidentDetectionModel
# import numpy as np
# import os
#
# # Load your model
# model = AccidentDetectionModel("model.json", "model_weights.keras")
# font = cv2.FONT_HERSHEY_SIMPLEX
#
#
# def startapplication(video_path):
#     # Open the video file
#     video = cv2.VideoCapture(video_path)
#
#     if not video.isOpened():
#         print(f"Error: Could not open video file {video_path}.")
#         return
#
#     while True:
#         ret, frame = video.read()
#         if not ret or frame is None:
#             print("End of video. Exiting.")
#             break
#
#         # Preprocessing frame for prediction
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         roi = cv2.resize(gray_frame, (250, 250))
#
#         # Predict accidents using the model
#         pred, prob = model.predict_accident(roi[np.newaxis, :, :])
#         if pred == "Accident":
#             prob = round(prob[0][0] * 100, 2)
#
#             # Uncomment this line for sound alert on a local machine
#             # if prob > 99:
#             #     os.system("say beep")  # Not supported in Colab
#
#             # Display prediction on the video
#             cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
#             cv2.putText(frame, f"{pred} {prob}%", (20, 30), font, 1, (255, 255, 0), 2)
#
#         # Press 'q' to exit
#         if cv2.waitKey(33) & 0xFF == ord('q'):
#             break
#
#         # Display the video frame (not visible in Colab; works locally)
#         cv2.imshow("Accident Detection", frame)
#
#     video.release()
#     cv2.destroyAllWindows()
#
#
# if __name__ == "__main__":
#     # Provide the relative path to the video file
#     video_file_path = "./video.mp4"
#     startapplication(video_file_path)
import cv2
from detection import AccidentDetectionModel
import numpy as np
import os

# Load your model
model = AccidentDetectionModel("model.json", "model_weights.keras")
font = cv2.FONT_HERSHEY_SIMPLEX


def startapplication(video_path, output_path="output_video.avi"):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print(f"Error: Could not open video file {video_path}.")
        return

    # Get video properties
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = video.read()
        if not ret or frame is None:
            print("End of video. Exiting.")
            break

        # Preprocessing frame for prediction
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(gray_frame, (250, 250))

        # Predict accidents using the model
        pred, prob = model.predict_accident(roi[np.newaxis, :, :])
        if pred == "Accident":
            prob = round(prob[0][0] * 100, 2)

            # Add prediction overlay to frame
            cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
            cv2.putText(frame, f"{pred} {prob}%", (20, 30), font, 1, (255, 255, 0), 2)

        # Write the frame to the output video
        out.write(frame)

    video.release()
    out.release()
    print(f"Output saved to {output_path}")
