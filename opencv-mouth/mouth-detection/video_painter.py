import cv2 as cv
import numpy as np
from detect_face_parts import detect_face_parts
from imutils import face_utils
import dlib



def play_video(filepath):
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")    
    detector = dlib.get_frontal_face_detector()
    
    # Video
    video = cv.VideoCapture(vid_dir + vid_japanese)
    fps = video.get(cv.CAP_PROP_FPS)
    faster = 0.65 # less is faster, higher is slower
    sleep_ms = int(np.round((1/fps)*1000*faster))
    
    # Audio
    total_frames = 0
    while video.isOpened():
        ret, frame = video.read()

        
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Draw face visualizations
        if total_frames % 60 == 0:
            faces = detector(gray, 1)
        else:
            faces = []
        
        if not len(faces) == 0:
            print(len(faces))
            shape = predictor(gray, faces[0])
            shape = face_utils.shape_to_np(shape)
        
            output = face_utils.visualize_facial_landmarks(gray, shape)
        else:
            output = frame
        
        # gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        cv.imshow('frame', output)
        
        # cv.imshow('frame', frame)
        if cv.waitKey(sleep_ms) == ord('q'):
            break
        
        total_frames += 1
        
    video.release()
    cv.destroyAllWindows()
    
    
    
if __name__ == "__main__":
    vid_dir = "../test-data/test-videos/"
    vid_japanese = "zerotwo_japanese.mkv"
    vid_english = "zerotwo_english.mkv"
    play_video(vid_dir + vid_japanese)