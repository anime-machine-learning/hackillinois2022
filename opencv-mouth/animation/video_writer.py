import cv2 as cv
import numpy as np
from scipy import rand
from prediction import predict

RATE = 1
PREDICTION_FRAMES = 1

BASE_IMAGE = cv.imread("images/base.png", cv.IMREAD_UNCHANGED)
# BASE_IMAGE_DIM = BASE_IMAGE.shape # (19.. x 1078 x 3)
BASE_IMAGE_DIM = (1918, 1076)

# Define animation faces
AGAPE = cv.imread("images/agape.png", cv.IMREAD_UNCHANGED)
CLOSED = cv.imread("images/closed.png", cv.IMREAD_UNCHANGED)
OPEN = cv.imread("images/open.png", cv.IMREAD_UNCHANGED)
ANIMATION_FACES = {
    -1: BASE_IMAGE,
    0: CLOSED,
    1: AGAPE,
    2: OPEN
}
def write(output_video, face, base_image) -> None:
    """Writes the animation face on the base image.
    """
    output_video.write(cv.addWeighted(base_image, 1, face, 0.5, 0))
    
    
    
def create_animation_from_video(filepath : str, predictor=None) -> None:
    
    # Input Video
    if filepath == "capture":
        input_video = cv.VideoCapture(0)
    else:
        input_video = cv.VideoCapture(filepath)
    fps = input_video.get(cv.CAP_PROP_FPS)
    sleep_ms = int(np.round((1/fps)*1000*RATE))
    
    # Ouput Video
    # print(BASE_IMAGE_DIM[:2])
    output_video = cv.VideoWriter('output.avi', cv.VideoWriter_fourcc('M','J','P','G'), fps, BASE_IMAGE_DIM)
    # output_video = cv.VideoWriter('output.avi', 0xc10100be, int(fps / PREDICTION_FRAMES), BASE_IMAGE_DIM[:2])
    # output_video = cv.VideoWriter("output4.mp4", cv.VideoWriter_fourcc(*'MP4V'), int(fps / PREDICTION_FRAMES), BASE_IMAGE_DIM[:2])
    # print(fps)
    # print(int(fps / PREDICTION_FRAMES))
    # output_video = cv.VideoWriter('output.avi', -1, 20, BASE_IMAGE_DIM[:2])
    
    total_frames = 0
    prev_class = 0
    while input_video.isOpened():
        # if total_frames == 10*fps:
        #     break
        ret, frame = input_video.read()
        
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        if total_frames % PREDICTION_FRAMES == 0:
            # print(predict)
            classification = predict(frame)
            if classification is None:
                # classification = -1
                classification = prev_class
            prev_class = classification
            # classification = total_frames % 3
            
            # Draw face visualizations on base image
            # print(ANIMATION_FACES[classification][750,710])
            # write(output_video, ANIMATION_FACES[classification], BASE_IMAGE)
            
            # print("class",classification)
            # cv.imshow('frame', ANIMATION_FACES[classification])
            # if cv.waitKey(sleep_ms) == ord('q'):
            #     break
            
            # output_video.write(cv.addWeighted(BASE_IMAGE, 1, ANIMATION_FACES[classification], 0.5, 0))
            # print(ANIMATION_FACES[-1].shape)
            newimg = ANIMATION_FACES[classification]
            img = BASE_IMAGE
            mask = newimg[:,:,3]==255
            img[mask] =newimg[mask]
            output_video.write(img[:,:,:3])

        # Show video
        # cv.imshow('frame', frame)
        # if cv.waitKey(sleep_ms) == ord('q'):
        #     break
        
        total_frames += 1
        
    input_video.release()
    output_video.release()
    cv.destroyAllWindows()
    

def play_video(filepath):
    video = cv.VideoCapture(filepath)
    fps = video.get(cv.CAP_PROP_FPS)
    sleep_ms = int(np.round((1/fps)*1000*RATE))
    
    while video.isOpened():
        ret, frame = video.read()
        cv.imshow('frame', frame)
        
        if cv.waitKey(sleep_ms) == ord('q'):
            break
        
    video.release()
    cv.destroyAllWindows()
    
if __name__ == "__main__":
    create_animation_from_video("video.mp4")
    # create_animation_from_video("capture")
    play_video("output.avi")
    # cv.imshow('frame', ANIMATION_FACES[0])
    # if cv.waitKey(0) == ord('q'):
    #     pass
    