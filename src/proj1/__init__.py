import sys
import numpy as np
import cv2 as cv2
import time
from yolov2tiny import YOLO_V2_TINY, postprocessing

def open_video_with_opencv(in_video_path, out_video_path):
    # Open an object of input video using cv2.VideoCapture.
    input_video = cv2.VideoCapture(in_video_path)    

    # Open an object of output video using cv2.VideoWriter.
    fps = input_video.get(5)
    width  = int(input_video.get(3))
    height = int(input_video.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    output_video = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    # Return the video objects and anything you want for further process.
    return input_video, output_video, (width, height)

def resize_input(im):
    imsz = cv2.resize(im, (416, 416))
    imsz = imsz / 255.
    imsz = imsz[:,:,::-1]

    # resize img
    imsz = np.array(imsz).reshape(1, 416, 416, 3)

    return np.asarray(imsz, dtype=np.float32)

def recover_input(im, dim):
    img = np.array(im).reshape(416, 416, 3)
    img = img[:,:,::-1]
    img = (img*255).astype(np.uint8)
    # img = cv2.resize(img, dim)
    return img

def video_object_detection(in_video_path, out_video_path, proc="cpu"):
    # Open video using open_video_with_opencv.
    input_video, output_video, dim = open_video_with_opencv(in_video_path, out_video_path)
    in_shape = (1, 416, 416, 3)
    pickle_path = "../../models/y2t_weights.pickle"
    total_elapsed_time = 0
    # scale_w = dim[0] / 416
    # scale_h = dim[1] / 416

    # Check if video is opened. Otherwise, exit.
    if not input_video.isOpened():
        print('video is not opened')
        sys.exit()
    # Create an instance of the YOLO_V2_TINY class. Pass the dimension of
    # the input, a path to weight file, and which device you will use as arguments.
    model = YOLO_V2_TINY(in_shape, pickle_path, proc)
    first = True

    # Start the main loop. For each frame of the video, the loop must do the followings:
    # 1. Do the inference.
    # 2. Run postprocessing using the inference result, accumulate them through the video writer object.
    #    The coordinates from postprocessing are calculated according to resized input; you must adjust
    #    them to fit into the original video.
    # 3. Measure the end-to-end time and the time spent only for inferencing.
    # 4. Save the intermediate values for the first layer.
    # Note that your input must be adjusted to fit into the algorithm,
    # including resizing the frame and changing the dimension.
    while True:
        ret, img = input_video.read()
        if not ret:
            break
        img = resize_input(img)

        start = time.time()
        output_tensors = model.inference(img)
        if first:
            first = False
            for i, tensor in enumerate(output_tensors):
                np.save("../../intermediate/layer_{}.npy".format(i+1), tensor)
        output_tensor = output_tensors[-1]
        end = time.time()
        elapsed_time = end-start
        total_elapsed_time += elapsed_time
        # print("Elapsed time to run inference: {}".format(elapsed_time))

        label_boxes = postprocessing(output_tensor)
        # print(len(label_boxes))
        img = recover_input(img, dim)
        for cl, (x1, y1), (x2, y2), (b, g, r) in  label_boxes:
            # cl, (x1, y1), (x2, y2), col = label_boxes
            # x1 = int(x1*scale_w)
            # y1 = int(y1*scale_h)
            # x2 = int(x2*scale_w)
            # y2 = int(y2*scale_h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (r, g, b), 3)
            cv2.putText(img, cl, (x1, y1), cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),1)
        img = cv2.resize(img, dim)
        output_video.write(img)


    # Check the inference peformance; end-to-end elapsed time and inferencing time.
    # Check how many frames are processed per second respectivly.
    # length = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = input_video.get(5)
    performance = fps / total_elapsed_time
    print("Total elapsed time for running inference: {}".format(total_elapsed_time))
    print("FPS processed per second: {}".format(performance))

    # Release the opened videos.
    input_video.release()
    output_video.release()

def main():
    if len(sys.argv) < 3:
        print ("Usage: python3 __init__.py [in_video.mp4] [out_video.mp4] ([cpu|gpu])")
        sys.exit()

    in_video_path = sys.argv[1] 
    out_video_path = sys.argv[2] 

    if len(sys.argv) == 4:
        proc = sys.argv[3]
    else:
        proc = "cpu"

    video_object_detection(in_video_path, out_video_path, proc)

if __name__ == "__main__":
    main()
