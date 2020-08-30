import numpy as np  # numpy - manipulate the packet data returned by depthai
import cv2  # opencv - display the video stream
import depthai  # access the camera and its data packets
import consts.resource_paths  # load paths to depthai resources
from time import time
if not depthai.init_device(consts.resource_paths.device_cmd_fpath):
    raise RuntimeError("Error initializing device. Try to reset it.")
# Create the pipeline using the 'previewout' stream, establishing the first connection to the device.
pipeline = depthai.create_pipeline(config={
    'streams': ['previewout', 'depth_raw'],
    'ai': {
        "blob_file": "/home/sardesaim/Downloads/3class_deeplabv3_256/deeplab_v3_plus_mnv3_decoder_256_3_class.blob",
        "blob_file_config": "/home/sardesaim/Downloads/3class_deeplabv3_256/deeplab_v3_plus_mnv3_decoder_256_3_class.json",
        #'shaves' : 14,
        #'cmx_slices' : 14,
        #'NN_engines' : 2,
    },
    'app':
    {
        'sync_video_meta_streams': True,
    },
})
if pipeline is None:
    raise RuntimeError('Pipeline creation failed!')
# The model deeplab_v3_plus_mnv3_decoder_256_sh4cmx4.blob returns a 256x256 of int32
# Each int32 is either 0 (background) or 1 (person)
class_colors = [[0,0,255],  [0,255,0], [255,0,0]]
class_colors = np.asarray(class_colors, dtype=np.uint8)
entries_prev = []
output_colors = None
t_start = time()
frame_count_nn = 0
frame_count_cam = 0
while True:
    # Retrieve data packets from the device.
    # A data packet contains the video frame data.
    nnet_packets, data_packets = pipeline.get_available_nnet_and_data_packets()
       
    for _, nnet_packet in enumerate(nnet_packets):
        frame_count_nn += 1
        raw = nnet_packet.get_tensor("out")        
        raw.dtype = np.int32
        outputs = nnet_packet.entries()[0]
        output = raw[:len(outputs[0])]
        output = np.reshape(output, (256,256)) 
        output_colors = np.take(class_colors, output, axis=0)
            
    for packet in data_packets:
        # By default, DepthAI adds other streams (notably 'meta_2dh'). Only process `previewout`.
        data=packet.getData()
        if packet.stream_name == 'previewout':
            frame_count_cam += 1
            # data = packet.getData()
            data0 = data[0, :, :]
            data1 = data[1, :, :]
            data2 = data[2, :, :]
            frame = cv2.merge([data0, data1, data2])
            img_h = frame.shape[0]
            img_w = frame.shape[1]
            t_curr = time()
            if t_curr > t_start + 1:
                t_diff = t_curr - t_start
                print(f"Cam fps = {frame_count_cam/t_diff:0.1f}")
                print(f"NN fps = {frame_count_nn/t_diff:0.1f}")
                t_start = t_curr
                frame_count_cam = 0
                frame_count_nn = 0
            if output_colors is not None:
                mixed = cv2.addWeighted(frame,1, output_colors,0.2,0)
                cv2.imshow("Mixed", mixed)
            cv2.imshow('previewout', frame)
        elif packet.stream_name.startswith('depth') or packet.stream_name == 'disparity_color':
            frame=data
            if len(frame.shape) == 2:
                if frame.dtype == np.uint8: # grayscale
                    cv2.putText(frame, packet.stream_name, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
                    cv2.putText(frame, "fps: " + str(frame_count_cam), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
                else: # uint16
                    frame = (65535 // frame).astype(np.uint8)
                    cv2.imwrite('frame.png', frame)
                    #colorize depth map, comment out code below to obtain grayscale
                    frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
                    # frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
                    cv2.putText(frame, packet.stream_name, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255)
                    cv2.putText(frame, "fps: " + str(frame_count_cam), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255)
            else: # bgr
                cv2.putText(frame, packet.stream_name, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
                cv2.putText(frame, "fps: " + str(frame_count_cam), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255)
    key = cv2.waitKey(1)
    if  key == ord('q'):
        break
# The pipeline object should be deleted after exiting the loop. Otherwise device will continue working.
# This is required if you are going to add code after exiting the loop.
del pipeline