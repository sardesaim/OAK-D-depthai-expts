import numpy as np  # numpy - manipulate the packet data returned by depthai
import cv2  # opencv - display the video stream
import depthai  # access the camera and its data packets
print('Using depthai module from: ', depthai.__file__, depthai.__version__)

import consts.resource_paths  # load paths to depthai resources
from time import time

device = depthai.Device('', False)
device.send_disparity_confidence_threshold(255)
#if not depthai.init_device(consts.resource_paths.device_cmd_fpath):
#   raise RuntimeError("Error initializing device. Try to reset it.")
# Create the pipeline using the 'previewout' stream, establishing the first connection to the device.
pipeline = device.create_pipeline(config={
    'streams': ['previewout', 'metaout', 'depth'],
    'depth':
    {
        'calibration_file': consts.resource_paths.calib_fpath,
        'left_mesh_file': consts.resource_paths.left_mesh_fpath,
        'right_mesh_file': consts.resource_paths.right_mesh_fpath,
        'padding_factor': 0.3,
        'depth_limit_m': 10.0, # In meters, for filtering purpose during x,y,z calc
        'confidence_threshold' : 0.5, #Depth is calculated for bounding boxes with confidence higher than this number
        'median_kernel_size': 7,
        'lr_check': False,
        'warp_rectify':
        {
            'use_mesh' : False, # if False, will use homography
            'mirror_frame': True, # if False, the disparity will be mirrored instead
            'edge_fill_color': 0, # gray 0..255, or -1 to replicate pixel values
        },
    },
    'ai': {
#        "blob_file": "/home/pi/Downloads/3class_deeplabv3_256/deeplab_v3_plus_mnv3_decoder_256_3_class.blob",
        "blob_file": "/home/pi/OAK-D-depthai-expts/03-depthai-3class/3class_256.blob",
        #"blob_file_config": "/home/pi/Downloads/3class_deeplabv3_256/deeplab_v3_plus_mnv3_decoder_256_3_class.json",
        "blob_file_config": "/home/pi/OAK-D-depthai-expts/03-depthai-3class/3class_256.json",
        'shaves' : 14,
        'cmx_slices' : 14,
        'NN_engines' : 2,
    },
    'app':
    {
        'sync_video_meta_streams': True,
    },
    'camera':
    {
        'rgb':
        {
            # 3840x2160, 1920x1080
            # only UHD/1080p/30 fps supported for now
            'resolution_h': 1080,
            'fps': 30,
        },
        'mono':
        {
            # 1280x720, 1280x800, 640x400 (binning enabled)
            'resolution_h': 720,
            'fps': 30,
        },
    },
}
)
if pipeline is None:
    raise RuntimeError('Pipeline creation failed!')
# The model deeplab_v3_plus_mnv3_decoder_256_sh4cmx4.blob returns a 256x256 of int32
# Each int32 is either 0 (background) or 1 (person)
class_colors = [[0,0,0], [255,0,0], [0,255,0], [0,0,255]]
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
        print(len(raw))
        print(len(outputs[0]))
        output = raw[:len(outputs[0])]
        output = np.reshape(output, (256,256)) 
        output_colors = np.take(class_colors, output, axis=0)
        # print(output_colors.shape)    
    for packet in data_packets:
        # By default, DepthAI adds other streams (notably 'meta_2dh'). Only process `previewout`.
        if packet.stream_name == 'previewout':
            frame_count_cam += 1
            data = packet.getData()
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
                # cv2.imshow("Output", output_colors)
            cv2.imshow('previewout', frame)
        elif packet.stream_name.startswith('depth') or packet.stream_name == 'disparity_color':
                    frame = packet.getData()

                    if len(frame.shape) == 2:
                        if frame.dtype == np.uint8: # grayscale
                            cv2.putText(frame, packet.stream_name, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
                        else: # uint16
                            frame = (65535 // frame).astype(np.uint8)
                            #colorize depth map, comment out code below to obtain grayscale
                            frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
                            # frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
                            cv2.putText(frame, packet.stream_name, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255)
                            
                    else: # bgr
                        cv2.putText(frame, packet.stream_name, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
                    cv2.imshow('depth', frame)
                    
    key = cv2.waitKey(1)
    if  key == ord('q'):
        break
# The pipeline object should be deleted after exiting the loop. Otherwise device will continue working.
# This is required if you are going to add code after exiting the loop.
del pipeline
