import numpy as np  # numpy - manipulate the packet data returned by depthai
import cv2  # opencv - display the video stream
import depthai  # access the camera and its data packets
print('Using depthai module from: ', depthai.__file__, depthai.__version__)

import consts.resource_paths  # load paths to depthai resources
from time import time

device = depthai.Device('', False)
# print(device.get_left_homography())
# print(device.get_left_intrinsic())
# print(device.get_right_homography())
# print(device.get_right_intrinsic())
# print(device.get_rotation())
# print(device.get_translation())
lh=device.get_left_homography()
rh=device.get_right_homography()
li=device.get_left_intrinsic()
ri=device.get_right_intrinsic()
ro=device.get_rotation()
t=device.get_translation()
calib_data=np.vstack((lh,rh,li,ri,ro,t))
print(calib_data)
#if not depthai.init_device(consts.resource_paths.device_cmd_fpath):
#   raise RuntimeError("Error initializing device. Try to reset it.")
# Create the pipeline using the 'previewout' stream, establishing the first connection to the device.
# pipeline = device.create_pipeline(config={
#     'streams': ['previewout', 'metaout', 'depth'],
#     'depth':
#     {
#         'calibration_file': consts.resource_paths.calib_fpath,
#         'left_mesh_file': consts.resource_paths.left_mesh_fpath,
#         'right_mesh_file': consts.resource_paths.right_mesh_fpath,
#         'padding_factor': 0.3,
#         'depth_limit_m': 10.0, # In meters, for filtering purpose during x,y,z calc
#         'confidence_threshold' : 0.5, #Depth is calculated for bounding boxes with confidence higher than this number
#         'median_kernel_size': 7,
#         'lr_check': False,
#         'warp_rectify':
#         {
#             'use_mesh' : False, # if False, will use homography
#             'mirror_frame': True, # if False, the disparity will be mirrored instead
#             'edge_fill_color': 0, # gray 0..255, or -1 to replicate pixel values
#         },
#     },
#     'ai': {
# #        "blob_file": "/home/pi/Downloads/3class_deeplabv3_256/deeplab_v3_plus_mnv3_decoder_256_3_class.blob",
#         "blob_file": "/home/pi/OAK-D-depthai-expts/03-depthai-3class/3class_256.blob",
#         #"blob_file_config": "/home/pi/Downloads/3class_deeplabv3_256/deeplab_v3_plus_mnv3_decoder_256_3_class.json",
#         "blob_file_config": "/home/pi/OAK-D-depthai-expts/03-depthai-3class/3class_256.json",
#         'shaves' : 14,
#         'cmx_slices' : 14,
#         'NN_engines' : 2,
#     },
#     'app':
#     {
#         'sync_video_meta_streams': True,
#     },
#     'camera':
#     {
#         'rgb':
#         {
#             # 3840x2160, 1920x1080
#             # only UHD/1080p/30 fps supported for now
#             'resolution_h': 1080,
#             'fps': 30,
#         },
#         'mono':
#         {
#             # 1280x720, 1280x800, 640x400 (binning enabled)
#             'resolution_h': 720,
#             'fps': 30,
#         },
#     },
# }
# )

