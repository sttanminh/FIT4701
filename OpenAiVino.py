import numpy as np
from openvino.inference_engine import IECore
import cv2


model_xml = "public/human-pose-estimation-3d-0001/FP32/human-pose-estimation-3d-0001.xml"
model_bin = "public/human-pose-estimation-3d-0001/FP32/human-pose-estimation-3d-0001.bin"
device = "CPU"

# Load model
ie = IECore()
net = ie.read_network(model=model_xml, weights=model_bin)
exec_net = ie.load_network(network=net, device_name=device)

# Open video file
# cap = cv2.VideoCapture("How To Pose People Who Are NOT MODELS  _Top 10 Pose Ideas From a Photographer_.mp4")
cap = cv2.VideoCapture(0)
count = 1
ret,frame = cap.read()
while ret:
    # Read a frame
    ret,frame = cap.read()
    # count +=1
    # print(count)
    # # Preprocess frame
    # input_blob = next(iter(net.input_info))
    # input_info = net.input_info[input_blob]
    # n, c, h, w = input_info.tensor_desc.dims
    # resized_frame = cv2.resize(frame, (w, h))
    # preprocessed_frame = resized_frame.transpose((2, 0, 1)).reshape((n, c, h, w))

    # # Run inference
    # output = next(iter(net.outputs))
    # results = exec_net.infer(inputs={input_blob: preprocessed_frame})

    # joints_data = results[output]
    # joints_data = np.squeeze(joints_data)
    # num_joints = 19# assuming that each joint has 3 dimension coordinates i.e. (x,y,z)

    # joints_array = joints_data.reshape(-1, num_joints, 3)
    cv2.imshow("testing",frame)
    print(frame)
    # for coords in joints_array[0]:
    #     x, y, z = coords
    #     print(f'Coordinate: ({x:.2f}, {y:.2f}, {z:.2f})')

        



    # Display result
    # Draw the 3D poses on the frame
    # Display the frame

cap.release()
cv2.destroyAllWindows()
