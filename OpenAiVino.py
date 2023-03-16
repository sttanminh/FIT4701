import numpy
from openvino.inference_engine import IECore
import cv2


model_xml = "public\human-pose-estimation-3d-0001\FP16\human-pose-estimation-3d-0001.xml"
model_bin = "public\human-pose-estimation-3d-0001\FP16\human-pose-estimation-3d-0001.bin"
device = "CPU"

# Load model
ie = IECore()
net = ie.read_network(model=model_xml, weights=model_bin)
exec_net = ie.load_network(network=net, device_name=device)

# Open video file
cap = cv2.VideoCapture("path/to/video.mp4")

while True:
    # Read a frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess frame
    input_blob = next(iter(net.inputs))
    n, c, h, w = net.inputs[input_blob].shape
    resized_frame = cv2.resize(frame, (w, h))
    preprocessed_frame = resized_frame.transpose((2, 0, 1)).reshape((n, c, h, w))

    # Run inference
    output_blob = next(iter(net.outputs))
    results = exec_net.infer(inputs={input_blob: preprocessed_frame})
    feet_coords_3d = results[output_blob]

    # Display result
    # Draw the 3D poses on the frame
    # Display the frame

cap.release()
cv2.destroyAllWindows()
