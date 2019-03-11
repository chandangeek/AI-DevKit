#!/usr/bin/env python
"""
 Copyright (c) 2018 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""


from __future__ import print_function
import sys
import os
from argparse import ArgumentParser

# Append OpenVINO python path
sys.path.append("C:\Intel\computer_vision_sdk_2018.5.456\python\python3.6")
import cv2
import numpy as np
import time
from openvino.inference_engine import IENetwork, IEPlugin

# Input arguments
def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", help="Path to an .xml file with a trained model.", required=True, type=str)
    parser.add_argument("-i", "--input", help="Path to a folder with images or path to an image files", required=True,
                        type=str)
    parser.add_argument("-l", "--cpu_extension",
                        help="MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernels "
                             "impl.", type=str, default=None)
    parser.add_argument("-pp", "--plugin_dir", help="Path to a plugin folder", type=str, default=None)
    parser.add_argument("-d", "--device",
                        help="Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device specified (CPU by default)", default="CPU",
                        type=str)
    parser.add_argument("-nt", "--number_top", help="Number of top results", default=10, type=int)
    parser.add_argument("-thresh", "--threshold", help="Confidence threshold for classification", default=0.3, type=float)
    return parser


def main():
	# Get the input arguments
    args = build_argparser().parse_args() 
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    model_threshold = args.threshold

    # Plugin initialization for specified device and load extensions library if specified
    plugin = IEPlugin(device=args.device, plugin_dirs=args.plugin_dir)
    if args.cpu_extension and 'CPU' in args.device:
        plugin.add_cpu_extension(args.cpu_extension)
    # Read IR
    net = IENetwork.from_ir(model=model_xml, weights=model_bin)
    assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
	#Loading IR to the plugin
    print("Loading IR to the plugin...")
    exec_net = plugin.load(network=net, num_requests=2)

	
    # Read and pre-process input image
    n, c, h, w = net.inputs[input_blob].shape
	
    del net
	
    if args.input == 'cam':
        input_stream = 0
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"
	# get the input video stream
    cap = cv2.VideoCapture(input_stream)

    cur_request_id = 0
    next_request_id = 1

    print("Starting inference in async mode...")
    print("To switch between sync and async modes press Tab button")
    print("To stop the sample execution press Esc button")
	# setting async mode
    is_async_mode = False
    render_time = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        initial_w = cap.get(3)
        initial_h = cap.get(4)
        in_frame = cv2.resize(frame, (w, h))
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((n, c, h, w))

        # Main sync point:
        # in the truly Async mode we start the NEXT infer request, while waiting for the CURRENT to complete
        # in the regular mode we start the CURRENT request and immediately wait for it's completion
        inf_start = time.time()
        if is_async_mode:
            exec_net.start_async(request_id=next_request_id, inputs={input_blob: in_frame})
        else:
            exec_net.start_async(request_id=cur_request_id, inputs={input_blob: in_frame})
        if exec_net.requests[cur_request_id].wait(-1) == 0:
            inf_end = time.time()
            det_time = inf_end - inf_start

            # Parse detection results of the current request
            res = exec_net.infer(inputs={input_blob: in_frame})
            #res = exec_net.requests[cur_request_id].outputs[out_blob]
            if (res != ""):
                top_ind = np.argsort(res[out_blob], axis=1)[0, -args.number_top:][::-1]
                outLabel = ""
                labelsEnabled = False
				# fetch the file name of labels
                labelFileName = os.path.splitext(model_xml)[0] + ".labels"
                labels = []
				# append the labels to a list named labels
                with open(labelFileName,"r") as inputFile:
                    for strLine in inputFile:
                        strLine.strip()
                        strLine.splitlines()
                        labels.append(strLine)
                    labelsEnabled = True
				#comparing with the threshold
                if (res[out_blob][0, top_ind[0][0][0]][0][0] > model_threshold) :				
                    cv2.putText(frame, str(labels[top_ind[0][0][0]]) , (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255,235,205), 2)
									
                #cv2.putText(frame, str(res[out_blob][0, top_ind[0][0][0]][0][0]) + str(labels[top_ind[0][0][0]]) , (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.6, (10, 10, 200), 1)
			
        render_start = time.time()
		# Show the output in a popup window, together with the label
        cv2.imshow("Detection Results", frame)
        render_end = time.time()
		#calculat the render time using starting and current time
        render_time = render_end - render_start

        key = cv2.waitKey(1)
        if key == 27:
            break
        if (9 == key):
            is_async_mode = not is_async_mode
            print("Switched to {} mode".format("async" if is_async_mode else "sync"))
        if is_async_mode:
            cur_request_id, next_request_id = next_request_id, cur_request_id

    cv2.destroyAllWindows()
    del exec_net
    del plugin


if __name__ == '__main__':
    sys.exit(main() or 0)

