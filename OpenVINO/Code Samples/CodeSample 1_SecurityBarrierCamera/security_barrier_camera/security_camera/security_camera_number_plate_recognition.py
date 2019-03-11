
from __future__ import print_function
import sys
sys.path.append("C:\Intel\computer_vision_sdk_2018.5.456\python\python3.6")
import os
from argparse import ArgumentParser
import cv2
import time
import logging as log
from openvino.inference_engine import IENetwork, IEPlugin
import numpy as np


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", help="Path to an .xml file with a trained model.", required=True, type=str)
    parser.add_argument("-m_va", "--model_va", help="Path to an .xml file with a vehicle attribute trained model.", required=True, type=str)
    parser.add_argument("-m_lpr", "--model_lpr", help="Path to an .xml file with a trained model to recognize license plate.", required=True, type=str)
    parser.add_argument("-i", "--input",
                        help="Path to video file or image. 'cam' for capturing video stream from camera", required=True,
                        type=str)
    ##If you are running in CPU, Please uncomment the below lines.
    parser.add_argument("-l", "--cpu_extension",
                        help="MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernels "
                             "impl.", type=str, default=None)
    parser.add_argument("-pp", "--plugin_dir", help="Path to a plugin folder", type=str, default=None)
    parser.add_argument("-d", "--device",
                        help="Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. Demo "
                             "will look for a suitable plugin for device specified (CPU by default)", default="CPU",
                        type=str)
    parser.add_argument("-d_va", "--device_va",
                        help="Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. Demo "
                             "will look for a suitable plugin for device specified (CPU by default)", default="CPU",
                        type=str)
    parser.add_argument("-d_lpr", "--device_lpr",
                        help="Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. Demo "
                             "will look for a suitable plugin for device specified (CPU by default)", default="CPU",
                        type=str)
    parser.add_argument("--labels", help="Labels mapping file", default=None, type=str)
    parser.add_argument("-pt", "--prob_threshold", help="Probability threshold for detections filtering",
                        default=0.5, type=float)

    return parser
list_lpr=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9","<Anhui>", "<Beijing>", "<Chongqing>", "<Fujian>","<Gansu>", "<Guangdong>", "<Guangxi>", "<Guizhou>","<Hainan>", "<Hebei>", "<Heilongjiang>", "<Henan>","<HongKong>", "<Hubei>", "<Hunan>", "<InnerMongolia>","<Jiangsu>", "<Jiangxi>", "<Jilin>", "<Liaoning>","<Macau>", "<Ningxia>", "<Qinghai>", "<Shaanxi>","<Shandong>", "<Shanghai>", "<Shanxi>", "<Sichuan>","<Tianjin>", "<Tibet>", "<Xinjiang>", "<Yunnan>","<Zhejiang>", "<police>","A", "B", "C", "D", "E", "F", "G", "H", "I", "J","K", "L", "M", "N", "O", "P", "Q", "R", "S", "T","U", "V", "W", "X", "Y", "Z"]

def maximum(a, n): 
    maxpos = a.index(max(a))
    return max(a),maxpos
def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    model_xml_xml = args.model
    model_xml_bin = os.path.splitext(model_xml_xml)[0] + ".bin"
    model_va_xml = args.model_va
    model_va_bin = os.path.splitext(model_va_xml)[0] + ".bin"
    model_lpr_xml = args.model_lpr
    model_lpr_bin = os.path.splitext(model_lpr_xml)[0] + ".bin"
    # Plugin initialization for specified device and load extensions library if specified
    log.info("Initializing plugin for {} device...".format(args.device))
    plugin = IEPlugin(device=args.device, plugin_dirs=args.plugin_dir)
    #If you are running in CPU, Please uncomment the below lines.
    if args.cpu_extension and 'CPU' in args.device:
        plugin.add_cpu_extension(args.cpu_extension)
    log.info("Reading IR...")
    net_recognition = IENetwork(model=model_xml_xml, weights=model_xml_bin)
    #If you are running in CPU, Please uncomment the below lines.
    if plugin.device == "CPU":
        supported_layers = plugin.get_supported_layers(net_recognition)
        not_supported_layers = [l for l in net_recognition.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(plugin.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in demo's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)
    assert len(net_recognition.inputs.keys()) == 1, "Demo supports only single input topologies"
    assert len(net_recognition.outputs) == 1, "Demo supports only single output topologies"
    input_blob = next(iter(net_recognition.inputs))
    out_blob = next(iter(net_recognition.outputs))
    log.info("Loading IR to the plugin...")
    exec_net_recognition = plugin.load(network=net_recognition, num_requests=2)
    # Read and pre-process input image
    n, c, h, w = net_recognition.inputs[input_blob].shape
    print("	n, c, h, w ",n, c, h, w )
    del net_recognition
    if args.input == 'cam':
        input_stream = 0
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"
    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.strip() for x in f]
    else:
        labels_map = None
    cap = cv2.VideoCapture(input_stream)
    cur_request_id = 0
    next_request_id = 1

    log.info("Starting inference in async mode...")
    log.info("To switch between sync and async modes press Tab button")
    log.info("To stop the demo execution press Esc button")
    is_async_mode = False
    render_time = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        next_frame=frame
        initial_w = frame.shape[0]
        initial_h = frame.shape[1]
        # Main sync point:
        # in the truly Async mode we start the NEXT infer request, while waiting for the CURRENT to complete
        # in the regular mode we start the CURRENT request and immediately wait for it's completion
        inf_start = time.time()
        if is_async_mode:
            in_frame = cv2.resize(frame, (w, h))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))
            exec_net_recognition.start_async(request_id=next_request_id, inputs={input_blob: in_frame})
        else:
            in_frame = cv2.resize(frame, (w, h))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))
            exec_net_recognition.start_async(request_id=cur_request_id, inputs={input_blob: in_frame})
        if exec_net_recognition.requests[cur_request_id].wait(-1) == 0:
            inf_end = time.time()
            det_time = inf_end - inf_start
            # Parse detection results of the current request
            res = exec_net_recognition.requests[cur_request_id].outputs[out_blob]

            for obj in res[0][0]:
                # Draw only objects when probability more than specified threshold
                if obj[2] > args.prob_threshold :
                    if obj[2]==1:
                        xmin = int(obj[3] * initial_h)
                        ymin = int(obj[4] * initial_w)
                        xmax = int(obj[5] * initial_h)
                        ymax = int(obj[6] * initial_w)
                        class_id_vehicle = int(obj[1])
                    # Draw box and label\class_id
                        color = (min(class_id_vehicle * 12.5, 10), min(class_id_vehicle * 7, 10), min(class_id_vehicle * 5, 10))
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,255), 2)
                        crop_vehicle = frame[ymin:ymax,xmin:xmax]
                        crop_img = crop_vehicle
                        net_va_recognition = IENetwork(model=model_va_xml, weights=model_va_bin)
                        input_blob_va = next(iter(net_va_recognition.inputs))
                        out_blob_va = next(iter(net_va_recognition.outputs))
                        log.info("Loading IR to the plugin...")
                        exec_net_va_recog = plugin.load(network=net_va_recognition, num_requests=2)
                        n_va, c_va, h_va, w_va = net_va_recognition.inputs[input_blob_va].shape
                        if is_async_mode==False:
                            if crop_img.any():
                                in_frame_va = cv2.resize(crop_img, (w_va, h_va))
                            else:
                                in_frame_va = cv2.resize(frame, (w_va, h_va))
                            in_frame_va = in_frame_va.transpose((2, 0, 1))  # Change data layout from HWC to CHW
                            in_frame_va = in_frame_va.reshape((n_va, c_va, h_va, w_va))
                            res_va = exec_net_va_recog.infer(inputs={input_blob_va: in_frame_va})
                        
					# Processing output blob
                        if exec_net_recognition.requests[cur_request_id].wait(-1) == 0:
                            lis_res_color=[]
                            log.info("Processing output blob")
                            res_colord=res_va["color"]
                            for l_i in np.nditer(res_colord):
                                val1=int(l_i*100)
                                lis_res_color.append(val1)
                            labels_map_list_color = ["white", "gray", "yellow", "red", "green", "blue", "black"]
                            max_color_value,max_color_index=maximum(lis_res_color, len(lis_res_color))					
                            class_id_va=max_color_index
                            det_label_va = labels_map_list_color[class_id_va] 
                            lis_res_type=[]
                            res_type=res_va["type"]
                            for type_i in np.nditer(res_type):
                                val1_type_i=int(type_i*100)
                                lis_res_type.append(val1_type_i)
                            labels_map_list_type = ["car", "van", "truck", "bus"]
                            max_type_value,max_type_index=maximum(lis_res_type, len(lis_res_type))	
                            class_id_type=max_type_index
                            det_label_type = labels_map_list_type[class_id_type] 
                        cv2.putText(frame, det_label_va + ' ' + det_label_type + ' ', (xmin, ymin - 7),cv2.FONT_HERSHEY_COMPLEX, 0.6, (10, 10, 200), 2)
						
                        del exec_net_va_recog
						
						
						
                    if obj[2]!=1:
                        xmin = int(obj[3] * initial_h)
                        ymin = int(obj[4] * initial_w)
                        xmax = int(obj[5] * initial_h)
                        ymax = int(obj[6] * initial_w)
                        class_id = int(obj[1])
                    # Draw box and label\class_id
                        color = (min(class_id * 12.5, 10), min(class_id * 7, 10), min(class_id * 5, 10))
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,255), 2)
                        crop_plate = frame[ymin:ymax,xmin:xmax]
                        crop_img_lpr = crop_plate
                        net_lpr_recognition = IENetwork(model=model_lpr_xml, weights=model_lpr_bin)
                        input_blob_lpr = list(net_lpr_recognition.inputs.keys())[0]
                        seq_id = list(net_lpr_recognition.inputs.keys())[1]
                        out_blob_lpr = next(iter(net_lpr_recognition.outputs))
                        log.info("Loading IR to the plugin...")
                        exec_net_lpr_recog = plugin.load(network=net_lpr_recognition, num_requests=2)
                        n_lpr, c_lpr, h_lpr, w_lpr = net_lpr_recognition.inputs[input_blob_lpr].shape
                        in_frame_seq_id = np.ones(88, dtype=np.int32)
                        in_frame_seq_id = in_frame_seq_id.reshape((88,1))
                        in_frame_seq_id[0] = 0
                        if is_async_mode==False:
                            if crop_img_lpr.any():
                                in_frame_lpr = cv2.resize(crop_img_lpr, (w_lpr, h_lpr))
                            else:
                                in_frame_lpr = cv2.resize(frame, (w_lpr, h_lpr))
                            in_frame_lpr = in_frame_lpr.transpose((2, 0, 1))  # Change data layout from HWC to CHW
                            in_frame_lpr = in_frame_lpr.reshape((n_lpr, c_lpr, h_lpr, w_lpr))
                            res_lpr = exec_net_lpr_recog.infer(inputs={input_blob_lpr: in_frame_lpr, seq_id: in_frame_seq_id})							
                            lis_res_lpr=[]							
                            res_lpr_decode=res_lpr['decode']					
                            for type_i in np.nditer(res_lpr_decode):
                                val1_lpr_i=int(type_i)
                                lis_res_lpr.append(val1_lpr_i)
                        class_id_lpr=lis_res_lpr[0]
                        det_label_lpr = list_lpr[class_id_lpr] 
                        list_plate=[]						
                        for item_list in lis_res_lpr:
                            if item_list==-1:
                                break
                            else: 								
                                list_plate.append(list_lpr[item_list])
                        print("Number_plate",list_plate)	
                        if len(	list_plate)==6:
                            	list_plate.remove(list_plate[0])
                            	str_lpr= ''.join(list_plate)								
                            	cv2.putText(frame, det_label_lpr + ' '+str_lpr, (xmin, ymin - 7),cv2.FONT_HERSHEY_COMPLEX, 0.8, (10, 10, 200), 2)
                        del exec_net_lpr_recog    
            
                    
            # Draw performance stats
            inf_time_message = "Inference time: N\A for async mode" if is_async_mode else \
			"Inference time: {:.3f} ms".format(det_time * 1000)
            render_time_message = "OpenCV rendering time: {:.3f} ms".format(render_time * 1000)
            async_mode_message = "Async mode is on. Processing request {}".format(cur_request_id) if is_async_mode else \
			"Async mode is off. Processing request {}".format(cur_request_id)

            cv2.putText(frame, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            cv2.putText(frame, render_time_message, (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
            cv2.putText(frame, async_mode_message, (10, int(initial_h - 20)), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (10, 10, 200), 1)
        
        render_start = time.time()
        cv2.imshow("Detection Results", frame)
        render_end = time.time()
        render_time = render_end - render_start
        
        if is_async_mode:
            cur_request_id, next_request_id = next_request_id, cur_request_id
            frame = next_frame
            
        key = cv2.waitKey(500)
        if key == 27:
            break      
        if (9 == key):
            is_async_mode = not is_async_mode
            log.info("Switched to {} mode".format("async" if is_async_mode else "sync"))
        			
    cv2.destroyAllWindows()
    del exec_net_recognition    
    del plugin


if __name__ == '__main__':
    sys.exit(main() or 0)
