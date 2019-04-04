# Object Detection YOLO* V3 Demo, Async API Performance Showcase

This demo showcases Object Detection with YOLO* V3 and Async API.

Other demo objectives are:
* Video as input support via OpenCV*
* Visualization of the resulting bounding boxes and text labels (from the `.labels` file) or class number (if no file is provided)
* OpenCV provides resulting bounding boxes, labels, and other information.
You can copy and paste this code without pulling Inference Engine samples helpers into your application
* Demonstration of the Async API in action. For this, the demo features two modes toggled by the **Tab** key:
    -  Old-style "Sync" way, where the frame captured with OpenCV executes back-to-back with the Detection
    -  Truly "Async" way, where the detection is performed on a current frame, while OpenCV captures the next frame

### How It Works

On the start-up, the application reads command-line parameters and loads a network to the Inference
Engine. Upon getting a frame from the OpenCV VideoCapture, it performs inference and displays the results.

## Running

Running the application with the <code>-h</code> option yields the following usage message:
```sh
python object_detection_demo_yolov3_async -h
usage: object_detection_demo_yolov3.py [-h] -m MODEL -i INPUT
                                       [-l CPU_EXTENSION] [-pp PLUGIN_DIR]
                                       [-d DEVICE] [--labels LABELS]
                                       [-pt PROB_THRESHOLD]
                                       [-iout IOU_THRESHOLD] [-ni NUMBER_ITER]
                                       [-pc]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Path to an .xml file with a trained model.
  -i INPUT, --input INPUT
                        Path to a image/video file. (Specify 'cam' to work
                        with camera)
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        MKLDNN (CPU)-targeted custom layers.Absolute path to a
                        shared library with the kernels impl.
  -pp PLUGIN_DIR, --plugin_dir PLUGIN_DIR
                        Path to a plugin folder
  -d DEVICE, --device DEVICE
                        Specify the target device to infer on; CPU, GPU, FPGA
                        or MYRIAD is acceptable. Sample will look for a
                        suitable plugin for device specified (CPU by default)
  --labels LABELS       Labels mapping file
  -pt PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD
                        Probability threshold for detections filtering
  -iout IOU_THRESHOLD, --iou_threshold IOU_THRESHOLD
                        Intersection over union threshold for overlapping
                        detections filtering
  -ni NUMBER_ITER, --number_iter NUMBER_ITER
                        Number of inference iterations
  -pc, --perf_counts    Report performance counters
```

Running the application with the empty list of options yields the usage message given above and an error message.
You can use the following command to do inference on GPU with a pre-trained object detection model:
```sh
python object_detection_demo_yolov3.py -i <path_to_video>/inputVideo.mp4 -m <path_to_model>/yolo_v3.xml -d GPU

The cpu_extension.dll needs to be added to the arguments list for inference on CPU. Hence, the inference on CPU with a pre-trained object detection model would be the following: 
python object_detection_demo_yolov3.py <path_to_video>/inputVideo.mp4 -m <path_to_model>/yolo_v3.xml -d CPU --cpu_extension <path_to_cpu_extension>\cpu_extension.dll

Sample usage with **python_samples** for Movidius
python object_detection_demo_yolov3.py -i "C:\AI Devkit\Python_Samples\CodeSample 3_ObjectDetectionYolov3\model_files\objects.mp4" -m "C:\AI Devkit\Python_Samples\CodeSample 3_ObjectDetectionYolov3\model_files\FP16\frozen_darknet_yolov3_model.xml" -d MYRIAD

```
> **NOTE**: Public models should be first converted to the Inference Engine format (`*.xml` + `*.bin`) using the [Model Optimizer](./docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md) tool.

The only GUI knob is to use **Tab** to switch between the synchronized execution and the true Async mode.

### Demo Output

The demo uses OpenCV to display the resulting frame with detections (rendered as bounding boxes and labels, if provided).
In the default mode, the demo reports:
* **OpenCV time**: frame decoding + time to render the bounding boxes, labels, and to display the results.
* **Detection time**: inference time for the object detection network. It is reported in the Sync mode only.
* **Wallclock time**, which is combined application-level performance.