# Project Write-Up

## Explaining Custom Layers

The process behind converting custom layers involves - introducing the custom layer for both Model Optimizer and Inference Engine. If both are not done, both steps of using OpenVINO toolkit will give out errors. For example, Model Optimizer will not proceed when tasked to convert a model with unrecognized layer into IR unless provided with Custom Layer Attribute Extraction and Custom Layer Implementation files. For Inference Engine, when fed with IR from a model with a custom layer must be provided with Custom Layer Implementation file that corresponds to the custom layer that the IR has. This files can be generated with the Extension Generation Tool. We have different terminologies like Extension, Custom Layer, Kernel when using this tool but mostly differentiates at what step of the pipeline the addition of custom operation is involved. Extension is used when the custom operation is added in MO, and Custom Layer and Kernel when added to IE, Custom Layer is used for CPU inference, and Kernel for GPU inference.

#### Custom Layer Files

Declare the Custom Layer and its attributes and parameters at MO

- Custom Layer Attribute Extraction(.py) - Identify Custom Layer Operation and extract parameters for each instance of the custom layer
- Custom Layer Implementation(.py) - specifies the attributes supported by the Custom Layer and compute the output shape for each instance of the Custom Layer

Implement the Custom Layer Logic at IE

- Custom Layer Implementation(.dll/.so) - contains optimized operations to execute the Custom Layer

#### Custom Layers Pre-work:

 1. See if when running mo.py we have the following output:

    ```[ ERROR ] List of operartions that cannot be converted to Inference Engine IR```

    This means we have custom operations that needs to be implemented at MO and IE.

	2. Use the following script to generate the required files for Custom Layer Operation. (Install requirements with package manager of your choice.) `/opt/intel/openvino/deployment_tools/tools/extension_generator/extgen.py`

    Passing -h as an argument gives the following

    ```
    usage: You can use any combination of the following arguments:

    Arguments to configure extension generation in the interactive mode:

    optional arguments:
      -h, --help            show this help message and exit
      --mo-caffe-ext        generate a Model Optimizer Caffe* extractor
      --mo-mxnet-ext        generate a Model Optimizer MXNet* extractor
      --mo-tf-ext           generate a Model Optimizer TensorFlow* extractor
      --mo-op               generate a Model Optimizer operation
      --ie-cpu-ext          generate an Inference Engine CPU extension
      --ie-gpu-ext          generate an Inference Engine GPU extension
      --output_dir OUTPUT_DIR
                            set an output directory. If not specified, the current
                            directory is used by default.
    ```

    Example to generate Custom Layer files for a TF CPU configurations we can:

    `python /opt/intel/openvino/deployment_tools/tools/extension_generator/extgen.py --mo-op --mo-tf-ext --ie-cpu-ext` --output_dir $OUTPUT_DIR

    This will execute a template, prompting for Layer Name, Parameters, Shape info etc.

    Files are found at the `$OUTPUT_DIR`

#### #### Custom Layer at MO:

​	Just provide the `user_mo_extensions` dir found inside the `$OUTPUT_DIR` of the Custom Layer Pre-work to `--extensions` argument when creating IR. This contains the files discussed above required for Custom Layer MO operation.

​	DOD: Error message found in pre-work not found anymore

#### Custom Layer at IE:

​	This step contains the process implementation of the exact operation of custom layer inside a .cpp file found inisde `user_ie_extensions` in the `$OUTPUT_DIR`. `CMakeLists.txt` also needs some changes as we will use this to compile .cpp file earlier to create the .dll/.so extension we will feed to the IE.

​	After acquiring the the .so/.dll extension we can just give it to, for example, our main.py with the appropriate argument(`-l`/ `--cpu_extension` in this project) to have the custom layer as part of the operation of IE.

### Rationale

Some of the potential reasons for handling custom layers is that the layers unsupported are crucial to inference, and that approaching the problem with other solution such as passing the handling to, say, the origin of the model(TF, Caffe, etc.) is not possible.

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations were doing inference to the models before conversion, with the original framework

The size of the model pre- and post-conversion was larger pre-conversion and smaller post conversion. This is more notable on when we play with the precision values, going lower up to FP16.

| Size     | ssd_inception_v2_coco_2018_01_28 | ssd_mobilenet_v2_coco_2018_03_29 | faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28 |
| -------- | -------------------------------- | -------------------------------- | ------------------------------------------------------ |
| Original | 97 MB                            | 66 MB                            | 235 MB                                                 |
| FP32     | 95 MB                            | 64 MB                            | 229 MB                                                 |
| FP16     | 48 MB                            | 32 MB                            | 115 MB                                                 |

The inference time of the model pre- and post-conversion was was the most pre converted models where faster in seconds(mostly 5 seconds) when compared to post converted models

| Inference Time | ssd_inception_v2_coco_2018_01_28 | ssd_mobilenet_v2_coco_2018_03_29 | faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28 |
| -------------- | -------------------------------- | -------------------------------- | ------------------------------------------------------ |
| Original       | 140 ms                           | 55 ms                            | 155 ms                                                 |
| FP32           | 150 ms                           | 60 ms                            | 160 ms                                                 |

## Assess Model Use Cases

Some of the potential use cases of the people counter app are:

Data Analytics, we can count people that goes to our retail, say, on specific times, dates, and events. We can use people count to say what piques customer interest.

 We can also use People Counter App on Pedestrian Crossings, we can efficiently automate the traffic lights depending on the count of the pedestrian waiting to be able to cross over the other side of the street, adding to already used metric of time interval of stop and go signals.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model. The potential effects of each of these are as follows:

Since this things will probably differ between training and inference, it will be much like the model is operating on an unseen data. It will then depend on how great the model was trained prior its use. Lighting has great effect since it gives the model while training the advantage of accurately/precisely locating people, but training the model under poor lighting condition and being able to locate people well will help inference on the same conditions. As for the camera specifications such as focal length, this concers with the image size and the maginification of the objects, the shorter the focal length the wider the viewing angle hence larger image and lower the magnification, objects appearing smaller, the longer the focal length, the opposite it has on the image size and magnification. This well the inference if different specs are used between inference
and training. Say in training you have a long focal length, objects appearing larger, your bounding boxes will have to depend on that, and on inference when shorter focal length is used, objects appearing smaller, the bounding boxes should have been smaller then at the training.

## Model Research

In investigating potential people counter models, I tried each of the following three models:

Command used for all model conversion(variable value found under the specific model):

```python /opt/intel/openvino_2020.2.117/deployment_tools/model_optimizer/mo_tf.py --tensorflow_object_detection_api_pipeline_config $pip_config --tensorflow_use_custom_operations_config $cust_config --reverse_input_channels --input_model $input_model```

- Model 1: `ssd_inception_v2_coco_2018_01_28`
  - http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gzO
  - I converted the model to an Intermediate Representation with the following arguments...
    - ```input_model=ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.pb```
      ```pip_config=ssd_inception_v2_coco_2018_03_29/pipeline.config```
    ```cust_config=/opt/intel/openvino_2020.2.117/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json```
  - The model was insufficient for the app because even with improvements, the detection is janky, with the statistics shown on the app depending on the mqtt publish, the only solution is to make sure that model does not get detection wrong.
  - I tried to improve the model for the app by trying out different precisions and handling the wrong detections in `main.py` trying to suppress detections when time intervals for the detection are near impossible values such as values nearing 0 seconds as this presents impossibly quick movements near the edge of the camera.

- Model 2: `ssd_mobilenet_v2_coco_2018_03_29`
  - http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz

  - I converted the model to an Intermediate Representation with the following arguments...

  - ```input_model=ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb```

      ```pip_config=ssd_inception_v2_coco_2018_03_29/pipeline.config```
      ```cust_config=/opt/intel/openvino_2020.2.117/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json```

  - The model was insufficient for the app because even with improvements, the detection is janky, with the statistics shown on the app depending on the mqtt publish, the only solution is to make sure that model does not get detection wrong.

  - I tried to improve the model for the app by trying out different precisions and handling the wrong detections in `main.py` trying to suppress detections when time intervals for the detection are near impossible values such as values nearing 0 seconds as this presents impossibly quick movements near the edge of the camera.

- Model 3: `faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28`
  - http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz
  - I converted the model to an Intermediate Representation with the following arguments...
    - ```input_model=faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/frozen_inference_graph.pb```
      ```pip_config=faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/pipeline.config```
      ```cust_config=/opt/intel/openvino_2020.2.117/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json```
  - The model was insufficient for the app because even with improvements, the detection is janky, with the statistics shown on the app depending on the mqtt publish, the only solution is to make sure that model does not get detection wrong.
  - I tried to improve the model for the app by trying out different precisions and handling the wrong detections in `main.py` trying to suppress detections when time intervals for the detection are near impossible values such as values nearing 0 seconds as this presents impossibly quick movements near the edge of the camera

## Model Used

In the end I just looked up for models in the OpenVino pre-trained libraries and found out about `person-detection-retail-0013`. This has performed well with the resource video, eliminating the issue I have faced with the converted tensorflow models. The use of the model didn't completely eliminate the wrong detection especially on the edge of the video playback, when person is leaving the frame, but compared to the converted tensorflow models, this was easy to handle with the rule I have shared, which is to count it if the entry and exit time interval is greater than the near impossible time of ~0 seconds. Again, since the counter depends on the publishing with MQTT, with the said rule, publishing is done correctly.
