# Project Write-Up

## Explaining Custom Layers

The process behind converting custom layers involves - introducing the custom layer for both Model Optimizer and Inference Engine. If both are not done, both steps of using OpenVINO toolkit will give out errors. For example, Model Optimizer will not proceed when tasked to convert a model with unrecognized layer into IR unless provided with Custom Layer Attribute Extraction and Custom Layer Implementation files. For Inference Engine, when fed with IR from a model with a custom layer must be provided with Custom Layer Implementation file that corresponds to the custom layer that the IR has. This files can be generated with the Extension Generation Tool. We have different terminologies like Extension, Custom Layer, Kernel when using this tool but mostly differentiates at what step of the pipeline the addition of custom operation is involved. Extension is used when the custom operation is added in MO, and Custom Layer and Kernel when added to IE, Custom Layer is used for CPU inference, and Kernel for GPU inference.

Some of the potential reasons for handling custom layers is that the layers unsupported are crucial to inference, and that approaching the problem with other solution such as passing the handling to, say, the origin of the model(TF, Caffe, etc.) is not possible.

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations were doing inference to the models before conversion, with the original framework

The difference between model accuracy pre- and post-conversion was that most of the pre converted models where more accurate when compared to post-converted models.

The size of the model pre- and post-conversion was bigger pre-conversion and smaller post conversion. This is more notable on when we play with the precision values, going lower up to FP16.

The inference time of the model pre- and post-conversion was was the most pre converted models where faster in seconds(mostly 5 seconds) when compared to post converted models

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
  - [Tensorflow]
  - I converted the model to an Intermediate Representation with the following arguments...
    - ```input_model=ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.pb```
      ```pip_config=ssd_inception_v2_coco_2018_03_29/pipeline.config```
    ```cust_config=/opt/intel/openvino_2020.2.117/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json```
  - The model was insufficient for the app because even with improvements, the detection is janky, with the statistics shown on the app depending on the mqtt publish, the only solution is to make sure that model does not get detection wrong.
  - I tried to improve the model for the app by trying out different precisions and handling the wrong detections in `main.py` trying to suppress detections when time intervals for the detection are near impossible values such as values nearing 0 seconds as this presents impossibly quick movements near the edge of the camera.

- Model 2: `ssd_mobilenet_v2_coco_2018_03_29`
  - [Tensorflow]

  - I converted the model to an Intermediate Representation with the following arguments...

  - ```input_model=ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb```

      ```pip_config=ssd_inception_v2_coco_2018_03_29/pipeline.config```
      ```cust_config=/opt/intel/openvino_2020.2.117/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json```

  - The model was insufficient for the app because even with improvements, the detection is janky, with the statistics shown on the app depending on the mqtt publish, the only solution is to make sure that model does not get detection wrong.

  - I tried to improve the model for the app by trying out different precisions and handling the wrong detections in `main.py` trying to suppress detections when time intervals for the detection are near impossible values such as values nearing 0 seconds as this presents impossibly quick movements near the edge of the camera.

- Model 3: `faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28`
  - [Tensorflow]
  - I converted the model to an Intermediate Representation with the following arguments...
    - ```input_model=faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/frozen_inference_graph.pb```
      ```pip_config=faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/pipeline.config```
      ```cust_config=/opt/intel/openvino_2020.2.117/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json```
  - The model was insufficient for the app because even with improvements, the detection is janky, with the statistics shown on the app depending on the mqtt publish, the only solution is to make sure that model does not get detection wrong.
  - I tried to improve the model for the app by trying out different precisions and handling the wrong detections in `main.py` trying to suppress detections when time intervals for the detection are near impossible values such as values nearing 0 seconds as this presents impossibly quick movements near the edge of the camera

## Model Used

In the end I just looked up for models in the OpenVino pre-trained libraries and found out about `person-detection-retail-0013`. This has performed well with the resource video, eliminating the issue I have faced with the converted tensorflow models. The use of the model didn't completely eliminate the wrong detection especially on the edge of the video playback, when person is leaving the frame, but compared to the converted tensorflow models, this was easy to handle with the rule I have shared, which is to count it if the entry and exit time interval is greater than the near impossible time of ~0 seconds. Again, since the counter depends on the publishing with MQTT, with the said rule, publishing is done correctly.
