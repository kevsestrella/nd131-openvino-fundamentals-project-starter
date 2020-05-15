# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

The process behind converting custom layers involves...introducing the custom layer for both Model Optimizer and
Inference Engine. If both are not done, both steps will give out errors. For example, Model Optimizer will not proceed
when tasked to convert an unrecognized model into IR, for Inference Engine, when fed with IR from a model with a custom
layer must be provided with extension that corresponds to the custom layer that the IR has. The extension and IR will be coming
from the 1st step , extgen to produce the files for MO and IE.

Some of the potential reasons for handling custom layers are...potential reasons for handling custom layer is that the
layers unsupported are crucial to inference, and that approaching the problem with other solution such as passing the handling to,
say, the origin of the model(TF, Caffe, etc.) is not possible.

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...

The difference between model accuracy pre- and post-conversion was...

The size of the model pre- and post-conversion was...bigger pre-conversion and smaller post conversion.
This is more notable on when we play with the precision values, going lower up to FP16.

The inference time of the model pre- and post-conversion was...

## Assess Model Use Cases

Some of the potential use cases of the people counter app are...Data Analytics, we can count people that goes to our retail, say,
on specific times, dates, and events. We can use people count to say what piques customer interest. We can also use People Counter App
on Pedestrian Crossings, we can efficiently automate the traffic lights depending on the count of the pedestrian waiting to be able to
cross over the other side of the street.

Each of these use cases would be useful because...reasons provided above.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...Since this things will probably differ between training
and inference, it will be much like the model is operating on an unseen data. It will then depend on how great the model was trained
prior its use. Lighting has great effect since it gives the model while training the advantage of accurately/precisely locating people,
but training the model under poor lighting condition and being able to locate people well will help inference on the same conditions.
As for the camera specifications such as focal length, this concers with the image size and the maginification of the objects, the shorter
the focal length the wider the viewing angle hence larger image and lower the magnification, objects appearing smaller, the longer the
focal length, the opposite it has on the image size and magnification. This well the inference if different specs are used between inference
and training. Say in training you have a long focal length, objects appearing larger, your bounding boxes will have to depend on that, and
on inference when shorter focal length is used, objects appearing smaller, the bounding boxes should have been smaller then at the training.

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [Name]
  - [Tensorflow]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...

- Model 2: [Name]
  - [Tensorflow]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...

- Model 3: [Name]
  - [Tensorflow]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
