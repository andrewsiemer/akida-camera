import time
import os
import cv2
import threading
from imutils.video import VideoStream
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
# from pynput import keyboard

from akida_models import akidanet_edge_imagenet_pretrained
from cnn2snn import convert
from akida import Model, FullyConnected, devices

from flask import Flask, render_template, Response
import cv2

OUTPUT = False
OUTPUT_VID = "out.avi"
OUTPUT_FPS = 30

MODEL_FBZ = "models/edge_learning_example.fbz"

# CAMERA_SRC = 0
CAMERA_SRC = 'rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4'

INFERENCE_PER_SECOND = 1

TEXT_COLOUR = (190, 30, 255)

NUM_NEURONS_PER_CLASS = 1
NUM_WEIGHTS = 350
NUM_CLASSES = 10

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

TARGET_WIDTH = 224
TARGET_HEIGHT = 224

NEURON_KEYS = [str(i) for i in range(10)]
SAVE_BUTTON = "s"

LABELS = {0: "Background"}
SAVED = []
SHOTS = {}


##################################################
# Web App
##################################################

app = Flask(__name__)

@app.route('/video_feed')
def video_feed():

    """
    Video streaming route.
    """

    # Put this in the src attribute of an img tag
    return Response(camera.show_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():

    """
    Video streaming home page.
    """

    return render_template('index.html')

##################################################
# Akida Demo
##################################################

# class Controls:

#     """
#     Class to capture key presses to save/learn
#     """

#     def __init__(self, inference):
#         self.listener = keyboard.Listener(
#             on_press=self.on_press, on_release=self.on_release
#         )
#         self.listener.start()
#         self.inference = inference

#     def on_press(self, key):
#         try:

#             if key.char in NEURON_KEYS:
#                 print("learned class {}".format(int(key.char)))
#                 self.inference.learn(int(key.char))

#             if key.char == SAVE_BUTTON:
#                 print("saved model to {}".format(MODEL_FBZ))
#                 self.inference.save()

#         except AttributeError:
#             pass

#     def on_release(self, key):
#         if key == keyboard.Key.esc:
#             return False


class Camera:

    """
    Class to capture video feed from webcam
    """

    def __init__(self):
        self.stream = VideoStream(
            src=CAMERA_SRC, resolution=(FRAME_WIDTH, FRAME_HEIGHT)
        ).start()
        self.label = ""
        self.shots = ""
        self.text_display_timer = 0
        if OUTPUT:
            self.out = cv2.VideoWriter(
                OUTPUT_VID,
                cv2.VideoWriter_fourcc(*"XVID"),
                OUTPUT_FPS,
                (FRAME_WIDTH, FRAME_HEIGHT),
            )

    def get_frame(self):
        frame = cv2.resize(self.stream.read(), (TARGET_WIDTH, TARGET_HEIGHT))
        return frame

    def get_input_array(self):
        frame = cv2.resize(self.stream.read(), (TARGET_WIDTH, TARGET_HEIGHT))
        input_array = img_to_array(frame)
        input_array = np.array([input_array], dtype="uint8")
        return input_array

    def show_frame(self):
        while True:
            frame = self.label_frame(self.stream.read())
            if OUTPUT:
                self.out.write(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

    def label_frame(self, frame):
        frame = cv2.putText(
            frame,
            str(self.label),
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            TEXT_COLOUR,
            1,
            cv2.LINE_AA,
        )
        frame = cv2.putText(
            frame,
            str(self.shots),
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            TEXT_COLOUR,
            1,
            cv2.LINE_AA,
        )
        return frame


class Inference:

    """
    Class to run inference over frames from the webcam
    """

    def __init__(self, camera):

        # create a model if one doesnt exist
        if not os.path.exists(MODEL_FBZ):
            print("Initialising Akida model")
            self.initialise()

        self.camera = camera

        # run inference in separate thread
        self.t1 = threading.Thread(target=self.infer)
        self.t1.start()

        # load the akida model
        self.model_ak = Model(filename=MODEL_FBZ)

        if len(devices()) > 0:
            device = devices()[0]
            self.model_ak.map(device)

    def initialise(self):

        """
        Method to initialise an Akida model if one doesn't exist
        """

        # fetch pretrained imagenet
        model_keras = akidanet_edge_imagenet_pretrained()

        # convert it to an Akida model
        model_ak = convert(model_keras)

        # remove the last layer of network, replace with Akida learning layer
        model_ak.pop_layer()
        layer_fc = FullyConnected(
            name="akida_edge_layer",
            units=NUM_CLASSES * NUM_NEURONS_PER_CLASS,
            activation=False,
        )
        # add learning layer to end of model
        model_ak.add(layer_fc)
        model_ak.compile(
            num_weights=NUM_WEIGHTS, num_classes=NUM_CLASSES, learning_competition=0.1
        )
        # save new model
        model_ak.save(MODEL_FBZ)

    def infer(self):
        while True:
            input_array = camera.get_input_array()
            predictions = self.model_ak.predict_classes(input_array, num_classes=NUM_CLASSES)
            if predictions[0] in SAVED:
                self.camera.label = LABELS.get(predictions[0], predictions[0])
                self.camera.shots = "{} shot/s".format(SHOTS.get(predictions[0]))
            time.sleep(1 / INFERENCE_PER_SECOND)

    def learn(self, neuron):
        if neuron not in SAVED:
            SAVED.append(neuron)
            SHOTS[neuron] = 1
        else:
            SHOTS[neuron] += 1

        input_array = self.camera.get_input_array()
        self.model_ak.fit(input_array, neuron)
        self.camera.label = "Learned {}".format(LABELS.get(neuron, neuron))

    def save(self):
        self.model_ak.save(MODEL_FBZ)


camera = Camera()
inference = Inference(camera)
# controls = Controls(inference)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')