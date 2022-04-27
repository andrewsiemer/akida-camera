import time
import os
import cv2
import threading
from imutils.video import VideoStream
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

from akida_models import akidanet_edge_imagenet_pretrained
from cnn2snn import convert
from akida import Model, FullyConnected, devices
import cv2

from fastapi import FastAPI, Request, WebSocket, Form, Response, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse, RedirectResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

MODEL_FBZ = "models/edge_learning_example.fbz"

# RTSP remote webcam address (set to 0 to use local webcam)
# CAMERA_SRC = "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4"
# CAMERA_SRC = "0" # use local webcam
CAMERA_SRC = 'rtsp://104.10.177.210:8554/unicast'

INFERENCE_PER_SECOND = 1

TEXT_COLOR = (190, 30, 255)

NUM_NEURONS_PER_CLASS = 1
NUM_WEIGHTS = 350
NUM_CLASSES = 10

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

TARGET_WIDTH = 224
TARGET_HEIGHT = 224

LABELS = {}
SAVED = []
SHOTS = {}
STATS = ""

# utility function to return key for any value
def get_key(val):
    for key, value in LABELS.items():
         if val == value:
             return key
 
    return -1

##################################################
# Web App
##################################################

app = FastAPI() # define application

# location of web service static files and html templates
templates = Jinja2Templates(directory='templates')

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.route('/video_feed')
def video_feed(request: Request):

    """
    Video streaming route.
    """

    # Put this in the src attribute of an img tag
    return StreamingResponse(camera.show_frame(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/")
def index(request: Request):

    """
    Video streaming home page.
    """
    
    return templates.TemplateResponse('index.html', { 'request': request })

@app.post("/add")
async def add(request: Request, label: str = Form(...)):
    # getting input with in HTML form
    if (label != ''):
        if (get_key(label) == -1):
            # add new class
            LABELS.update({len(LABELS):label})
        # learn class
        inference.learn(get_key(label))
        print("Learned Class: {}.".format(label))
    
    return templates.TemplateResponse('index.html', { 'request': request })

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(STATS.replace('\n', '</br>'))
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast("Client left.")

@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request, exc):
    return RedirectResponse("/")

##################################################
# Akida Demo
##################################################

class Camera:

    """
    Class to capture video feed from webcam
    """

    def __init__(self):
        print("Initializing camera")
        self.stream = VideoStream(
            src=CAMERA_SRC, resolution=(FRAME_WIDTH, FRAME_HEIGHT)
        ).start()
        self.label = ""
        self.shots = ""
        self.text_display_timer = 0

    def get_frame(self):
        frame = cv2.resize(self.stream.read(), (TARGET_WIDTH, TARGET_HEIGHT))
        return frame

    def get_input_array(self):
        frame = cv2.resize(self.stream.read(), (TARGET_WIDTH, TARGET_HEIGHT))
        input_array = img_to_array(frame)
        input_array = np.array([input_array], dtype="uint8")
        return input_array

    def show_frame(self):
        try:
            while True:
                frame = self.label_frame(self.stream.read())
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
        except GeneratorExit:
            pass

    def label_frame(self, frame):
        frame = cv2.putText(
            frame,
            str(self.label),
            (5, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )
        frame = cv2.putText(
            frame,
            str(self.shots),
            (5, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            TEXT_COLOR,
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
            device.soc.power_measurement_enabled = True
            #device.soc.clock_mode = akida.soc.ClockMode.Performance

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
        global STATS
        fps = pavg = pmin = pmax = energy = 0

        while True:
            input_array = camera.get_input_array()
            predictions = self.model_ak.predict_classes(input_array, num_classes=NUM_CLASSES)

            if predictions[0] in SAVED:
                self.camera.label = LABELS.get(predictions[0], predictions[0])
                self.camera.shots = "{} shot/s".format(SHOTS.get(predictions[0]))

            stats_raw = self.model_ak.statistics.__dict__
            fps = round(stats_raw.get('_fps'))
            powers = stats_raw.get('_powers')
            if powers:
                pavg = round(powers.get('Avg'))
                pmin = powers.get('Min')
                pmax = powers.get('Max')
            if stats_raw.get('_energy'):
                energy = round(stats_raw.get('_energy'))

            STATS = "Average framerate = {} fps\n".format(fps) + \
                    "Last inference power range (mW):  Avg {} / Min {} / Max {}\n".format(pavg, pmin, pmax) + \
                    "Last inference energy consumed (mJ/frame): {}".format(energy)

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
