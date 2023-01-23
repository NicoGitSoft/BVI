import numpy as np
import depthai as dai
from pathlib import Path
import time, json, cv2
from math import pi, floor, atan2, cos, sin
from scipy.special import expit

def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    return cv2.resize(arr, shape).transpose(2,0,1)#.flatten()

def warp_rect_img(rect_points, img, w, h):
        src = np.array(rect_points[1:], dtype=np.float32) # rect_points[0] is left bottom point !
        dst = np.array([(0, 0), (h, 0), (h, w)], dtype=np.float32)
        mat = cv2.getAffineTransform(src, dst)
        return cv2.warpAffine(img, mat, (w, h))

def normalize_radians(angle):
    return angle - 2 * pi * floor((angle + pi) / (2 * pi))

def rotated_rect_to_points(cx, cy, w, h, rotation):
    b = cos(rotation) * 0.5
    a = sin(rotation) * 0.5
    p0x = cx - a*h - b*w
    p0y = cy + b*h - a*w
    p1x = cx + a*h - b*w
    p1y = cy - b*h - a*w
    p2x = int(2*cx - p0x)
    p2y = int(2*cy - p0y)
    p3x = int(2*cx - p1x)
    p3y = int(2*cy - p1y)
    p0x, p0y, p1x, p1y = int(p0x), int(p0y), int(p1x), int(p1y)
    return [[p0x,p0y], [p1x,p1y], [p2x,p2y], [p3x,p3y]]

def decode_bboxes(score_thresh, scores, bboxes, anchors, scale=128, best_only=False):
    regions = []
    scores = expit(scores) # 1 / (1 + np.exp(-scores))
    if best_only:
        best_id = np.argmax(scores)
        if scores[best_id] < score_thresh: return regions
        det_scores = scores[best_id:best_id+1]
        det_bboxes2 = bboxes[best_id:best_id+1]
        det_anchors = anchors[best_id:best_id+1]
    else:
        detection_mask = scores > score_thresh
        det_scores = scores[detection_mask]
        if det_scores.size == 0: return regions
        det_bboxes2 = bboxes[detection_mask]
        det_anchors = anchors[detection_mask]
    det_bboxes = det_bboxes2* np.tile(det_anchors[:,2:4], 9) / scale + np.tile(det_anchors[:,0:2],9)
    det_bboxes[:,2:4] = det_bboxes[:,2:4] - det_anchors[:,0:2]
    # box = [cx - w*0.5, cy - h*0.5, w, h]
    det_bboxes[:,0:2] = det_bboxes[:,0:2] - det_bboxes[:,3:4] * 0.5
    for i in range(det_bboxes.shape[0]):
        score = det_scores[i]
        box = det_bboxes[i,0:4]
        if box[2] < 0 or box[3] < 0: continue
        kps = []
        for kp in range(7):
            kps.append(det_bboxes[i,4+kp*2:6+kp*2])
        regions.append(HandRegion(float(score), box, kps))
    return regions

def detections_to_rect(regions):
    target_angle = pi * 0.5 # 90 = pi/2
    for region in regions:
        
        region.rect_w = region.pd_box[2]
        region.rect_h = region.pd_box[3]
        region.rect_x_center = region.pd_box[0] + region.rect_w / 2
        region.rect_y_center = region.pd_box[1] + region.rect_h / 2

        x0, y0 = region.pd_kps[0] # wrist center
        x1, y1 = region.pd_kps[2] # middle finger
        rotation = target_angle - atan2(-(y1 - y0), x1 - x0)
        region.rotation = normalize_radians(rotation)

def rect_transformation(regions, w, h):
    scale_x = 2.9
    scale_y = 2.9
    shift_x = 0
    shift_y = -0.5
    for region in regions:
        width = region.rect_w
        height = region.rect_h
        rotation = region.rotation
        if rotation == 0:
            region.rect_x_center_a = (region.rect_x_center + width * shift_x) * w
            region.rect_y_center_a = (region.rect_y_center + height * shift_y) * h
        else:
            x_shift = (w * width * shift_x * cos(rotation) - h * height * shift_y * sin(rotation)) #/ w
            y_shift = (w * width * shift_x * sin(rotation) + h * height * shift_y * cos(rotation)) #/ h
            region.rect_x_center_a = region.rect_x_center*w + x_shift
            region.rect_y_center_a = region.rect_y_center*h + y_shift

        # square_long: true
        long_side = max(width * w, height * h)
        region.rect_w_a = long_side * scale_x
        region.rect_h_a = long_side * scale_y
        region.rect_points = rotated_rect_to_points(region.rect_x_center_a, region.rect_y_center_a, region.rect_w_a, region.rect_h_a, region.rotation)


def hand_landmarks_to_rect(hand):
    # Calculates the ROI for the next frame from the current hand landmarks
    id_wrist = 0
    id_index_mcp = 5
    id_middle_mcp = 9
    id_ring_mcp =13
    
    lms_xy =  hand.landmarks[:,:2]
    # print(lms_xy)
    # Compute rotation
    x0, y0 = lms_xy[id_wrist]
    x1, y1 = 0.25 * (lms_xy[id_index_mcp] + lms_xy[id_ring_mcp]) + 0.5 * lms_xy[id_middle_mcp]
    rotation = 0.5 * pi - atan2(y0 - y1, x1 - x0)
    rotation = normalize_radians(rotation)
    # Now we work only on a subset of the landmarks
    ids_for_bounding_box = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18]
    lms_xy = lms_xy[ids_for_bounding_box]
    # Find center of the boundaries of landmarks
    axis_aligned_center = 0.5 * (np.min(lms_xy, axis=0) + np.max(lms_xy, axis=0))
    # Find boundaries of rotated landmarks
    original = lms_xy - axis_aligned_center
    c, s = np.cos(rotation), np.sin(rotation)
    rot_mat = np.array(((c, -s), (s, c)))
    projected = original.dot(rot_mat)
    min_proj = np.min(projected, axis=0)
    max_proj = np.max(projected, axis=0)
    projected_center = 0.5 * (min_proj + max_proj)
    center = rot_mat.dot(projected_center) + axis_aligned_center
    width, height = max_proj - min_proj
    next_hand = HandRegion()
    next_hand.rect_w_a = next_hand.rect_h_a = 2 * max(width, height)
    next_hand.rect_x_center_a = center[0] + 0.1 * height * s
    next_hand.rect_y_center_a = center[1] - 0.1 * height * c
    next_hand.rotation = rotation
    next_hand.rect_points = rotated_rect_to_points(
        next_hand.rect_x_center_a,
        next_hand.rect_y_center_a,
        next_hand.rect_w_a,
        next_hand.rect_h_a,
        next_hand.rotation)
    return next_hand

class HandRegion:
    def __init__(self, pd_score=None, pd_box=None, pd_kps=None):
        self.pd_score = pd_score # Palm detection score 
        self.pd_box = pd_box # Palm detection box [x, y, w, h] normalized
        self.pd_kps = pd_kps # Palm detection keypoints

    def get_rotated_world_landmarks(self):
        world_landmarks_rotated = self.world_landmarks.copy()
        sin_rot = sin(self.rotation)
        cos_rot = cos(self.rotation)
        rot_m = np.array([[cos_rot, sin_rot], [-sin_rot, cos_rot]])
        world_landmarks_rotated[:,:2] = np.dot(world_landmarks_rotated[:,:2], rot_m)
        return world_landmarks_rotated


SCRIPT_DIR = Path(__file__).resolve().parent
PALM_DETECTION_MODEL = str(SCRIPT_DIR / "Models/Hands Models/palm_detection_sh4.blob")
LANDMARK_MODEL_LITE = str(SCRIPT_DIR / "Models/Hands Models/hand_landmark_lite_sh4.blob")
YOLO_MODEL = str(SCRIPT_DIR / "Models/BVI Models/best3_openvino_2021.4_6shave.blob")
YOLO_CONFIG = str(SCRIPT_DIR / "Models/BVI Models/best3.json")

class YoloHandTracker:
    def __init__(self,
        pd_model=PALM_DETECTION_MODEL,
        ml_model=LANDMARK_MODEL_LITE,
        yolo_model=YOLO_MODEL,
        yolo_configurations=YOLO_CONFIG,
        use_yolo=True,
        use_hand=True,
        temperature_sensing=False, 
        pd_score_thresh=0.5,
        pd_nms_thresh=0.3,
        use_lm=True,
        lm_score_thresh=0.5,
        use_world_landmarks=False,
        solo=True,
        crop=True,
        xyz=True,
        internal_fps=23,
        internal_frame_height=640,
        use_handedness_average=False,
        single_hand_tolerance_thresh=10,
        stats=False,
        trace=0, 
        ):

        self.temperature_sensing = temperature_sensing
        if temperature_sensing: print("Temperature Sensing Enabled")

        self.use_hand = use_hand
        if use_hand: print("Hand Tracking Enabled")

        self.use_yolo = use_yolo
        if use_yolo:
            print("Yolo Enabled")
            self.yolo_model = yolo_model
            self.yolo_configurations = yolo_configurations
            print(f"YoloModel blob          : {self.yolo_model}")
            print(f"YoloConfigurations json : {self.yolo_configurations}")

            # Extraer metadata del archivo de configuración .json
            with open(self.yolo_configurations, 'r') as file:
                config = json.load(file)
            metadata = config.get("nn_config").get("NN_specific_metadata")
            self.yolo_classes = metadata.get("classes")
            self.yolo_coordinates = metadata.get("coordinates")
            self.yolo_anchors = metadata.get("anchors")
            self.yolo_anchor_masks = metadata.get("anchor_masks")
            self.yolo_iou_threshold = metadata.get("iou_threshold")
            self.yolo_confidence_threshold = metadata.get("confidence_threshold")
            self.yolo_labels = config.get("mappings").get("labels")
            self.width, self.height = tuple(map(int, config.get("nn_config").get("input_size").split("x")))

        self.pd_model = pd_model
        print(f"Palm detection blob     : {self.pd_model}")
        self.lm_model = ml_model
        print(f"Landmark blob           : {self.lm_model}")
        self.pd_score_thresh = pd_score_thresh
        self.pd_nms_thresh = pd_nms_thresh
        self.use_lm = use_lm
        self.lm_score_thresh = lm_score_thresh
        self.solo = solo
        self.lm_nb_threads = 2      
        self.max_hands = 1
        self.xyz = xyz
        self.crop = crop 
        self.use_world_landmarks = use_world_landmarks
        self.internal_fps = internal_fps     
        self.stats = stats
        self.trace = trace
        self.use_handedness_average = use_handedness_average
        self.single_hand_tolerance_thresh = single_hand_tolerance_thresh

        # Create Device
        self.device = dai.Device()

        self.resolution = (1920, 1080)
        print("Sensor resolution:", self.resolution)
        self.frame_size, self.scale_nd = internal_frame_height, (16, 27) # mpu.find_isp_scale_params(internal_frame_height, self.resolution)
        self.img_h = self.img_w = self.frame_size
        self.pad_w = self.pad_h = 0
        self.crop_w = (int(round(self.resolution[0] * self.scale_nd[0] / self.scale_nd[1])) - self.img_w) // 2
        print(f"Internal camera image size: {self.img_w} x {self.img_h} - crop_w:{self.crop_w} pad_h: {self.pad_h}")
        
        # MediaPipe Anchors 
        self.pd_input_length = 128 # 192 Palm detection
        self.anchors = np.loadtxt( str(SCRIPT_DIR / "MediaPipeAnchors.txt"), dtype=np.float16) # Open anchors file
        self.nb_anchors = self.anchors.shape[0]

        # Define and start pipeline
        self.device.startPipeline(self.create_pipeline())

        # Define data queues
        self.q_video = self.device.getOutputQueue(name="cam_out", maxSize=1, blocking=False)
        self.q_stereo_out = self.device.getOutputQueue(name="stereo_out", maxSize=4, blocking=False)

        if self.use_hand:
            self.q_pd_out = self.device.getOutputQueue(name="pd_out", maxSize=2, blocking=False)
            self.q_manip_cfg = self.device.getInputQueue(name="manip_cfg")
            self.q_lm_out = self.device.getOutputQueue(name="lm_out", maxSize=2, blocking=False)
            self.q_lm_in = self.device.getInputQueue(name="lm_in")
        if self.use_yolo:
            self.q_yolo_out = self.device.getOutputQueue(name="yolo_out", maxSize=4, blocking=False)
        if self.temperature_sensing:
            self.qSysInfo = self.device.getOutputQueue(name="sysinfo", maxSize=4, blocking=False)

        self.nb_frames_pd_inference = 0
        self.nb_frames_lm_inference = 0
        self.nb_lm_inferences = 0
        self.nb_failed_lm_inferences = 0
        self.nb_frames_lm_inference_after_landmarks_ROI = 0
        self.nb_frames_no_hand = 0
        self.nb_spatial_requests = 0
        self.glob_pd_rtrip_time = 0
        self.glob_lm_rtrip_time = 0
        self.glob_spatial_rtrip_time = 0
        self.use_previous_landmarks = False
        self.nb_hands_in_previous_frame = 0
        

    def create_pipeline(self):
        print("Creating pipeline...")
        # Start defining a pipeline
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2021_4)

        # ColorCamera
        print("Creating Color Camera...")
        cam = pipeline.createColorCamera()
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam.setInterleaved(False)
        cam.setIspScale(self.scale_nd[0], self.scale_nd[1])
        cam.setFps(self.internal_fps)
        cam.setVideoSize(self.frame_size, self.frame_size)
        cam.setPreviewSize(self.img_w, self.img_h)
        cam_out = pipeline.createXLinkOut()
        cam_out.setStreamName("cam_out")
        cam_out.input.setQueueSize(1)
        cam_out.input.setBlocking(False)
        cam.video.link(cam_out.input)

        mono_resolution = dai.MonoCameraProperties.SensorResolution.THE_400_P
        left = pipeline.createMonoCamera()
        left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        left.setResolution(mono_resolution)
        left.setFps(self.internal_fps)
        right = pipeline.createMonoCamera()
        right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        right.setResolution(mono_resolution)
        right.setFps(self.internal_fps)
        stereo = pipeline.createStereoDepth()
        stereo.initialConfig.setConfidenceThreshold(230)
        stereo.setLeftRightCheck(True)
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        stereo.setOutputSize(left.getResolutionWidth(), right.getResolutionHeight())
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        setero_out = pipeline.createXLinkOut()
        setero_out.setStreamName("stereo_out")
        stereo.depth.link(setero_out.input)
        left.out.link(stereo.left)
        right.out.link(stereo.right)

        if self.use_hand:
            # ImageManip
            manip = pipeline.createImageManip()   
            manip.inputImage.setQueueSize(1)
            manip.inputImage.setBlocking(False)
            manip.setMaxOutputFrameSize(1228800)
            manip.initialConfig.setResize(self.pd_input_length, self.pd_input_length)

            cam.preview.link(manip.inputImage)
            manip_cfg_in = pipeline.createXLinkIn()
            manip_cfg_in.setStreamName("manip_cfg")
            manip_cfg_in.out.link(manip.inputConfig)

            # Define palm detection model
            print("Creating Palm Detection Neural Network...")
            pd_nn = pipeline.createNeuralNetwork()
            pd_nn.setBlobPath(self.pd_model)     

            # Specify that network takes latest arriving frame in non-blocking manner
            pd_nn.input.setQueueSize(1)
            pd_nn.input.setBlocking(False)
            manip.out.link(pd_nn.input)

            # Palm detection output
            pd_out = pipeline.createXLinkOut()
            pd_out.setStreamName("pd_out")
            pd_nn.out.link(pd_out.input)
            
            # Define hand landmark model
            print("Creating Hand Landmark Neural Network")  
            lm_nn = pipeline.createNeuralNetwork()
            lm_nn.setBlobPath(self.lm_model)
            lm_nn.setNumInferenceThreads(self.lm_nb_threads)

            # Hand landmark input
            self.lm_input_length = 224
            lm_in = pipeline.createXLinkIn()
            lm_in.setStreamName("lm_in")
            lm_in.out.link(lm_nn.input)

            # Hand landmark output
            lm_out = pipeline.createXLinkOut()
            lm_out.setStreamName("lm_out")
            lm_nn.out.link(lm_out.input)
        
        if self.use_yolo:
            # YoloDetectionNetwork
            print("Creating Yolo Neural Network...")
            yolo = pipeline.create(dai.node.YoloDetectionNetwork)
            yolo.setBlobPath(self.yolo_model)
            yolo.setConfidenceThreshold(self.yolo_confidence_threshold)
            yolo.input.setBlocking(False)
            yolo.setNumClasses(self.yolo_classes)
            yolo.setCoordinateSize(self.yolo_coordinates)
            yolo.setAnchors(self.yolo_anchors)
            yolo.setAnchorMasks(self.yolo_anchor_masks)
            yolo.setIouThreshold(self.yolo_iou_threshold)
            yolo_out = pipeline.create(dai.node.XLinkOut)
            yolo_out.setStreamName("yolo_out")

            # Linking nodes for the output of yolo_out
            cam.preview.link(yolo.input)            # cam.preview -> yolo.input
            yolo.out.link(yolo_out.input)           # yolo.out -> yolo_out.input

        if self.temperature_sensing:
            sysLog = pipeline.create(dai.node.SystemLogger)
            sysLog.setRate(1)  # 1 Hz
            xoutTemp = pipeline.create(dai.node.XLinkOut)
            xoutTemp.setStreamName("sysinfo")
            sysLog.out.link(xoutTemp.input)          

        print("Pipeline created.")
        return pipeline        
   

    def pd_postprocess(self, inference):
        # print(inference.getAllLayerNames())
        scores = np.array(inference.getLayerFp16("classificators"), dtype=np.float64) # 896
        bboxes = np.array(inference.getLayerFp16("regressors"), dtype=np.float16).reshape((self.nb_anchors,18)) # 896x18
        # Decode bboxes
        hands = decode_bboxes(self.pd_score_thresh, scores, bboxes, self.anchors, scale=self.pd_input_length, best_only=self.solo)
        # Non maximum suppression (not needed if solo)
        detections_to_rect(hands)
        rect_transformation(hands, self.frame_size, self.frame_size)
        return hands

    def lm_postprocess(self, hand, inference):
        hand.lm_score = inference.getLayerFp16("Identity_1")[0]  
        if hand.lm_score > self.lm_score_thresh:  
            hand.handedness = inference.getLayerFp16("Identity_2")[0]
            lm_raw = np.array(inference.getLayerFp16("Identity_dense/BiasAdd/Add")).reshape(-1,3)
            # hand.norm_landmarks contains the normalized ([0:1]) 3D coordinates of landmarks in the square rotated body bounding box
            hand.norm_landmarks = lm_raw / self.lm_input_length

            # Now calculate hand.landmarks = the landmarks in the image coordinate system (in pixel)
            src = np.array([(0, 0), (1, 0), (1, 1)], dtype=np.float32)
            dst = np.array([ (x, y) for x,y in hand.rect_points[1:]], dtype=np.float32) # hand.rect_points[0] is left bottom point and points going clockwise!
            mat = cv2.getAffineTransform(src, dst)
            lm_xy = np.expand_dims(hand.norm_landmarks[:,:2], axis=0)
            hand.landmarks = np.squeeze(cv2.transform(lm_xy, mat)).astype(np.int32)


    def next_frame(self):
        self.depth_frame = self.q_stereo_out.get().getFrame()
        in_video = self.q_video.get()
        video_frame = in_video.getCvFrame()

        if self.temperature_sensing:
            self.chipTemperature = self.qSysInfo.get().chipTemperature.average
        else:
            self.chipTemperature = None

        if self.use_yolo:
            self.yolo_detections = self.q_yolo_out.get()
            self.yolo_detections = self.yolo_detections.detections
        else:
            self.yolo_detections = []
            self.yolo_labels = []

        if not self.use_hand:
            self.hands = []
        else:
            if not self.use_previous_landmarks:
                # Send image manip config to the device
                cfg = dai.ImageManipConfig()
                # We prepare the input to the Palm detector
                cfg.setResizeThumbnail(self.pd_input_length, self.pd_input_length)
                self.q_manip_cfg.send(cfg)

            if self.pad_h:
                square_frame = cv2.copyMakeBorder(video_frame, self.pad_h, self.pad_h, self.pad_w, self.pad_w, cv2.BORDER_CONSTANT)
            else:
                square_frame = video_frame

            # Get palm detection
            if self.use_previous_landmarks:
                self.hands = self.hands_from_landmarks # Use previous landmarks
            else: # Use palm detection
                inference = self.q_pd_out.get()
                hands = self.pd_postprocess(inference)
                self.nb_frames_pd_inference += 1
                if not self.solo and self.nb_hands_in_previous_frame == 1 and len(hands) <= 1:
                    self.hands = self.hands_from_landmarks
                else:
                    self.hands = hands
        
            if len(self.hands) == 0: # No hand detected
                self.nb_frames_no_hand += 1
            
            if self.use_lm:
                nb_lm_inferences = len(self.hands)
                # Hand landmarks, send requests
                for i,h in enumerate(self.hands):
                    img_hand = warp_rect_img(h.rect_points, square_frame, self.lm_input_length, self.lm_input_length)
                    nn_data = dai.NNData()   
                    nn_data.setLayer("input_1", to_planar(img_hand, (self.lm_input_length, self.lm_input_length)))
                    self.q_lm_in.send(nn_data)
                
                # Get inference results
                for i,h in enumerate(self.hands):
                    inference = self.q_lm_out.get()
                    #if i == 0: self.glob_lm_rtrip_time += now() - lm_rtrip_time
                    self.lm_postprocess(h, inference)

                # Filter hands with low confidence
                self.hands = [ h for h in self.hands if h.lm_score > self.lm_score_thresh]

                self.hands_from_landmarks = [hand_landmarks_to_rect(hand) for hand in self.hands]
                
                nb_hands = len(self.hands)

                # Stats
                if nb_lm_inferences: self.nb_frames_lm_inference += 1
                self.nb_lm_inferences += nb_lm_inferences
                self.nb_failed_lm_inferences += nb_lm_inferences - nb_hands 
                if self.use_previous_landmarks: self.nb_frames_lm_inference_after_landmarks_ROI += 1

                self.use_previous_landmarks = True
                if nb_hands == 0:
                    self.use_previous_landmarks = False
                #else: cv2.imshow("img_hand", img_hand)
                
                self.nb_hands_in_previous_frame = nb_hands           
                
                for hand in self.hands:
                    # If we added padding to make the image square, we need to remove this padding from landmark coordinates and from rect_points
                    if self.pad_h > 0:
                        hand.landmarks[:,1] -= self.pad_h
                        for i in range(len(hand.rect_points)):
                            hand.rect_points[i][1] -= self.pad_h
                    if self.pad_w > 0:
                        hand.landmarks[:,0] -= self.pad_w
                        for i in range(len(hand.rect_points)):
                            hand.rect_points[i][0] -= self.pad_w

                    # Set the hand label
                    hand.label = "right" if hand.handedness > 0.5 else "left"
        
        return video_frame, self.hands, self.yolo_detections, self.yolo_labels, self.img_h, self.img_w, self.depth_frame, self.chipTemperature


    def exit(self):
        self.device.close()       


###################### POGRAMA PRINCIPAL ######################
import cv2, serial, time, math, os, subprocess, platform
import scipy.io as sio
import numpy as np

# Coordenadas de los vertices un bounding box
def Vertices(detection):        
    x1 = int(detection.xmin * width)
    x2 = int(detection.xmax * width)
    y1 = int(detection.ymin * height)
    y2 = int(detection.ymax * height)
    return x1, x2, y1, y2

# Coordenada del centro de un bounding box
def Center(x1, x2, y1, y2):
    x = int((x1 + x2) / 2)
    y = int((y1 + y2) / 2)
    return x, y

# Coordenadas del centro del bounding box más cercano a un punto de referencia
def Nearest_Coordinate(refPoint, centroids):
    x_ref, y_ref = refPoint
    minDist = min((x-x_ref)**2 + (y-y_ref)**2 for x, y in centroids)
    for index, (x, y) in enumerate(centroids):
        if (x-x_ref)**2 + (y-y_ref)**2 == minDist:
            return x, y, index

# Función que crea una ROI a partir de una coordenada y un radio DELTA
def ROI(COORDINATE, DELTA):
    x, y = COORDINATE
    x1 = x - DELTA
    x2 = x + DELTA
    y1 = y - DELTA
    y2 = y + DELTA
    return x1, x2, y1, y2

# Función mapeada las coordenadas de una ROI x1, x2, y1, y2 en coordenadas de profundidad
def ROI2DepthROI(COORDINATES):
    x1, x2, y1, y2 = COORDINATES
    x1_depth = int((x1+192*2)/1024*640) - 120
    x2_depth = int((x2+192*2)/1024*640) - 120
    y1_depth = int(y1/1024*640)
    y2_depth = int(y2/1024*640)
    return x1_depth, x2_depth, y1_depth, y2_depth

# Función que calcula el promedio de los valores de profundidad de un ROI en metros
def AverageDepth(ROI, depthFrame):
    x1, x2, y1, y2 = ROI
    depth = depthFrame[y1:y2, x1:x2]
    z = np.nanmean(depth)/1000
    return z

# funciones anónimas para incremento de la frecuencia de pulsos en los vibradores
f1 = lambda x: math.sqrt(1 + x) - 1
f2 = lambda x: (x + 1)**2 - 1

##################### CONSTANTES Y CONFIGURACIONES #####################
width = height = 640 # Resolución de entrada de la red neuronal
VideoRGB = cv2.VideoWriter('VideoRGB.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

# Región de interés (ROI) para la estimación de la distancia de obstáculos z
DELTA = 30 # Radio de la ROI Cental en pixeles
DELTA_DOLL = 15 # Umbral de distancia en pixeles entre la referencia y el centroide de la detección más cercana (centroid of the nearest detection CND)
x_center, y_center = width//2, height//2 # Coordenadas del centro de la imagen
CentralROI = ROI((x_center, y_center), DELTA)

# Configuración del buzzer
bps = 10 # bits por segundo (envío de datos para sonar el buzzer por segundo)
zmin, zmax = 1, 2 # Umbral de distancia mínima y máxima para sonar el buzzer

# Estilos de dibujo (colores y timpografía)
BoxesColor = (0, 255, 0)
BoxesSize = 2
LineColor = (0, 0, 255)
CircleColor = (255, 0, 0)
TextColor = (0,0,255)
FontFace = cv2.FONT_HERSHEY_SIMPLEX
FontSize = 1

voice_messages = [
    'símbolo avión', 'recogida de equipajes', 'baños', 'peligro electricidad', 'flecha hacia abajo',
    'flecha hacia abajo de emergencia', 'salida de emergencia', "flecha izquierda de emergencia",
    "flecha derecha de emergencia", "flecha arriba de emergencia", "símbolo de extintor", "extintor", 
    "preferencial", "flecha izquierda", "no pasar", "restaurantes", "flecha derecha", "flecha izquierda",
    "flecha derecha", "flecha arriba"]

##################### inicialización de variables #####################
times = []      # Muestras de los tiempos de ejecución del programa desde la primera captura de frame
d = []   # Muestras de la distancias a la detecion más cercana con visión monocular
z = []   # Muestras de las distancias a los obstáculos en la ROI central con visión estereoscópica
h = []   # Muestras de las distancias horizontales en piexeles de la detección más cercana al centro de la imagen
v = []   # Muestras de las distancias verticales en piexeles de la detección más cercana al centro de la imagen 

haptic_messages  = []  # Muestras de los mensajes de activación del vibrador UP de la interfaz háptica
buzzer_messages  = []  # Muestras de los mensajes de activación del buzzer
nearest_labels = []  # Muestras de la etiquta de la deteción más cercana

# Muestras de temperatura
chipTemperatures = []  # Muestras de la temperatura del chip
cpuTemperature = []  # Muestras de la temperatura del CPU
max6675Temperature = []  # Muestras de la temperatura del sensor DHT22

# Declaración de variables
mentioned_object = False # bandera booleana para evitar que se repita el nombre del objeto detectado
mentioned_hand_use = False # bandera booleana para evitar que se repita el la mencion TTS del uso de la mano
frame_start_time = 0
frames_timer = 0
frames_counter = 0
start_move_time = 0
start_buzze_time = 0

# Rutas de los modelos neuronales y configuraciones
SCRIPT_DIR = Path(__file__).resolve().parent
PALM_DETECTION_MODEL = str(SCRIPT_DIR / "Models/Hands Models/palm_detection_sh4.blob")
LANDMARK_MODEL_LITE = str(SCRIPT_DIR / "Models/Hands Models/hand_landmark_lite_sh4.blob")
MY_YOLO_MODEL = str(SCRIPT_DIR / "Models/BVI Models/best3_openvino_2021.4_6shave.blob")
YOLO_CONFIG = str(SCRIPT_DIR / "Models/BVI Models/best3.json")

##################### Inicialización de objetos #####################
# Detectar si el sistema operativo es Raspbian para usar la termocupla MAX6675
try:
    import RPi.GPIO, max6675
    # set the pin for communicate with MAX6675
    cs = 22
    sck = 18
    so = 16
    max6675.set_pin(cs, sck, so, 1)
    Measure = True
except(ImportError, RuntimeError):
    Measure = False

# Inicializar el objeto para la detección de manos y señalizaciones de espacios interiores
tracker = YoloHandTracker(
    temperature_sensing = Measure,
    use_hand = False,
    use_yolo = True,
    yolo_model = MY_YOLO_MODEL,
    yolo_configurations = YOLO_CONFIG,
    )

visualize = True
try:# Intentar establecer un objeto para comunicación serial a usando UART 
    serial = serial.Serial("/dev/ttyS0", 9600, timeout=1)
    serial_is_connected = True
except:
    serial_is_connected = False



##################### Bucle principal #####################
loop_start_time = time.time()
while True:
    frame, hands, yolo_detections, labels, width, height, depthFrame, chip_temperature = tracker.next_frame()

    # Almacenar la distancia de los obstáculos en la ROI central
    z.append(AverageDepth(CentralROI, depthFrame))

    # Dedección de obstáculos en la ROI Central
    """ Uso del buzzer para sonar más frecuentemente a medida que se acerca un objeto en la ROI central"""
    if z[-1] < zmax and z[-1] > zmin: # si el obstáculo está a una distancia menor a zmax y mayor a zmin
        # Calcular el tiempo de espera entre cada envío de datos para sonar el buzzer
        wait_time_buzzer = 1/(bps)*z[-1]/(zmax-zmin) # segundos
        if f1(time.time() - start_buzze_time) > f2(wait_time_buzzer):
            if serial_is_connected: serial.write(b'0') # Sonar el buzzer
            start_buzze_time = time.time()
            buzzer_messages.append('1')
        else:
            buzzer_messages.append('nan')
    else:
        buzzer_messages.append('nan')

    # Detección de la mano del usuario
    """ Si se detecta una mano, se determina si la distandia de la muñeca a la cámera es menor a 1 metro,
    si es así, se le informa al usuario que que su mano está siendo untilizada como referencia para la 
    detección del objeto más cercano, de lo contrario se usa centro de la imagen como referencia """

    x_ref, y_ref = (x_center, y_center) # Usar coordenadas del centro de la imagen como referencia en primera instancia
    if len(hands) > 0: # Si se detecta la mano del usuario, cambiar la referencia a la punta del dedo índice
        x_doll, y_doll = hands[0].landmarks[0,:2] # Coordenadas de la muñeca
        x_index_finger, y_index_finger = hands[0].landmarks[8,:2] # Coordenadas de la punta del dedo índice
        dollROI = ROI((x_doll, y_doll), DELTA_DOLL) # ROI de la muñeca en la imagen de color
        dollDepthROI = ROI2DepthROI(dollROI) # ROI de la muñeca en la imagen de profundidad
        dollDistance = AverageDepth(dollDepthROI, depthFrame) # Distancia de la muñeca a la cámara
        color_use_hand = (0, 0, 255) # Color rojo si la punta del dedo índice está siendo utilizada como referencia
        
        # Identificar si es la mano del usuario o no en base a la distancia de la muñeca a la cámara
        if np.isnan(dollDistance) or dollDistance < .5: # Si la distancia de la muñeca a la cámara es menor a 50 cm
            x_ref, y_ref = x_index_finger, y_index_finger # Usar coordenadas de la punta del dedo índice como referencia
            color_use_hand = (0, 255, 0) # Color verde si la punta del dedo índice está siendo utilizada como referencia

            if not mentioned_hand_use:
                os.system("spd-say 'Dedo índice como referencia'") # Informar al usuario que su mano está siendo utilizada como referencia
                mentioned_hand_use = True # Activar la bandera para evitar que se repita el mensaje de uso de la mano
        else:
            mentioned_hand_use = False # Desactivar la bandera para que se vuelva a mencionar el uso de la mano

        if visualize: # Visualizar en rojo la ROI de la muñeca y la punta del dedo índice
            cv2.putText(frame, f"{dollDistance:.2f} m", (dollROI[1], dollROI[2]+DELTA_DOLL), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_use_hand, 1, cv2.LINE_AA)
            cv2.rectangle(frame, (dollROI[0], dollROI[2]), (dollROI[1], dollROI[3]), color_use_hand, 1)
            cv2.circle(frame, (x_doll, y_doll), 5, color_use_hand, -1) # Dibujar un círculo rojo en la muñeca
            cv2.circle(frame, (x_index_finger, y_index_finger), 5, color_use_hand, -1) # Dibujar un círculo rojo en la punta del dedo índice
            cv2.rectangle(depthFrame, (dollDepthROI[0], dollDepthROI[2]), (dollDepthROI[1], dollDepthROI[3]), color_use_hand, 1)

    # Detección del centro del bounding box más cercano al punto de referencia (CNBB)
    if len(yolo_detections) > 0:
        Centroids = []  # Coordenadas del centro de los objetos detectados
        for detection in yolo_detections:
            x1, x2, y1, y2 = Vertices(detection)
            x, y = Center(x1, x2, y1, y2)
            Centroids.append((x, y))
            label = labels[detection.label]
            confidence = detection.confidence * 100
            distance = AverageDepth(ROI2DepthROI(Vertices(detection)), depthFrame)
            if visualize:
                cv2.putText(frame, label , (x1, y1), FontFace, FontSize, TextColor, 2)
                cv2.putText(frame, "{:.0f} %".format(confidence), (x2, y), FontFace, FontSize, TextColor, 1)
                cv2.putText(frame, "{:.2f} [m]".format(distance) , (x2, y2), FontFace, FontSize, TextColor)
                cv2.rectangle(frame, (x1, y1), (x2, y2), BoxesColor, BoxesSize) 
        x, y, index = Nearest_Coordinate((x_ref, y_ref), Centroids) # Coordenadas del CNBB
        nearest_labels.append(labels[yolo_detections[index].label]) # Almacenar la etiqueta del CNBB
        d.append(AverageDepth(Vertices(yolo_detections[index]), depthFrame)) # Almacenar la profundidad del CNBB
        if visualize: cv2.arrowedLine(frame, (x_ref, y_ref), (x, y), LineColor, 2) # Dibujar una flecha desde el punto de referencia al CNBB

        # Comunicación serial para actuadores vibrotáctiles y buzzer
        """ Envío de mensajes para indicarcaciones vibrotáctiles {up,down,left,right} al usuario,
        y mencionar el nombre del objeto detectado más cercano si se entra en la ROI central
        mediante una voz sintetizada en el microprocesador. """
        h.append( abs(x - x_ref) )
        v.append( abs(y - y_ref) )
        if h[-1] < DELTA and v[-1] < DELTA:
            """ Decir mensaje traducido al español, basado en el objeto detectado más cercano y 
            la distancia a la cámara a la cual se encuentra. """
            if not mentioned_object:
                haptic_messages.append('c')
                msg = voice_messages[yolo_detections[index].label]
                os.system("spd-say '" + msg + "{:.2f} [m]".format(d[-1]) + " metros'")
                if serial_is_connected: serial.write(b'0')
                mentioned_object = True
            else:
                haptic_messages.append('nan')
        else:
            if h[-1] > v[-1]:
                wait_time = 1/bps * h[-1]/((width-DELTA)//2)
                if f1(time.time() - start_move_time) > f2(wait_time):
                    if (x - x_ref) > 0: # El objeto está a la derecha del centro de la imagen
                        if serial_is_connected: serial.write(b'r') # 68 ASCII
                        haptic_messages.append('r')
                    else: # El objeto está a la izquierda del centro de la imagen
                        if serial_is_connected: serial.write(b'l') # 76 ASCII
                        haptic_messages.append('l')
                    start_move_time = time.time()
                else:
                    haptic_messages.append('nan')
            else:
                wait_time = 1/bps * v[-1]/((height-DELTA)//2)
                if f1(time.time() - start_move_time) > f2(wait_time):
                    if (y - y_ref) > 0: # El objeto está abajo del centro de la imagen
                        if serial_is_connected: serial.write(b'd') # 82 ASCII
                        haptic_messages.append('d')
                    else: # El objeto está arriba del centro de la imagen
                        if serial_is_connected: serial.write(b'u') # 85 ASCII
                        haptic_messages.append('u')
                    start_move_time = time.time()
                else:
                    haptic_messages.append('nan')
            mentioned_object = False
    else:
        haptic_messages.append("nan")
        nearest_labels.append("nan")
        d.append("nan")
        h.append("nan")
        v.append("nan")

    # Contador de FPS 
    frames_counter += 1
    current_time = time.time()
    times.append(current_time - loop_start_time) # Almacenar el tiempo de ejecución
    if (current_time - frames_timer) > 1:
        if Measure: # Almacenar la temperatura del chip de sensor OAK-D cada segundo
            chipTemperatures.append(chip_temperature)
            max6675Temperature.append(max6675.read_temp(cs)) # Temperatura del sensor DHT22
            cpuTemperature.append(float(subprocess.check_output("vcgencmd measure_temp", shell=True).decode("utf-8").replace("temp=","").replace("'C\n",""))) # Temperatura de la CPU de la Raspberry Pi
        fps = ( frames_counter / (current_time - frames_timer) )
        frames_counter = 0
        frames_timer = current_time

    if visualize:
        # Visualizar la mapa de profundidad
        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)
        cv2.imshow("depth", depthFrameColor)

        # Mostrar el frame CentralROI y la profundidad de los obstáculos en la ROI central
        cv2.rectangle(frame, (CentralROI[0], CentralROI[2]), (CentralROI[1], CentralROI[3]), LineColor, 2)
        cv2.putText(frame, "{:.2f} m".format(z[-1]), (CentralROI[1], CentralROI[2]+DELTA), FontFace, FontSize, TextColor, 1)
        # Mostrar FPS y tiempo de ejecución
        cv2.putText(frame, "fps: {:.2f}".format(fps), (0,height-FontSize-6), FontFace, FontSize, TextColor, 2)
        cv2.putText(frame, "t: " + ("{:.2f} s".format(times[-1])), (0, 25), FontFace, FontSize, TextColor, 2) 
        # Mostrar el frame de la cámara RGB
        cv2.imshow("frame", frame)

    # Salir del programa si alguna de estas teclas son presionadas {ESC, SPACE, q}
    if cv2.waitKey(1) in [27, 32, ord('q')]:
        break

tracker.exit()

# Guardar los datos en un archivo .mat
if Measure: 
    sio.savemat('data.mat', {
        'z': z,
        'd': d,
        'h': h,
        'v': v,
        'times': times,
        'haptic_messages': haptic_messages,
        'nearest_labels': nearest_labels,
        'chipTemperatures': chipTemperatures,
        'max6675Temperature': max6675Temperature,
        'cpuTemperature': cpuTemperature
    })  
