import numpy as np
import depthai as dai
from pathlib import Path
from FPS import now
import time, json, cv2
from math import pi, floor, atan2, cos, sin


SCRIPT_DIR = Path(__file__).resolve().parent
PALM_DETECTION_MODEL = str(SCRIPT_DIR / "models/palm_detection_sh4.blob")
LANDMARK_MODEL_FULL = str(SCRIPT_DIR / "models/hand_landmark_full_sh4.blob")
LANDMARK_MODEL_LITE = str(SCRIPT_DIR / "models/hand_landmark_lite_sh4.blob")
LANDMARK_MODEL_SPARSE = str(SCRIPT_DIR / "models/hand_landmark_sparse_sh4.blob")

use_yolo = False

if use_yolo:
    YOLO_MODEL = str(SCRIPT_DIR / "models/yoloV4_tiny_openvino_2021.4_6shave.blob")
    YOLO_CONFIG = str(SCRIPT_DIR / "models/yoloV4_tiny.json")

    # Extraer metadata del archivo de configuración .json
    with open(YOLO_CONFIG, 'r') as file:
        config = json.load(file)
    metadata = config.get("nn_config").get("NN_specific_metadata")
    classes = metadata.get("classes")
    coordinates = metadata.get("coordinates")
    anchors = metadata.get("anchors")
    anchorMasks = metadata.get("anchor_masks")
    iouThreshold = metadata.get("iou_threshold")
    confidenceThreshold = metadata.get("confidence_threshold")
    labels = config.get("mappings").get("labels")
    width, height = tuple(map(int, config.get("nn_config").get("input_size").split("x")))

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
    points = []
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
    scores = 1 / (1 + np.exp(-scores))
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

class HandTracker:
    def __init__(self, input_src=None,
                pd_model=PALM_DETECTION_MODEL, 
                pd_score_thresh=0.5, pd_nms_thresh=0.3,
                use_lm=True,
                lm_score_thresh=0.5,
                use_world_landmarks=False,
                solo=True,
                crop=True,
                xyz=True,
                internal_fps=23,
                internal_frame_height=640,
                use_handedness_average=True,
                single_hand_tolerance_thresh=10,
                stats=False,
                trace=0, 
                ):

        self.pd_model = pd_model
        print(f"Palm detection blob : {self.pd_model}")

        self.lm_model = LANDMARK_MODEL_LITE
        print(f"Landmark blob       : {self.lm_model}")

        self.pd_score_thresh = pd_score_thresh
        self.pd_nms_thresh = pd_nms_thresh
        self.use_lm = use_lm
        self.lm_score_thresh = lm_score_thresh
        self.solo = solo

        self.lm_nb_threads = 1
        print("In Solo mode, # of landmark model threads is forced to 1")
            
        self.max_hands = 1
        
        self.xyz = True
        self.crop = crop 
        self.use_world_landmarks = use_world_landmarks
        self.internal_fps = internal_fps     
        self.stats = stats
        self.trace = trace
        self.use_handedness_average = use_handedness_average
        self.single_hand_tolerance_thresh = single_hand_tolerance_thresh

        self.input_type = "rgb"

        self.device = dai.Device()

        self.resolution = (1920, 1080)
        print("Sensor resolution:", self.resolution)

        #self.video_fps = self.internal_fps # Used when saving the output in a video file. Should be close to the real fps
        self.frame_size, self.scale_nd = 640, (16, 27) # mpu.find_isp_scale_params(internal_frame_height, self.resolution)
        self.img_h = self.img_w = self.frame_size
        self.pad_w = self.pad_h = 0
        self.crop_w = (int(round(self.resolution[0] * self.scale_nd[0] / self.scale_nd[1])) - self.img_w) // 2

        print(f"Internal camera image size: {self.img_w} x {self.img_h} - crop_w:{self.crop_w} pad_h: {self.pad_h}")
        

        # Create SSD anchors 
        self.pd_input_length = 128 # Palm detection
        # self.pd_input_length = 192 # Palm detection
        #self.anchors = mpu.generate_handtracker_anchors(self.pd_input_length, self.pd_input_length)
        self.anchors = np.loadtxt( str(SCRIPT_DIR / "MediaPipeAnchors.txt"), dtype=np.float16) # Open anchors file
        #print("\n"*2, self.anchors, "\n"*2)
        # save anchors to file txt
        #np.savetxt(".\\BVI\\Scripts\\YoloHand\\anchors.txt", self.anchors, fmt="%f")

        self.nb_anchors = self.anchors.shape[0]
        print(f"{self.nb_anchors} anchors have been created")

        # Define and start pipeline
        usb_speed = self.device.getUsbSpeed()
        self.device.startPipeline(self.create_pipeline())
        print(f"Pipeline started - USB speed: {str(usb_speed).split('.')[-1]}")

        # Define data queues
        if use_yolo:
            self.q_yolo_out = self.device.getOutputQueue(name="yolo_out", maxSize=4, blocking=False)

        self.q_video = self.device.getOutputQueue(name="cam_out", maxSize=1, blocking=False)
        self.q_pd_out = self.device.getOutputQueue(name="pd_out", maxSize=1, blocking=False)
        self.q_manip_cfg = self.device.getInputQueue(name="manip_cfg")


        self.q_lm_out = self.device.getOutputQueue(name="lm_out", maxSize=2, blocking=False)
        self.q_lm_in = self.device.getInputQueue(name="lm_in")

        self.q_spatial_data = self.device.getOutputQueue(name="spatial_data_out", maxSize=4, blocking=False)
        self.q_spatial_config = self.device.getInputQueue("spatial_calc_config_in")
        self.q_stereo_out = self.device.getOutputQueue(name="stereo_out", maxSize=4, blocking=False)

        #self.fps = FPS()

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
        if not self.solo: self.single_hand_count = 0
        

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

        # ImageManip
        manip = pipeline.createImageManip()
        manip.setMaxOutputFrameSize(self.pd_input_length*self.pd_input_length*3)
        manip.setWaitForConfigInput(True)
        manip.inputImage.setQueueSize(1)
        manip.inputImage.setBlocking(False)
        cam.preview.link(manip.inputImage)

        cam.setVideoSize(self.frame_size, self.frame_size)
        cam.setPreviewSize(self.img_w, self.img_h)
                
        manip_cfg_in = pipeline.createXLinkIn()
        manip_cfg_in.setStreamName("manip_cfg")
        manip_cfg_in.out.link(manip.inputConfig)

        cam_out = pipeline.createXLinkOut()
        cam_out.setStreamName("cam_out")
        cam_out.input.setQueueSize(1)
        cam_out.input.setBlocking(False)
        cam.video.link(cam_out.input)


        print("Creating MonoCameras, Stereo and SpatialLocationCalculator nodes...")
        # For now, RGB needs fixed focus to properly align with depth.
        # The value used during calibration should be used here
        calib_data = self.device.readCalibration()
        calib_lens_pos = calib_data.getLensPosition(dai.CameraBoardSocket.RGB)
        print(f"RGB calibration lens position: {calib_lens_pos}")
        cam.initialControl.setManualFocus(calib_lens_pos)

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
        
        stereo.setConfidenceThreshold(230)
        # LR-check is required for depth alignment
        stereo.setLeftRightCheck(True)
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        stereo.setOutputSize(left.getResolutionWidth(), right.getResolutionHeight())
        stereo.setSubpixel(False)  # subpixel True -> latency
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

        setero_out = pipeline.createXLinkOut()
        setero_out.setStreamName("stereo_out")
        stereo.depth.link(setero_out.input)
        
        spatial_location_calculator = pipeline.createSpatialLocationCalculator()
        spatial_location_calculator.setWaitForConfigInput(True)
        spatial_location_calculator.inputDepth.setBlocking(False)
        spatial_location_calculator.inputDepth.setQueueSize(1)

        spatial_data_out = pipeline.createXLinkOut()
        spatial_data_out.setStreamName("spatial_data_out")
        spatial_data_out.input.setQueueSize(1)
        spatial_data_out.input.setBlocking(False)

        spatial_calc_config_in = pipeline.createXLinkIn()
        spatial_calc_config_in.setStreamName("spatial_calc_config_in")

        left.out.link(stereo.left)
        right.out.link(stereo.right)    

        stereo.depth.link(spatial_location_calculator.inputDepth)

        spatial_location_calculator.out.link(spatial_data_out.input)
        spatial_calc_config_in.out.link(spatial_location_calculator.inputConfig)

        # Define palm detection model
        print("Creating Palm Detection Neural Network...")
        pd_nn = pipeline.createNeuralNetwork()
        pd_nn.setBlobPath(self.pd_model)
        # Palm detection input        

        # Specify that network takes latest arriving frame in non-blocking manner
        pd_nn.input.setQueueSize(1)
        pd_nn.input.setBlocking(False)
        manip.out.link(pd_nn.input)

        # Palm detection output
        pd_out = pipeline.createXLinkOut()
        pd_out.setStreamName("pd_out")
        pd_nn.out.link(pd_out.input)
        
         # Define hand landmark model
        print(f"Creating Hand Landmark Neural Network ({'1 thread' if self.lm_nb_threads == 1 else '2 threads'})...")         
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
        
        if use_yolo:
            # YoloDetectionNetwork
            print("Creating Yolo Neural Network...")
            yolo = pipeline.create(dai.node.YoloDetectionNetwork)
            yolo.setBlobPath(YOLO_MODEL)
            yolo.setConfidenceThreshold(confidenceThreshold)
            yolo.input.setBlocking(False)
            yolo.setNumClasses(classes)
            yolo.setCoordinateSize(coordinates)
            yolo.setAnchors(anchors)
            yolo.setAnchorMasks(anchorMasks)
            yolo.setIouThreshold(iouThreshold)
            yolo_out = pipeline.create(dai.node.XLinkOut)
            yolo_out.setStreamName("yolo_out")

            # Linking nodes for the output of yolo_out
            cam.preview.link(yolo.input)            # cam.preview -> yolo.input
            yolo.passthrough.link(cam_out.input)    # yolo.passthrough -> cam_out.input
            yolo.out.link(yolo_out.input)           # yolo.out -> yolo_out.input

        print("Pipeline created.")
        return pipeline        
   

    def pd_postprocess(self, inference):
        # print(inference.getAllLayerNames())
        scores = np.array(inference.getLayerFp16("classificators"), dtype=np.float16) # 896
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
            # hand.norm_landmarks[:,2] /= 0.4

            # Now calculate hand.landmarks = the landmarks in the image coordinate system (in pixel)
            src = np.array([(0, 0), (1, 0), (1, 1)], dtype=np.float32)
            dst = np.array([ (x, y) for x,y in hand.rect_points[1:]], dtype=np.float32) # hand.rect_points[0] is left bottom point and points going clockwise!
            mat = cv2.getAffineTransform(src, dst)
            lm_xy = np.expand_dims(hand.norm_landmarks[:,:2], axis=0)
            # lm_z = hand.norm_landmarks[:,2:3] * hand.rect_w_a  / 0.4
            hand.landmarks = np.squeeze(cv2.transform(lm_xy, mat)).astype(np.int32)


    def spatial_loc_roi_from_wrist_landmark(self, hand):
        zone_size = max(int(hand.rect_w_a / 10), 8)
        rect_center = dai.Point2f(*(hand.landmarks[0]-np.array((zone_size//2 - self.crop_w, zone_size//2 + self.pad_h))))
        rect_size = dai.Size2f(zone_size, zone_size)
        return dai.Rect(rect_center, rect_size)

    def query_xyz(self, spatial_loc_roi_func):
        conf_datas = []
        for h in self.hands:
            conf_data = dai.SpatialLocationCalculatorConfigData()
            conf_data.depthThresholds.lowerThreshold = 100
            conf_data.depthThresholds.upperThreshold = 10000
            conf_data.roi = spatial_loc_roi_func(h)
            conf_datas.append(conf_data)
        if len(conf_datas) > 0:
            cfg = dai.SpatialLocationCalculatorConfig()
            cfg.setROIs(conf_datas)
            
            spatial_rtrip_time = now()
            self.q_spatial_config.send(cfg)

            # Receives spatial locations
            spatial_data = self.q_spatial_data.get().getSpatialLocations()
            self.glob_spatial_rtrip_time += now() - spatial_rtrip_time
            self.nb_spatial_requests += 1
            for i,sd in enumerate(spatial_data):
                self.hands[i].xyz_zone =  [
                    int(sd.config.roi.topLeft().x) - self.crop_w,
                    int(sd.config.roi.topLeft().y),
                    int(sd.config.roi.bottomRight().x) - self.crop_w,
                    int(sd.config.roi.bottomRight().y)
                    ]
                self.hands[i].xyz = np.array([
                    sd.spatialCoordinates.x,
                    sd.spatialCoordinates.y,
                    sd.spatialCoordinates.z
                    ])

    def next_frame(self):

        depht_frame = self.q_stereo_out.get().getFrame()

        if use_yolo:
            yolo_detections = self.q_yolo_out.get()
            yolo_detections = yolo_detections.detections

        if not self.use_previous_landmarks:
            # Send image manip config to the device
            cfg = dai.ImageManipConfig()
            # We prepare the input to the Palm detector
            cfg.setResizeThumbnail(self.pd_input_length, self.pd_input_length)
            self.q_manip_cfg.send(cfg)

        in_video = self.q_video.get()
        video_frame = in_video.getCvFrame()
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
                cv2.imshow("img_hand", img_hand)
                nn_data = dai.NNData()   
                nn_data.setLayer("input_1", to_planar(img_hand, (self.lm_input_length, self.lm_input_length)))
                self.q_lm_in.send(nn_data)
                if i == 0: lm_rtrip_time = now() # We measure only for the first hand
            
            # Get inference results
            for i,h in enumerate(self.hands):
                inference = self.q_lm_out.get()
                #if i == 0: self.glob_lm_rtrip_time += now() - lm_rtrip_time
                self.lm_postprocess(h, inference)

            # Filter hands with low confidence
            self.hands = [ h for h in self.hands if h.lm_score > self.lm_score_thresh]

            if self.xyz:
                self.query_xyz(self.spatial_loc_roi_from_wrist_landmark)

            self.hands_from_landmarks = [hand_landmarks_to_rect(h) for h in self.hands]
            
            nb_hands = len(self.hands)

            # Stats
            if nb_lm_inferences: self.nb_frames_lm_inference += 1
            self.nb_lm_inferences += nb_lm_inferences
            self.nb_failed_lm_inferences += nb_lm_inferences - nb_hands 
            if self.use_previous_landmarks: self.nb_frames_lm_inference_after_landmarks_ROI += 1

            self.use_previous_landmarks = True
            if nb_hands == 0:
                self.use_previous_landmarks = False
            elif not self.solo and nb_hands == 1:
                    if self.single_hand_count >= self.single_hand_tolerance_thresh:
                        self.use_previous_landmarks = False
                        self.single_hand_count = 0
            
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


        if use_yolo:
            return video_frame, self.hands, yolo_detections, labels, width, height, depht_frame
        else:
            return video_frame, self.hands, self.img_w, self.img_h, depht_frame


##################### Inicialización de objetos #####################
tracker = HandTracker(xyz=True, solo=True, crop=True)

##################### Bucle principal #####################
while True:
    if use_yolo:
        frame, hands, yoloDetections, labels, width, height, depthFrame = tracker.next_frame()
    else:
        frame, hands, width, height, depthFrame = tracker.next_frame()

    x_center, y_center = (width//2, height//2) # Coordenadas del centro de la imagen
    x_ref, y_ref = (x_center, y_center) # Usar coordenadas del centro de la imagen como referencia en primera instancia
    if len(hands) > 0: # Si se detecta la mano del usuario, cambiar la referencia a la punta del dedo índice
        x_doll, y_doll = hands[0].landmarks[,:2] # Coordenadas de la muñeca
        
        cv2.circle(frame, (x_doll, y_doll), 5, (0, 0, 255), -1) # Dibujar un círculo en la muñeca
    cv2.imshow("frame", frame)
    # Salir del programa si alguna de estas teclas son presionadas {ESC, SPACE, q}
    if cv2.waitKey(1) in [27, 32, ord('q')]:
        break

tracker.exit()