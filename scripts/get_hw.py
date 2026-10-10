#!/usr/bin/env python3
from picamera2 import Picamera2
picam2 = Picamera2()
config = picam2.create_video_configuration(raw={"format": "SRGGB10"})
picam2.configure(config)
print(picam2.camera_configuration()["raw"])
picam2.start()
req = picam2.capture_request()
raw = req.make_array("raw")
print("dtype:", raw.dtype, "shape:", raw.shape)
req.release()