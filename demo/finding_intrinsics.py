import pyrealsense2 as rs
# pipeline = rs.pipeline()
# cfg = pipeline.start() # Start pipeline and get the configuration it found
# profile = cfg.get_stream(rs.stream.color) # Fetch stream profile for depth stream
# intr = profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics
print(intr)