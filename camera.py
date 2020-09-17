# from pypylon import pylon
# from pypylon import genicam

# class Cameras:
#   def __init__(self):
#     print("init cameras")
#     self._tl_factory = pylon.TlFactory.GetInstance()
#     self._devices = self._tl_factory.EnumerateDevices()
#     self._cameras = pylon.InstantCameraArray(min(len(self._devices), 2))
#     print("{0} cameras found".format(len(self._devices)))

#     # Create and attach all Pylon Devices.
#     for i, cam in enumerate(self._cameras):
#       cam.Attach(self._tl_factory.CreateDevice(self._devices[i]))
#       self.setup_camera(cam)

#       # Print the model name of the camera.
#       print("Using device ", cam.GetDeviceInfo().GetSerialNumber())

#   def setup_camera(self, cam):
#     """ initialize camera """
#     try:
#       cam.Open()

#       # set default values
#       cam.OffsetX.SetValue(16)
#       cam.OffsetY.SetValue(16)
#       cam.Width.SetValue(640)
#       cam.Height.SetValue(480)
#       cam.BlackLevel.SetValue(0.0)
#       cam.Gain.SetValue(12)
#       cam.Gamma.SetValue(0.65)
#       cam.ExposureTime.SetValue(16000.0) #1100
#       cam.SensorReadoutMode.SetValue('Fast')
#       cam.AcquisitionFrameRateEnable.SetValue(False)
#       cam.AcquisitionFrameRate.SetValue(750)
#       cam.ChunkModeActive.SetValue(True)
#       cam.ChunkSelector.SetValue("CounterValue")
#       cam.ChunkEnable.SetValue(True)
#       cam.ChunkSelector.SetValue("Timestamp")
#       cam.ChunkEnable.SetValue(True)
#     except genicam.GenericException as ex:
#       print("An exception occurred.")
#       print(ex.GetDescription())
#       self.close_camera(cam)

#   def grab_one(self):
#     converter = pylon.ImageFormatConverter()
#     converter.OutputPixelFormat = pylon.PixelType_Mono8
#     converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
#     frames = []
#     for i, cam in enumerate(self._cameras):
#       grab_result = cam.GrabOne(100)
#       image = converter.Convert(grab_result)
#       frames.append(image.GetArray())

#     return frames[0], frames[1]

#   @staticmethod
#   def close_camera(cam):
#     """ close camera and destory cv2 windows """
#     try:
#       print('closing {0}'.format(cam.GetDeviceInfo().GetSerialNumber()))
#       cam.Close()
#     except genicam.GenericException as ex:
#       print("An exception occurred.")
#       print(ex.GetDescription())

#   def close_cameras(self):
#     for _, cam in enumerate(self._cameras):
#       self.close_camera(cam)
