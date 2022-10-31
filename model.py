!pip install https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt

import torch
model = torch.hub.load('ultralytics/yolov5', 'custom', 'best.pt')  # custom trained model

# Images
im = 'tb0018.png'  # or file, Path, URL, PIL, OpenCV, numpy, list

# Inference
results = model(im)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
results.show()  # or .show(), .save(), .crop(), .pandas(), etc.
results.save()  # or .show(), .save(), .crop(), .pandas(), etc.


results.xyxy[0]  # im predictions (tensor)
results.pandas().xyxy[0]  # im predictions (pandas)