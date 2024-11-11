from ultralytics import YOLO, checks, hub
checks()

hub.login('ba0a16b08dc2e2640fcc1ca08265e4c9015d776310')

model = YOLO('https://hub.ultralytics.com/models/VTbduxyl8tiWp9Fo0Bkv')
results = model.train()
