from ultralytics import RTDETR, YOLO


# cfg为要测试的模型的
# source为用来测试的源

cfg = r"model/yolov8m-seg-2stream-add_4.yaml"
source = r"F:\Code\ultralytics_two_stream\ultralytics\assets\bus.jpg"


def test_model_forward():
    """Test the forward pass of the YOLO model."""
    model = YOLO(cfg, verbose=True)
    model(source)  # also test no source and augment
