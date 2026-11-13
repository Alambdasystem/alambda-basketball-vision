# Third-party materials

## YOLOX model and post-processing example

`process_clip.py` follows the YOLOX grid/stride decoding approach in the [OpenCV Zoo YOLOX example](https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_yolox).

The sample model is `object_detection_yolox_2022nov.onnx`. Its model-directory Apache 2.0 license is included in [assets/YOLOX_LICENSE.txt](assets/YOLOX_LICENSE.txt). The model is downloaded separately and verified against the SHA-256 recorded in `download_assets.py`.

## Basketball footage

The raw sample is linked by the [Roboflow basketball tutorial](https://roboflow.com/blog/identify-basketball-players) and its [notebook](https://github.com/roboflow/notebooks/blob/main/notebooks/basketball-ai-how-to-detect-track-and-identify-basketball-players.ipynb). [assets/source.json](assets/source.json) records the exact source and file properties.

The raw footage is not committed. The short processed demonstration and preview contain excerpts of that third-party broadcast. The repository does not claim ownership of the footage or grant a separate redistribution license for it. The Apache 2.0 project license covers the project code, not the underlying broadcast or trademarks.

## Dependencies

Python dependencies are listed in `requirements.txt`; FFmpeg is a separate system dependency. Each dependency remains subject to its own upstream license. They are installed from their respective distributions and are not vendored here.
