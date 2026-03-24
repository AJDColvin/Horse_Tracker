# Horse Activity Tracking

A Python-based computer vision tool for autonomously tracking animal using YOLO and BotSORT.

## Features

- **Automated Tracking:** Uses YOLO object detection combined with BotSORT tracking to maintain consistent IDs on animals across video frames.
- **Movement Classification:** Calculates the frame-by-frame Euclidean distance of horse centroids, normalised by bounding box size, to classify activity as `MOVING` or `STILL` or (`OUT OF FRAME`).
- **Data Smoothing:** Applies a rolling majority-vote to filter out any stutters in the activity classification.
- **Excel Report:** Automatically compiles an Excel (`.xlsx`) report detailing continuous activity blocks, start/end timestamps, coordinates, and total activity percentages per horse.
- **Video Output:** Optionally generate annotated playback videos with tracking boxes, real-time classifications, and timestamps.


## Requirements

Ensure you have Python 3.8+ installed along with the following primary libraries:
- `ultralytics`
- `opencv-python` (`cv2`)
- `torch`
- `pandas`
- `openpyxl`
- `matplotlib`

Install them via pip:
```bash
pip install ultralytics opencv-python torch pandas openpyxl matplotlib
```

## Usage

You can run the script from the command line and pass various arguments to customise the tracking algorithm to your needs. 


### Basic Run
Provide the path to the video. The script will use standard yolo11s, track 2 individuals, and export results to `horse_activity_log.xlsx` by default.
```bash
python horse_tracker.py --video "path/to/your/video.mp4"
```

### Advanced Run
Customise the YOLO model, thresholds, and export a annotated video.
```bash
python horse_tracker.py \
    --video "clip.mp4" \
    --model "models/yolo_custom_horse.pt" \
    --custom-model \
    --excel-out "outputs/logs/custom_report.xlsx" \
    --video-out "outputs/videos/amended_output.mp4" \
    --individuals 3 \
    --smoothing-window 75 \
    --plot
```

### CLI Arguments Reference

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--video` | Path to the input video. | `clip_1.mp4` |
| `--model` | Path to the `.pt` YOLO model. | `../YOLO_models/yolo11s.pt` |
| `--excel-out` | Path to save the output Excel file. | `horse_activity_log.xlsx` |
| `--video-out` | Path to export the annotated playback video. If unset, no video saves. | *None* |
| `--threshold` | Movement threshold used to distinguish moving/still boundaries. Increase if movement is detected while the object is still. Decrease if still is detected while the object is moving| `600.0` |
| `--individuals`| The maximum number of individuals to trace and log explicitly. For example if there are 5 horses in your paddock, set to 5.| `2` |
| `--smoothing-window`| Size of the rolling filter window in frames (removes stutters). Increase to ignore brief changes in states. | `75` |
| `--custom-model` | Flag to skip class rectification. Use if model is exclusively trained on one animal class eg. Horse. | `False` |
| `--plot` | Flag to render a popup `matplotlib` graph of activity states. | `False` |

