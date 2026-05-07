
# Vorax


![python](https://img.shields.io/badge/python-3.10%2B-blue)
![opencv](https://img.shields.io/badge/opencv-enabled-success)
![yolo](https://img.shields.io/badge/ultralytics-yolo-informational)


[![Vorax logo](assets/logo.gif)](https://youtu.be/ucNhOMzyAVw)

# Vorax

Vorax is a computer-vision security app that monitors a defined “porch / drop-off” zone in a video feed, detects when a delivery has been made, and then watches for suspicious behavior that could indicate package theft. It combines YOLO-based object detection and tracking with zone-aware logic to confirm a package is stationary (delivered), identify likely delivery drivers, detect package movement or disappearance after delivery, and automatically send Telegram notifications with captured screenshots. The biggest challenges are handling occlusions, short-lived tracker IDs, and avoiding false alarms when packages briefly vanish behind a person; future improvements include stronger re-identification across occlusions, multi-package support, and more flexible per-camera rules.
**https://youtu.be/ucNhOMzyAVw?si=8vpLuD3koxsE7qVT**

### What it does

- Loads a YOLO model and tracks objects (package + person).
- Lets you define a zone (ROI) to focus detection on the drop-off area.
- Confirms “package received” when a package stays in the zone and becomes stationary.
- Detects potential theft if the package is moved significantly or goes missing after delivery/unattended placement.
- Sends Telegram notifications and captures multiple screenshots for evidence.


### Challenges and future work

- Occlusions (package hidden behind a person) can cause short “missing” bursts.
- Tracker IDs can change; the logic works around this by using timing windows and zone context.
- Future improvements: multi-package tracking, stronger identity handling through occlusions, and per-zone schedules.

## Tech Stack

- Python
- Ultralytics YOLO (`ultralytics`)
- OpenCV (`opencv-python`)
- PyYAML
- Requests

## How to Install and Run

### 1) Create a virtual environment (recommended)

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

### 3) Provide a video source

By default, the app looks for the configured video inside the `videos/` folder.

### 4) Run

UI mode (recommended for first run so you can define the zone):

```powershell
python main.py
```

Headless mode (requires a saved zone for that video):

```powershell
python main.py --headless
```

Override the source video path:

```powershell
python main.py --source "videos/your_video.mp4"
```

## How to Use

### Define the zone (ROI)

- On first run, the window opens and you can draw the zone.
- **Left-click + drag** to draw the rectangle.
- The zone is saved automatically to `zone.json` under the current video name.
- **Right-click** clears the saved zone for that video.

### What you’ll see

- Green box: general detection.
- Yellow box / label: suspicious (person in zone during a sensitive window).
- Red label: theft alert.
- Delivery label: delivery driver identified at package-received moment.

### Files generated

- Incident logs: `logs/incidents/` (JSON)
- Alert screenshots: `logs/alert_screenshots/`

## Configuration

Main config: `config/main.yaml`

Common settings:

- `video.name`: which file inside `videos/` to process
- `yolo.weights_primary`: main weights file
- `yolo.conf_threshold`, `yolo.iou_threshold`, `yolo.img_size`: model inference settings
- `classes.package_class_id`, `classes.person_class_id`: class mapping used by your model
- `detector.*`: detector timing/threshold settings

Notifier config: `config/notifier.yaml` (controls screenshots/capture cadence)

## Notifications (Telegram)

Create a `credentials.json` file (see `config/main.yaml` path settings) with:

```json
{
  "telegram_bot_token": "YOUR_BOT_TOKEN",
  "telegram_chat_id": "YOUR_CHAT_ID"
}
```

Notes:

- If `credentials.json` is missing or invalid, notifications are disabled and the app still runs.
- The app sends a test message on start if `notifier.send_test_message_on_start: true`.

## How to Contribute

- Fork the repository and create a feature branch.
- Keep changes small and focused.
- If you add tests, include how to run them.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

