# Fall Detection System using YOLOv8 and Email Alerts

This project implements a **real-time fall detection system** using the **YOLOv8 object detection model** and **OpenCV**.  
The system tracks people in a video, identifies potential fall incidents using heuristics like **aspect ratio change**, **centroid shift**, and **motion velocity**, and automatically **sends an email alert** with snapshot and GIF evidence when a fall is detected.

---

## How It Works
1. **YOLOv8** detects and tracks persons in each video frame using **ByteTrack**.  
2. The system calculates metrics such as:
   - Aspect ratio drop (standing → lying)
   - Centroid downward shift  
   - Detection confidence  
3. When sustained abnormal postures are detected for a few frames, a **fall event** is triggered.  
4. The system saves:
   - A **short MP4 clip**  
   - A **compressed GIF**  
   - A **snapshot image**
5. An **email alert** is sent via Gmail SMTP with the GIF and snapshot attached.

##  Requirements

```bash
pip install ultralytics opencv-python numpy imageio
```
## How to Run

Run the detection system using:
``` bash
python fainallll.py --video path_to_video.mp4 --visualize --alert
```

## Command Options:
```
--video : Input video file path
--visualize : Enables GUI visualization
--alert : Sends email alerts on fall detection
```

## Before running, configure your Gmail credentials:

EMAIL_SENDER = "your_email@gmail.com"
EMAIL_PASSWORD = "your_app_password"
EMAIL_RECIPIENT = "recipient_email@gmail.com"

## Email Alert Example

When a fall is detected, an email is automatically sent with:

Subject: 🚨 FALL ALERT - Track ID

Attachments: fall snapshot and GIF sequence

### Performance
``` bash
Model: YOLOv8m.pt
Confidence Threshold: 0.5
Tested On: Indoor fall detection videos
Accuracy: ~93–95% for clear frontal and side views
Processing Speed: 15–25 FPS (depending on hardware)
```

## Demo and Video Files

You can view the input video, output detection results, and the demo video showing the fall detection and alert trigger inside the videos/
 folder.
