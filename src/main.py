import argparse
import time
import os
from collections import deque, defaultdict
from dataclasses import dataclass
import cv2
import numpy as np
from ultralytics import YOLO
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime
import imageio

MODEL_NAME = "yolov8m.pt"  # ultralytics pre-trained model
CONF_THRESHOLD = 0.5      # detection confidence threshold for fall trigger
ASPECT_DROP_NORM = 0.4     # expected aspect drop for strong signal
Y_SHIFT_NORM = 0.1         # expected downward centroid shift
WINDOW_SECONDS = 1.5       # seconds of history to analyze
SUSTAINED_FRAMES = 3       # frames for triggering
MIN_ASPECT_RATIO = 0.8     # fallen person should have aspect < this (width > height)
RECOVERY_ASPECT_RATIO = 1.2  # person is standing if aspect > this (height > width)
RECOVERY_FRAMES = 5        # frames needed to confirm recovery
GIF_DURATION = 6.0         # Duration of GIF/video clip in seconds

# Email configuration
EMAIL_SENDER = "priyadharshini*******@gmail.com"  
EMAIL_PASSWORD = "**** **** **** zwcp"    
EMAIL_RECIPIENT = "priyadharshini*********@gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# Output directory
OUT_DIR = "fall_outputs"
os.makedirs(OUT_DIR, exist_ok=True)


@dataclass
class TrackedPerson:
    id: int
    bbox_history: deque 
    last_seen: float
    fall_state: dict     


def xyxy_to_aspect_centroid(box, frame_w, frame_h):
    """Calculate aspect ratio and normalized centroid y-coordinate"""
    x1, y1, x2, y2 = box
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    aspect = h / w 
    centroid_y = ((y1 + y2) / 2.0) / frame_h 
    return aspect, centroid_y


def send_email_alert(subject, body, snapshot_path=None, gif_path=None, metrics=None):
    """Send email alert with image (with red text) and GIF attached"""
    from email.mime.base import MIMEBase
    from email import encoders
    import cv2

    for attempt in range(3):
        try:
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
            server.quit()
            print(f"[EMAIL] Alert with GIF and image sent to {EMAIL_RECIPIENT}")
            return True
        except Exception as e:
            print(f"[EMAIL] Attempt {attempt+1} failed: {e}")
            time.sleep(3)
    try:
        if snapshot_path and os.path.exists(snapshot_path):
            img = cv2.imread(snapshot_path)
            if img is not None:
                cv2.putText(img, "PERSON FALL DETECTED", (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)
                cv2.imwrite(snapshot_path, img)

        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECIPIENT
        msg['Subject'] = subject
        html_body = f"""
        <html>
        <body>
            <h2 style="color:red;">🚨 PERSON FALL DETECTED!</h2>
            <p>{body}</p>
        """
        if metrics:
            html_body += "<p><b>Detection Metrics:</b></p><ul>"
            for k, v in metrics.items():
                html_body += f"<li>{k}: {v}</li>"
            html_body += "</ul>"

        html_body += "</body></html>"
        msg.attach(MIMEText(html_body, 'html'))
        if snapshot_path and os.path.exists(snapshot_path):
            with open(snapshot_path, 'rb') as f:
                img_part = MIMEImage(f.read())
                img_part.add_header('Content-Disposition', 'attachment', filename='fall_snapshot.jpg')
                msg.attach(img_part)

        # --- Attach GIF (6–7 sec) ---

        if gif_path and os.path.exists(gif_path) and os.path.getsize(gif_path) > 20 * 1024 * 1024:
            print("[EMAIL] GIF too large, skipping attachment.")
            gif_path = None


        
        if gif_path and os.path.exists(gif_path):
            with open(gif_path, 'rb') as f:
                gif_part = MIMEBase('application', 'octet-stream')
                gif_part.set_payload(f.read())
            encoders.encode_base64(gif_part)
            gif_part.add_header('Content-Disposition', 'attachment', filename='fall_sequence.gif')
            msg.attach(gif_part)

        # Send via Gmail SMTP
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()

        print(f"[EMAIL] Alert with GIF and image sent to {EMAIL_RECIPIENT}")
        return True

    except Exception as e:
        print(f"[EMAIL] Failed to send alert: {e}")
        return False


def calculate_fall_confidence(bbox_history, fps):
    """
    Calculate fall detection confidence based on multiple heuristics
    Returns: (confidence, metrics_dict)
    """
    if len(bbox_history) < 3:
        return 0.0, {}
    
    # Extract data
    arr = list(bbox_history)
    times = np.array([x[0] for x in arr])
    aspects = np.array([x[5] for x in arr])  # aspect ratio
    centroids_y = np.array([x[6] for x in arr])  # vertical position
    det_confs = np.array([x[7] for x in arr])  # detection confidence
    
    # Split into earlier & recent halves
    n = len(arr)
    mid = max(1, n // 2)
    
    # 1. Aspect Ratio Analysis (shape change)
    prev_aspect = np.mean(aspects[:mid])
    recent_aspect = np.mean(aspects[mid:])
    aspect_drop = max(0.0, prev_aspect - recent_aspect)
    aspect_signal = min(1.0, aspect_drop / ASPECT_DROP_NORM)
    
    # 2. Vertical Position Change (downward movement)
    prev_cy = np.mean(centroids_y[:mid])
    recent_cy = np.mean(centroids_y[mid:])
    y_shift = recent_cy - prev_cy
    y_signal = min(1.0, max(0.0, y_shift / Y_SHIFT_NORM))
    
    # 3. Aspect Ratio Velocity (rate of shape change)
    time_span = max(1e-6, times[-1] - times[0])
    aspect_velocity = (aspects[-1] - aspects[0]) / time_span
    velocity_signal = min(1.0, max(0.0, -aspect_velocity * fps / 2.0))
    
    # 4. Absolute Aspect Check (is person currently horizontal?)
    current_aspect = aspects[-1]
    horizontal_signal = 1.0 if current_aspect < MIN_ASPECT_RATIO else 0.0
    
    # 5. Detection Confidence (is YOLO confident about detection?)
    avg_det_conf = np.mean(det_confs)
    det_signal = avg_det_conf
    
    # Combined confidence with weighted factors
    confidence = (
        0.35 * aspect_signal +      # Shape change is critical
        0.25 * y_signal +            # Downward movement important
        0.20 * velocity_signal +     # Speed of fall matters
        0.15 * horizontal_signal +   # Current orientation
        0.05 * det_signal            # Detection quality
    )
    
    confidence = float(np.clip(confidence, 0.0, 1.0))
    
    metrics = {
        'confidence': confidence,
        'aspect_change': aspect_drop,
        'y_shift': y_shift,
        'aspect_velocity': aspect_velocity,
        'current_aspect': current_aspect,
        'avg_detection_conf': avg_det_conf,
        'aspect_signal': aspect_signal,
        'y_signal': y_signal,
        'velocity_signal': velocity_signal,
        'horizontal_signal': horizontal_signal
    }
    
    return confidence, metrics


def check_recovery(bbox_history):
    """Check if person has recovered (standing up) from fall"""
    if len(bbox_history) < RECOVERY_FRAMES:
        return False
    
    # Get recent aspect ratios
    recent_aspects = [x[5] for x in list(bbox_history)[-RECOVERY_FRAMES:]]
    
    # Check if all recent frames show standing posture
    standing_count = sum(1 for aspect in recent_aspects if aspect >= RECOVERY_ASPECT_RATIO)
    
    # Person is recovered if most recent frames show standing
    return standing_count >= (RECOVERY_FRAMES - 1)


def save_gif_from_buffer(frames_buffer, out_path, fps=10):
    """Save frames as GIF"""
    try:
        if len(frames_buffer) == 0:
            return False
        
        # Convert BGR to RGB for imageio
        rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_buffer]
        
        # Save as GIF
        imageio.mimsave(out_path, rgb_frames, fps=fps, loop=0)
        print(f"[GIF] Saved to {out_path}")
        return True
    except Exception as e:
        print(f"[GIF] Failed to save: {e}")
        return False
    
def compress_gif(input_path, output_path, max_size=(360, 240), fps=5):
    """Compress GIF to reduce file size for Gmail."""
    import imageio, cv2
    frames = imageio.mimread(input_path)
    resized = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2BGR), max_size) for f in frames]
    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in resized]
    imageio.mimsave(output_path, rgb_frames, fps=fps, loop=0, palettesize=128, subrectangles=True)




def save_video_from_buffer(frames_buffer, out_path, fps=20.0):
    """Save frames as video"""
    try:
        if len(frames_buffer) == 0:
            return False
        
        h, w = frames_buffer[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        
        for frame in frames_buffer:
            out.write(frame)
        
        out.release()
        print(f"[VIDEO] Saved to {out_path}")
        return True
    except Exception as e:
        print(f"[VIDEO] Failed to save: {e}")
        return False


def detect_falls_in_video(video_path, visualize=True, save_clips_on_fall=False, alert_on_fall=False):
    """Main fall detection function using YOLO tracking"""
    
    # Initialize YOLO with tracking
    model = YOLO(MODEL_NAME)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    window_frames = max(3, int(WINDOW_SECONDS * fps))
    
    print(f"Video info: {frame_w}x{frame_h} @ {fps:.1f}fps, {total_frames} frames")
    
    tracked = {}  # track_id -> TrackedPerson
    frame_idx = 0
    start_time = time.time()
    
    # For saving clips/GIFs
    ring_buffer = deque(maxlen=int(fps * 10))  # last 10 seconds for context
    fall_events = []
    alerted_tracks = set()  # prevent duplicate alerts
    
    # For output video
    output_video_path = os.path.join(OUT_DIR, f"output_{os.path.basename(video_path)}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_w, frame_h))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # seconds
        frame_idx += 1
        
        # Use YOLO's built-in tracking (ByteTrack)
        results = model.track(
            source=frame,
            persist=True,      # persist tracks between frames
            tracker="bytetrack.yaml",  # use ByteTrack tracker
            conf=0.7,          # lower confidence for detection
            classes=[0],       # only track persons (class 0)
            verbose=False
        )
        
        # Create a copy for visualization
        display_frame = frame.copy()
        
        # Save frame to ring buffer with timestamp
        ring_buffer.append({'time': t, 'frame': frame.copy()})
        
        # Process tracked objects
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            
            # Update or create tracks
            current_tracks = set()
            for box, track_id, conf in zip(boxes, track_ids, confidences):
                current_tracks.add(track_id)
                x1, y1, x2, y2 = box
                aspect, cy = xyxy_to_aspect_centroid(box, frame_w, frame_h)
                
                if track_id not in tracked:
                    # Create new track
                    dq = deque(maxlen=window_frames)
                    dq.append((t, x1, y1, x2, y2, aspect, cy, conf))
                    tracked[track_id] = TrackedPerson(
                        id=track_id,
                        bbox_history=dq,
                        last_seen=t,
                        fall_state={
                            'sustained_count': 0,
                            'triggered': False,
                            'trigger_time': None,
                            'best_conf': 0.0,
                            'fall_frame': None,
                            'recording_started': False,
                            'recording_buffer': [],
                            'recording_start_time': None,
                            'recovery_count': 0
                        }
                    )
                else:
                    # Update existing track
                    tracked[track_id].bbox_history.append((t, x1, y1, x2, y2, aspect, cy, conf))
                    tracked[track_id].last_seen = t
            
            # Remove stale tracks
            stale = [tid for tid in tracked if tid not in current_tracks and t - tracked[tid].last_seen > 2.0]
            for tid in stale:
                del tracked[tid]
        
        # Evaluate fall detection for each active track
        for track_id, tr in tracked.items():
            # Need sufficient history
            if len(tr.bbox_history) < max(3, int(0.3 * fps)):
                continue
            
            # Calculate fall confidence
            conf, metrics = calculate_fall_confidence(tr.bbox_history, fps)
            
            # Update best confidence
            if conf > tr.fall_state['best_conf']:
                tr.fall_state['best_conf'] = conf
            
            # Check for recovery (person standing up)
            if tr.fall_state['triggered']:
                if check_recovery(tr.bbox_history):
                    tr.fall_state['recovery_count'] += 1
                    
                    # Confirm recovery after RECOVERY_FRAMES
                    if tr.fall_state['recovery_count'] >= RECOVERY_FRAMES:
                        print(f"\n[RECOVERY] Track {track_id} - Person has stood up")
                        # Reset fall state but keep track_id in alerted_tracks to prevent re-alerting
                        tr.fall_state['triggered'] = False
                        tr.fall_state['sustained_count'] = 0
                        tr.fall_state['recording_started'] = False
                        tr.fall_state['recovery_count'] = 0
                else:
                    tr.fall_state['recovery_count'] = 0
            
            # Check if fall detected (3 consecutive frames)
            if conf >= CONF_THRESHOLD and not tr.fall_state['triggered']:
                tr.fall_state['sustained_count'] += 1
            elif conf < CONF_THRESHOLD and not tr.fall_state['triggered']:
                tr.fall_state['sustained_count'] = 0
            
            # Start recording after 3 consecutive fall detections
            if tr.fall_state['sustained_count'] >= SUSTAINED_FRAMES and not tr.fall_state['triggered']:
                trigger_time = t
                tr.fall_state['triggered'] = True
                tr.fall_state['trigger_time'] = trigger_time
                tr.fall_state['recording_started'] = True
                tr.fall_state['recording_start_time'] = trigger_time
                
                print(f"\n{'='*60}")
                print(f"[FALL DETECTED] Track ID: {track_id}")
                print(f"Time: {trigger_time:.2f}s (Frame: {frame_idx})")
                print(f"Confidence: {conf:.2%}")
                print(f"Starting GIF/video recording...")
                print(f"{'='*60}\n")
            
            # Record frames for GIF/video (for 5-6 seconds after trigger)
            if tr.fall_state['recording_started'] and tr.fall_state['triggered']:
                tr.fall_state['recording_buffer'].append(display_frame.copy())
                
                # Check if we've recorded enough (5-6 seconds)
                recording_duration = t - tr.fall_state['recording_start_time']
                if recording_duration >= GIF_DURATION and track_id not in alerted_tracks:
                    # Save GIF
                    gif_path = os.path.join(OUT_DIR, f"fall_track{track_id}_{int(trigger_time)}.gif")
                    save_gif_from_buffer(tr.fall_state['recording_buffer'], gif_path, fps=10)
                    
                    # Save video clip
                    clip_path = os.path.join(OUT_DIR, f"fall_track{track_id}_{int(trigger_time)}.mp4")
                    save_video_from_buffer(tr.fall_state['recording_buffer'], clip_path, fps=fps)
                    
                    # Save snapshot
                    snapshot_path = os.path.join(OUT_DIR, f"fall_track{track_id}_{int(trigger_time)}.jpg")
                    cv2.imwrite(snapshot_path, tr.fall_state['recording_buffer'][0])
                    
                    # Send email alert (once per track)
                    if alert_on_fall:
                        email_metrics = {
                            'confidence': conf,
                            'time': trigger_time,
                            'track_id': track_id,
                            # 'aspect_change': metrics['aspect_change'],
                            # 'y_shift': metrics['y_shift']
                        }
                        subject = f"🚨 FALL ALERT - Track {track_id}"
                        body = f"Fall detected in video at {trigger_time:.2f} seconds.\n\nPlease check the attached snapshot and GIF sequence."
                        # Compress GIF before sending (to keep under Gmail 25MB limit)
                        compressed_gif = gif_path.replace(".gif", "_small.gif")
                        compress_gif(gif_path, compressed_gif)

                        send_email_alert(subject, body, snapshot_path, compressed_gif, email_metrics)

                    
                    alerted_tracks.add(track_id)
                    
                    # Record event
                    fall_events.append({
                        'track_id': track_id,
                        'time': trigger_time,
                        'frame': frame_idx,
                        'confidence': conf,
                        'metrics': metrics
                    })
                    
                    # Stop recording but keep fall state until recovery
                    tr.fall_state['recording_started'] = False
            
            # Visualization - SIMPLIFIED: Only OK (green) and FALL (red)
            if visualize:
                last = list(tr.bbox_history)[-1]
                x1, y1, x2, y2 = map(int, last[1:5])
                
                # Simple color coding: Green for OK, Red for FALL
                if tr.fall_state['triggered']:
                    color = (0, 0, 255)  # Red for fall
                    status = "FALL"
                else:
                    color = (0, 255, 0)  # Green for normal
                    status = "OK"
                
                # Draw bounding box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw info text
                label = f"ID:{track_id} | {status}"
                cv2.putText(display_frame, label, (x1, max(0, y1 - 10)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Display frame info
        if visualize:
            info_text = f"Frame: {frame_idx}/{total_frames} | Tracks: {len(tracked)} | Falls: {len(fall_events)}"
            cv2.putText(display_frame, info_text, (10, frame_h - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Progress bar
            progress = int((frame_idx / total_frames) * frame_w) if total_frames > 0 else 0
            cv2.rectangle(display_frame, (0, frame_h - 5), (progress, frame_h), (0, 255, 0), -1)
            
            cv2.imshow("Fall Detection System", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[INFO] User interrupted processing")
                break
        
        # Write frame to output video
        out_video.write(display_frame)
    
    # Cleanup
    cap.release()
    out_video.release()
    cv2.destroyAllWindows()
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Processing Complete")
    print(f"{'='*60}")
    print(f"Total frames processed: {frame_idx}")
    print(f"Processing time: {elapsed:.2f}s")
    print(f"Average FPS: {frame_idx/elapsed:.2f}")
    print(f"Fall events detected: {len(fall_events)}")
    print(f"Output video saved: {output_video_path}")
    
    if fall_events:
        print(f"\nDetected Falls:")
        for i, event in enumerate(fall_events, 1):
            print(f"  {i}. Track {event['track_id']} at {event['time']:.2f}s "
                  f"(confidence: {event['confidence']:.2%})")
    
    return fall_events


# ---------------------
# CLI
# ---------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fall Detection System with YOLO Tracking and Email Alerts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --video fall_video.mp4 --visualize
  python main.py --video fall_video.mp4 --visualize --alert
  
Note: For email alerts, configure EMAIL_SENDER and EMAIL_PASSWORD in the script.
Use Gmail App Password, not your regular password.
        """
    )
    parser.add_argument("--video", type=str, required=True, 
                       help="Input video path")
    parser.add_argument("--visualize", action="store_true", 
                       help="Show GUI visualization")
    parser.add_argument("--alert", action="store_true", 
                       help="Send email alerts on fall detection")
    
    args = parser.parse_args()
    
    # Validation
    if not os.path.exists(args.video):
        print(f"Error: Video file '{args.video}' not found")
        exit(1)
    
    if args.alert:
        if EMAIL_PASSWORD == "wqlo futq vjyn bzvm":  # Check if it's the placeholder
            print("\n  Warning: Please verify your email credentials are configured correctly!")
            response = input("\nContinue with email alerts? (y/n): ")
            if response.lower() != 'y':
                args.alert = False
    
    print(f"\n{'='*60}")
    print("Fall Detection System")
    print(f"{'='*60}")
    print(f"Model: {MODEL_NAME}")
    print(f"Confidence Threshold: {CONF_THRESHOLD:.2%}")
    print(f"Video: {args.video}")
    print(f"Visualization: {'ON' if args.visualize else 'OFF'}")
    print(f"Email Alerts: {'ON' if args.alert else 'OFF'}")
    print(f"GIF Duration: {GIF_DURATION}s")
    print(f"Trigger Frames: {SUSTAINED_FRAMES}")
    print(f"Recovery Frames: {RECOVERY_FRAMES}")
    print(f"Output Directory: {OUT_DIR}")
    print(f"{'='*60}\n")
    
    events = detect_falls_in_video(
        args.video,
        visualize=args.visualize,
        save_clips_on_fall=True,  # Always save clips/GIFs
        alert_on_fall=args.alert
    )
    
    print(f"\n{'='*60}")
    print(f"Summary: {len(events)} fall event(s) detected")
    print(f"{'='*60}\n")