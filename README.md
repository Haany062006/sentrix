# SENTRIX — AIoT-Based Smart Surveillance System

## 1. Project Overview

**SENTRIX** is an AIoT-based smart surveillance system that combines real-time video monitoring, motion detection, facial recognition, camera tamper detection, event logging, video recording, voice alerts, and automated security report generation.

Unlike a traditional CCTV system that mainly records video for later inspection, SENTRIX continuously analyzes the camera feed and identifies important security events in real time.

The system uses a laptop camera as the surveillance device and provides a web-based dashboard that can be accessed from a phone connected to the same Wi-Fi network.

---

## 2. Main Features

### Real-Time Camera Monitoring

* Captures live video using the laptop camera.
* Displays the live camera feed through the SENTRIX dashboard.
* Supports access from another device such as a smartphone.

### Motion Detection

* Continuously analyzes video frames for movement.
* Uses grayscale conversion, Gaussian blur, frame differencing, thresholding, dilation, and contour detection.
* Starts video recording when significant motion is detected.
* Continues recording for a short period after the last detected movement.

### Facial Recognition

* Detects faces from the live camera feed.
* Uses **FaceNet** embeddings for facial recognition.
* Supports registered team members.
* Uses multiple face embeddings when available to improve recognition reliability.
* Identifies unknown persons when the detected face does not sufficiently match a registered person.

### Person Registration

New people can be registered using the face registration module.

The registration process:

1. Captures face images.
2. Generates FaceNet embeddings.
3. Saves the embeddings in `registered_faces/`.
4. Automatically registers the person in the SQLite database.

### Camera Tamper Detection

SENTRIX monitors the camera for possible tampering conditions such as:

* Lens obstruction or covering
* Excessive blur
* Sudden camera movement

The tamper detector uses image brightness, blur variance, and histogram comparison to identify abnormal camera conditions.

### Video Recording

When motion is detected, SENTRIX automatically records the event.

Recordings are stored in:

```text
recordings/
```

The dashboard can display available recordings.

### Security Event Logging

Important events are recorded, including:

* Motion detected
* Known person detected
* Unknown person detected
* Camera tampering
* Recording started
* Recording stopped

SENTRIX stores structured security information using SQLite.

Database:

```text
data/sentrix.db
```

### Web Dashboard

The SENTRIX dashboard provides:

* Live camera feed
* Camera status
* Motion status
* Face detection status
* Detected people
* Tamper status
* Recording status
* Recent security events
* Available recordings
* Security report generation
* Voice alert controls

### Laptop-to-Phone Monitoring

The laptop runs the SENTRIX Flask server.

A phone connected to the same Wi-Fi network can access the dashboard using:

```text
http://<LAPTOP-IP>:5000
```

For example:

```text
http://192.168.1.10:5000
```

The exact IP address depends on the laptop's network.

### Voice Alerts

SENTRIX supports browser-based voice alerts using the Web Speech API.

Voice alerts can notify the user about important security events.

Voice functionality depends on browser and device support for speech synthesis.

### PDF Security Reports

SENTRIX can generate daily security reports containing information about detected security events and recordings.

Generated reports are stored in:

```text
reports/
```

---

## 3. System Architecture

```text
                    ┌─────────────────────┐
                    │     Laptop Camera   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Video Capture     │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
      ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
      │    Motion    │ │     Face     │ │    Tamper    │
      │  Detection   │ │ Recognition  │ │  Detection   │
      └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
             │                │                │
             └────────────────┼────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │   SENTRIX Engine    │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
       ┌────────────┐   ┌────────────┐   ┌────────────┐
       │   Video    │   │  SQLite    │   │   PDF      │
       │ Recording  │   │  Database  │   │  Reports   │
       └────────────┘   └────────────┘   └────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Web Dashboard     │
                    └──────────┬──────────┘
                               │
                         Same Wi-Fi
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Mobile Phone      │
                    │     Dashboard       │
                    └─────────────────────┘
```

---

## 4. Project Structure

```text
sentrix/
│
├── app/
│   ├── main.py
│   │
│   ├── camera/
│   │   └── capture.py
│   │
│   ├── dashboard/
│   │   ├── dashboard.py
│   │   └── templates/
│   │       └── index.html
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   └── db.py
│   │
│   ├── face/
│   │   ├── __init__.py
│   │   ├── detector.py
│   │   ├── recognizer.py
│   │   ├── register.py
│   │   └── build_embeddings.py
│   │
│   ├── motion/
│   │   └── detector.py
│   │
│   ├── reports/
│   │   └── generator.py
│   │
│   └── tamper/
│       ├── blur.py
│       ├── brightness.py
│       ├── detector.py
│       └── histogram.py
│
├── data/
│   └── sentrix.db
│
├── face_dataset/
│
├── registered_faces/
│
├── recordings/
│
├── reports/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 5. Technologies Used

| Component            | Technology              |
| -------------------- | ----------------------- |
| Programming Language | Python                  |
| Web Framework        | Flask                   |
| Computer Vision      | OpenCV                  |
| Face Detection       | MTCNN                   |
| Face Recognition     | FaceNet                 |
| Numerical Processing | NumPy / SciPy           |
| Database             | SQLite                  |
| PDF Generation       | ReportLab               |
| Dashboard            | HTML / CSS / JavaScript |
| Voice Alerts         | Web Speech API          |
| Version Control      | Git / GitHub            |

---

## 6. Installation

### Step 1 — Clone the repository

```bash
git clone <repository-url>
cd sentrix
```

### Step 2 — Create a virtual environment

Windows:

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3 — Install dependencies

```powershell
pip install -r requirements.txt
```

---

## 7. Running SENTRIX

Start the system using:

```powershell
python -m app.main
```

The Flask dashboard runs on:

```text
http://127.0.0.1:5000
```

For access from a phone on the same Wi-Fi network:

```text
http://<LAPTOP-IP>:5000
```

Find the laptop's IP address using:

```powershell
ipconfig
```

Look for the **IPv4 Address** of the active network adapter.

---

## 8. Registering a New Person

To register a new person:

1. Stop SENTRIX if it is currently using the camera.
2. Run:

```powershell
python -m app.face.register
```

3. Enter the person's name.
4. Follow the instructions to capture the face images.
5. The system generates the face embeddings automatically.
6. The person's information is added to the SQLite database.

After registration, restart SENTRIX.

The face recognition system automatically reloads registered face data periodically.

---

## 9. Face Recognition

SENTRIX uses FaceNet to convert detected faces into numerical embeddings.

Each embedding contains **512 numerical values** representing facial features.

During recognition, the live face embedding is compared with the stored registered embeddings.

A similarity score is calculated and compared with the configured recognition threshold.

The current recognition threshold is:

```text
0.60
```

A sufficiently high similarity score is treated as a known person; otherwise, the person is classified as unknown.

Multiple embeddings can be stored for a person to improve recognition across different poses and lighting conditions.

---

## 10. Motion Detection

The motion detector processes consecutive frames using:

1. Grayscale conversion
2. Gaussian blur
3. Absolute frame difference
4. Thresholding
5. Dilation
6. Contour detection
7. Minimum contour-area filtering

The current minimum motion contour area is:

```text
500 pixels
```

When motion is detected, SENTRIX starts recording.

The system continues recording for a configured period after the last detected motion.

---

## 11. Tamper Detection

The camera tamper system monitors three main image characteristics:

### Brightness

Used to identify conditions such as a covered or obstructed lens.

### Blur Variance

A low variance of the Laplacian indicates that the image may have become significantly blurred.

### Histogram Correlation

Compares the current frame with a reference frame to identify significant visual changes that may indicate camera movement.

Tamper detection also uses multiple consecutive frames to reduce false alarms caused by normal movement.

Possible statuses include:

```text
CAMERA OK
LENS COVERED
CAMERA BLURRED
CAMERA MOVED
```

---

## 12. Database

SENTRIX uses SQLite to store structured security information.

Database file:

```text
data/sentrix.db
```

The database contains information related to:

* Registered persons
* Security events
* Recordings
* Tamper events

The database is created automatically when SENTRIX starts.

---

## 13. Recordings

Motion-triggered recordings are stored in:

```text
recordings/
```

Each recording contains a timestamp in its filename.

Example:

```text
SENTRIX_20260915_220647.mp4
```

Very short recordings are ignored by the database system to prevent meaningless entries.

---

## 14. PDF Reports

Daily reports can be generated from the SENTRIX dashboard.

Reports contain security information such as:

* Report date
* Security events
* Motion activity
* Unknown-person detections
* Tamper events
* Recording information

Generated reports are stored in:

```text
reports/
```

---

## 15. Team Responsibilities

### Member 1 — Motion Detection & Video Capture

Responsible for:

* Camera capture
* Motion detection
* Video recording
* Motion-triggered recording logic

### Member 2 — Face Recognition

Responsible for:

* Face dataset
* Face registration
* Face detection
* Face embeddings
* Face recognition

### Member 3 — Dashboard & Integration

Responsible for:

* Web dashboard
* System integration
* Mobile dashboard access
* Event display
* System status display
* User interface

The final system integrates all three components into a single SENTRIX application.

---

## 16. Privacy and Security

Face images and face embeddings are biometric information and should not be publicly uploaded.

The following directories are intentionally excluded from GitHub:

```text
face_dataset/
registered_faces/
recordings/
```

Database files and generated report files are also excluded where appropriate.

When sharing the project with teammates, these files may be shared privately when required for the project demonstration.

---

## 17. Demonstration Workflow

For a project demonstration:

1. Start the SENTRIX application.
2. Open the dashboard on the laptop.
3. Connect the phone to the same Wi-Fi network.
4. Open the laptop's IP address on the phone.
5. Show the live camera feed.
6. Walk in front of the camera to trigger motion detection.
7. Show automatic video recording.
8. Show a registered person's recognition.
9. Show an unknown-person detection.
10. Cover the camera to demonstrate tamper detection.
11. Show the event history.
12. Show the recorded video.
13. Generate a daily PDF security report.
14. Demonstrate the dashboard from the phone.

---

## 18. Important Notes

* The laptop camera must be available before starting SENTRIX.
* The phone and laptop should be connected to the same local network for mobile dashboard access.
* Camera permissions must be enabled.
* Face recognition performance depends on lighting, camera quality, face position, and registration quality.
* Voice alerts depend on browser/device speech-synthesis support.
* Do not publicly upload face datasets or face embeddings.

---

## 19. Project Status

SENTRIX Phase 1 includes:

* Real-time camera monitoring
* Motion detection
* Motion-triggered recording
* Facial recognition
* Unknown-person detection
* Camera tamper detection
* SQLite event storage
* Web dashboard
* Mobile dashboard access
* Voice alerts
* Automated PDF security reports

The system is designed as an integrated AIoT smart surveillance prototype for academic demonstration and further development.
