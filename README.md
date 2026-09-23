# SENTRIX — AIoT-Based Smart Surveillance System

## 1. Project Overview

**SENTRIX** is an AIoT-based smart surveillance system designed for real-time security monitoring. It combines computer vision, motion detection, facial recognition, camera tamper detection, event logging, automatic video recording, voice alerts, a web-based dashboard, and automated PDF security reporting.

Unlike a traditional CCTV system that mainly records video for later inspection, SENTRIX processes the live camera feed and identifies important security events in real time.

The system uses a laptop camera as the surveillance device and provides a web dashboard that can be accessed from the laptop or from a smartphone connected to the same local network.

SENTRIX is developed as an academic prototype demonstrating the integration of **Artificial Intelligence, Internet of Things concepts, computer vision, web technologies, database management, and automated security reporting**.

---

# 2. Main Features

## Real-Time Camera Monitoring

* Captures live video using the laptop camera.
* Processes the camera feed continuously.
* Displays the live camera feed through the SENTRIX dashboard.
* Provides camera status information through the dashboard.
* Supports dashboard access from another device connected to the same Wi-Fi network.

---

## Motion Detection

SENTRIX continuously analyzes consecutive camera frames to identify significant movement.

The motion detection pipeline uses:

* Grayscale conversion
* Gaussian blur
* Frame differencing
* Thresholding
* Dilation
* Contour detection
* Minimum contour-area filtering

When sufficient motion is detected:

1. A motion event is generated.
2. The event is stored in SQLite.
3. Video recording is started automatically.
4. Recording continues while motion is detected.
5. Recording continues for a configured period after the last detected motion.
6. The recording is stored in the `recordings/` directory.

---

## Facial Recognition

SENTRIX uses:

* **MTCNN** for face detection
* **FaceNet** for facial feature embeddings
* Similarity-based comparison for recognition

The system supports:

* Registered-person recognition
* Unknown-person detection
* Multiple stored face embeddings for a registered person
* Confidence/similarity information in recorded recognition events

During recognition, the detected face is converted into a numerical FaceNet embedding and compared against stored registered embeddings.

The configured recognition threshold is:

```text
0.60
```

Recognition performance depends on:

* Lighting
* Camera quality
* Face position
* Face angle
* Registration image quality
* Distance from the camera

---

## Person Registration

New people can be registered using the face registration module.

The registration process is:

1. Enter the person's name.
2. Capture multiple face images.
3. Detect the face.
4. Generate FaceNet embeddings.
5. Store the embeddings in `registered_faces/`.
6. Register the person in the SQLite database.

Registration command:

```powershell
python -m app.face.register
```

After registration, SENTRIX can automatically reload registered face information while the system is running.

---

## Unknown-Person Detection

If a detected face does not sufficiently match any registered identity, SENTRIX classifies the person as an **unknown person**.

The system records an unknown-person event in the database.

The dashboard can display the event, and browser-based voice alerts can be used to notify the user.

Unknown-person events are also included in the daily security report.

---

## Camera Tamper Detection

SENTRIX monitors the camera for possible tampering conditions.

The current tamper detection system checks:

### Lens Obstruction

Brightness information is used to identify conditions where the camera lens may be covered or obstructed.

### Camera Blur

Blur variance is analyzed using image sharpness information. A significant reduction in sharpness can indicate that the camera has become blurred or obstructed.

### Camera Movement

Histogram comparison is used to identify significant visual changes that may indicate that the camera position has changed.

Possible tamper conditions include:

```text
CAMERA OK
LENS COVERED
CAMERA BLURRED
CAMERA MOVED
```

Multiple consecutive frames are considered to reduce false detections caused by normal temporary changes.

---

# 3. Video Recording

SENTRIX automatically records video when significant motion is detected.

Recordings are stored in:

```text
recordings/
```

The filename contains the recording timestamp.

Example:

```text
SENTRIX_20260915_220647.mp4
```

Very short recordings are ignored by the database system to avoid storing meaningless video entries.

Each valid recording is associated with information such as:

* Filename
* File path
* Creation time
* Duration

The available recordings can be viewed through the SENTRIX dashboard.

---

# 4. 15-Day Recording Retention

SENTRIX includes an automatic recording-retention mechanism.

Physical video files are retained for approximately **15 days**.

When SENTRIX starts, the system checks existing recording records and removes physical video files older than the configured retention period.

The current retention period is:

```text
15 days
```

### Important behavior

If a recording is older than the retention period:

* The physical `.mp4` file is deleted.
* The corresponding database record is retained.
* The historical recording information can still appear in database-based reports.
* The deleted video itself can no longer be viewed unless it was backed up separately.

The database record is intentionally retained so that SENTRIX can maintain a historical security record even after the physical video has been removed.

---

# 5. Security Event Logging

SENTRIX stores important security events in an SQLite database.

Events can include:

* Motion detected
* Registered person detected
* Unknown person detected
* Camera tampering
* Recording started
* Recording stopped
* System events

Database location:

```text
data/sentrix.db
```

Each event can contain information such as:

* Timestamp
* Event type
* Message
* Person name where applicable
* Recognition confidence where available

The event database forms the basis of the daily security report.

---

# 6. Web Dashboard

The SENTRIX dashboard provides a centralized interface for monitoring the system.

The dashboard can display:

* Live camera feed
* Camera status
* Motion status
* Face detection/recognition status
* Detected person information
* Tamper status
* Recording status
* Recent security events
* Available recordings
* Security reports
* Voice alert controls

The dashboard is implemented using:

* Flask
* HTML
* CSS
* JavaScript

---

# 7. Laptop-to-Phone Monitoring

SENTRIX runs a Flask web server on the laptop.

The dashboard can be accessed locally using:

```text
http://127.0.0.1:5000
```

For access from a smartphone connected to the same Wi-Fi network, use:

```text
http://<LAPTOP-IP>:5000
```

Example:

```text
http://192.168.1.10:5000
```

The actual IP address depends on the laptop's active network connection.

To find the laptop IP address on Windows:

```powershell
ipconfig
```

Look for the **IPv4 Address** of the active Wi-Fi or Ethernet adapter.

### Requirements for mobile access

* Laptop and phone should be connected to the same local network.
* Flask must be running.
* The laptop firewall/network configuration must allow the connection.
* The correct laptop IP address must be used.

---

# 8. Voice Alerts

SENTRIX supports browser-based voice announcements using the **Web Speech API**.

Voice alerts can be used for important security events such as:

* Registered person detection
* Unknown-person detection
* Camera tampering
* Other configured security events

Example announcements can include:

```text
Dharshana detected.
```

or:

```text
Unknown person detected.
```

Voice functionality depends on browser and device support for speech synthesis.

---

# 9. Automated PDF Security Reports

SENTRIX can generate a professional daily security report from the information stored in the database.

Generated reports are stored in:

```text
reports/
```

Example:

```text
reports/SENTRIX_Report_2026-09-23.pdf
```

The report generator is implemented using:

```text
ReportLab
```

The report can include:

* Report date
* Analysis period
* Report generation time
* Overall security state
* Total security events
* Motion activity
* Known-person activity
* Unknown-person activity
* Camera tamper activity
* Recognized people
* Face recognition details
* Video recording information
* Recording duration
* Complete chronological security timeline
* Day reconstruction
* Camera integrity information
* End-of-report system assessment

---

# 10. Daily Security Report Structure

The generated report is designed to reconstruct the security activity recorded during the selected day.

The report contains sections such as:

### 01 — What Happened Today

Provides an overview of the recorded security activity.

### 02 — Daily Intelligence

Summarizes:

* First event
* Latest event
* Motion activity
* Known-person activity
* Unknown activity
* Camera integrity events
* Video evidence
* Total recorded duration

### 03 — Recognized People

Lists registered identities detected during the reporting period.

### 04 — Unknown-Person Activity

Provides chronological information about unknown-person detections.

### 05 — Camera Integrity

Displays recorded tamper conditions and available camera-analysis values.

### 06 — Video Evidence

Lists recordings associated with the selected reporting period.

### 07 — Complete Security Timeline

Provides the recorded events in chronological order.

### 08 — Day Reconstruction

Creates a chronological narrative based on the events stored in the database.

### 09 — Face Recognition Details

Provides recognition timestamps, person names, and available confidence information.

### 10 — End-of-Report System Assessment

Provides a final factual summary of the recorded security activity.

The report only describes information stored by SENTRIX and does not invent events that were not recorded.

---

# 11. System Architecture

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
       └────────────┘   └──────┬─────┘   └────────────┘
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

# 12. Project Structure

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
│   ├── report/
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

### Important directory distinction

The source code for report generation is:

```text
app/report/generator.py
```

Generated PDF reports are stored separately in:

```text
reports/
```

---

# 13. Technologies Used

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

# 14. Installation

## Step 1 — Clone the Repository

```bash
git clone <repository-url>
cd sentrix
```

---

## Step 2 — Create a Virtual Environment

Windows:

```powershell
python -m venv venv
```

Activate the environment:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell execution policy prevents activation, use Command Prompt:

```cmd
venv\Scripts\activate
```

---

## Step 3 — Install Dependencies

```powershell
pip install -r requirements.txt
```

---

# 15. Running SENTRIX

Start the application using:

```powershell
python -m app.main
```

The local dashboard is available at:

```text
http://127.0.0.1:5000
```

For mobile access:

```text
http://<LAPTOP-IP>:5000
```

Example:

```text
http://192.168.1.10:5000
```

---

# 16. Registering a New Person

Before registering a new person, make sure SENTRIX is not currently using the camera.

Run:

```powershell
python -m app.face.register
```

Then:

1. Enter the person's name.
2. Follow the registration instructions.
3. Capture the required face images.
4. Generate FaceNet embeddings.
5. Store the embeddings in `registered_faces/`.
6. Add the person to the SQLite database.
7. Restart SENTRIX if required.

The recognition module periodically reloads registered face information.

---

# 17. Face Recognition

SENTRIX uses FaceNet to convert detected faces into numerical feature embeddings.

The recognition pipeline is:

```text
Camera Frame
     ↓
Face Detection
     ↓
Face Extraction
     ↓
FaceNet Embedding
     ↓
Compare With Registered Embeddings
     ↓
Similarity Score
     ↓
Known Person / Unknown Person
```

The configured recognition threshold is:

```text
0.60
```

Multiple embeddings can be stored for a registered person to improve recognition across different:

* Facial poses
* Lighting conditions
* Camera distances
* Expressions

---

# 18. Motion Detection

The motion detector processes consecutive camera frames using:

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

When sufficient motion is detected:

```text
Motion Detected
      ↓
Event Logged
      ↓
Recording Started
      ↓
Motion Continues
      ↓
Recording Continues
      ↓
No Motion
      ↓
Post-Motion Period
      ↓
Recording Stopped
```

The current post-motion recording period is:

```text
10 seconds
```

---

# 19. Tamper Detection

SENTRIX uses multiple image-analysis techniques to monitor camera integrity.

## Brightness Analysis

Brightness information can help identify a covered or obstructed lens.

## Blur Analysis

Blur variance is used to detect significant loss of image sharpness.

## Histogram Comparison

Histogram correlation is used to compare the current camera view with a reference view and identify significant visual changes.

Possible conditions include:

```text
CAMERA OK
LENS COVERED
CAMERA BLURRED
CAMERA MOVED
```

Tamper information is stored in the SQLite database and can be included in the daily PDF report.

---

# 20. Database

SENTRIX uses SQLite for structured security data storage.

Database location:

```text
data/sentrix.db
```

The database stores information related to:

* Registered persons
* Security events
* Video recordings
* Camera tamper events

The database is initialized automatically when SENTRIX starts.

The database also allows the reporting system to reconstruct historical security activity.

---

# 21. Database and Physical Storage Relationship

SENTRIX separates security records from physical video storage.

```text
              SENTRIX
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
   SQLite Database     Video Files
        │                 │
        │                 │
        ▼                 ▼
 Historical Records   recordings/
```

The database retains information about recordings even after old physical video files are removed by the 15-day retention mechanism.

This allows historical security reports to retain information about past recording events.

---

# 22. PDF Report Generation Flow

The report generation process follows this workflow:

```text
Dashboard
    ↓
Select Report Date
    ↓
Retrieve Database Events
    ↓
Retrieve Recording Records
    ↓
Retrieve Tamper Records
    ↓
Filter Selected Date
    ↓
Analyze Security Activity
    ↓
Generate ReportLab PDF
    ↓
Save PDF
    ↓
reports/SENTRIX_Report_DATE.pdf
```

The report generator is located at:

```text
app/report/generator.py
```

Generated reports are stored at:

```text
reports/
```

---

# 23. Security Status

The daily report determines a descriptive security state based on the recorded events.

Possible states include:

```text
NORMAL MONITORING
```

when no unknown-person or tamper events are recorded.

```text
UNKNOWN ACTIVITY DETECTED
```

when unknown-person events are recorded.

```text
ATTENTION REQUIRED
```

when camera tamper events are recorded.

These states summarize recorded system activity and do not claim that an actual security incident occurred beyond what SENTRIX detected.

---

# 24. Demonstration Workflow

For a project demonstration:

1. Start the SENTRIX application.
2. Open the dashboard on the laptop.
3. Connect a smartphone to the same Wi-Fi network.
4. Open the laptop IP address on the smartphone.
5. Show the live camera feed.
6. Walk in front of the camera.
7. Demonstrate motion detection.
8. Show automatic video recording.
9. Demonstrate recognition of a registered person.
10. Demonstrate unknown-person detection.
11. Cover the camera to demonstrate tamper detection.
12. Demonstrate camera movement if required.
13. Show security events in the dashboard.
14. Open an available recording.
15. Demonstrate voice alerts.
16. Generate the daily PDF security report.
17. Show the chronological security timeline in the report.
18. Demonstrate the dashboard from the smartphone.

---

# 25. Team Responsibilities

## Member 1 — Motion Detection & Video Capture

Responsibilities include:

* Camera capture
* Motion detection
* Video recording
* Motion-triggered recording logic
* Recording retention integration

---

## Member 2 — Face Recognition

Responsibilities include:

* Face dataset preparation
* Face registration
* Face detection
* Face embeddings
* Face recognition
* Registered-person identification
* Unknown-person detection

---

## Member 3 — Dashboard & Integration

Responsibilities include:

* Web dashboard
* System integration
* Mobile dashboard access
* Event display
* System status display
* User interface
* Report-generation integration
* Voice-alert interface

---

# 26. Privacy and Security

SENTRIX processes facial information, therefore privacy should be considered during development and demonstration.

Face images and face embeddings should not be publicly uploaded.

The following directories contain potentially sensitive project data:

```text
face_dataset/
registered_faces/
recordings/
```

These directories should be excluded from public Git repositories where appropriate.

Database files and generated reports may also contain security-related information and should be handled carefully.

When sharing the project with teammates, sensitive files should be transferred through appropriate private methods.

---

# 27. Recommended `.gitignore`

Sensitive and generated data should generally be excluded from GitHub.

Example:

```gitignore
venv/
__pycache__/
*.pyc

face_dataset/
registered_faces/
recordings/

data/*.db
reports/*.pdf

.env
```

---

# 28. Important Notes

* The laptop camera must be available before starting SENTRIX.
* Camera permissions must be enabled.
* Only one application should normally use the camera at a time.
* The phone and laptop should be connected to the same local network for mobile dashboard access.
* The laptop's firewall/network configuration may affect mobile access.
* Face recognition performance depends on lighting, camera quality, face position, distance, and registration quality.
* Voice alerts depend on browser and device support for the Web Speech API.
* Physical recordings are subject to the configured 15-day retention period.
* Database recording information is retained even when an old physical recording file is removed.
* Deleted physical recordings cannot be replayed unless a separate backup exists.
* Do not publicly upload face datasets or face embeddings.
* SENTRIX is an academic prototype and should not be treated as a replacement for professionally certified security systems.

---

# 29. Project Status

SENTRIX Phase 1 includes:

* Real-time camera monitoring
* Motion detection
* Motion-triggered video recording
* Facial recognition
* Registered-person recognition
* Unknown-person detection
* Camera tamper detection
* SQLite event storage
* SQLite recording storage
* Web dashboard
* Mobile dashboard access
* Browser-based voice alerts
* Automated PDF security reports
* Chronological security timeline
* Daily security activity reconstruction
* Camera integrity reporting
* 15-day physical recording retention

The system integrates computer vision, artificial intelligence, web technologies, database management, and automated reporting into a single AIoT-based smart surveillance prototype.

SENTRIX is designed for **academic demonstration, security-system prototyping, and future development**.

---

# 30. Future Development

Potential future improvements include:

* Cloud-based secure storage
* Remote notifications
* Improved multi-camera support
* Advanced person tracking
* Improved tamper classification
* Secure user authentication
* Encrypted storage
* Remote report access
* Improved mobile interface
* Advanced event correlation
* Backup of important recordings
* Configurable recording-retention policies
* More advanced security analytics

These features are potential future extensions and are not necessarily part of the current Phase 1 implementation.

---

# 31. Project Summary

SENTRIX combines multiple security technologies into one integrated system:

```text
             ┌─────────────────────┐
             │     SENTRIX         │
             │ Smart Surveillance  │
             └──────────┬──────────┘
                        │
       ┌────────────────┼────────────────┐
       │                │                │
       ▼                ▼                ▼
   Motion           Face             Tamper
 Detection       Recognition       Detection
       │                │                │
       └────────────────┼────────────────┘
                        │
                        ▼
                Security Events
                        │
             ┌──────────┼──────────┐
             │          │          │
             ▼          ▼          ▼
          Video      SQLite       PDF
        Recording   Database     Reports
             │          │          │
             └──────────┼──────────┘
                        │
                        ▼
                 Web Dashboard
                        │
                        ▼
                  Mobile Access
```

**SENTRIX provides a unified AIoT-based surveillance prototype capable of detecting, recording, recognizing, logging, reporting, and presenting security events through a web-based monitoring platform.**
