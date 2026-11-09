"""PC-side data-collection GUI.

Same recording pipeline as Collection/pc_agent.py (which still works on its own),
with three additions:
  - the camera is optional (tick box) and losing it never stops the sensor recording
  - inbound labels are marked live: press SPACE when someone enters the zone and
    again when the zone is empty -> saved straight to labels.csv
  - split, recording name, ROCK address and camera are chosen on screen

Run from the project root:
    python -m Collection.pc_collector_gui --host <ROCK_IP> --camera 1
(all arguments are optional; they only pre-fill the form)
"""
import argparse
import csv
import json
import shutil
import socket
import queue
import threading
import time
from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox

from Collection.pc_agent import (
    CAMERA_STALL_SECONDS,
    DATASET_ROOT,
    SENSOR_STALL_SECONDS,
    START_DELAY_SECONDS,
    VIDEO_FILE_FPS,
    measureClockOffset,
    openCamera,
    receiveLoop,
    recordVideo,
    sendCommand,
    writeMetadata,
)
from Collection.protocol import (
    CMD_START_AT,
    CMD_STOP,
    DEFAULT_PORT,
    KEY_PIR_BASELINE,
    KEY_PIR_SAMPLE_RATE,
    KEY_ROCK_START_TIME,
    KEY_STATUS,
    KEY_THERMAL_BACKGROUND,
    KEY_THERMAL_FRAME_SHAPE,
    KEY_THERMAL_RATE,
    KEY_THERMAL_SIGMA,
    STATUS_READY,
)

SPLITS = ("training", "validation", "testing")

# Connection Timing
CONNECT_TIMEOUT_SECONDS = 2.0
CONNECT_RETRY_SECONDS = 1.0
# ROCK Calibration Takes ~10 s (40 Frames At 4 Hz), So Allow Plenty Of Margin
READY_TIMEOUT_SECONDS = 60.0
CAMERA_READY_TIMEOUT_SECONDS = 20.0

# Ignore Key Auto-Repeat / Double Taps Closer Together Than This
KEY_DEBOUNCE_SECONDS = 0.3

# How Often The GUI Checks Streams And Refreshes Counters
MONITOR_PERIOD_MS = 200

LABEL_FILE = "labels.csv"

# GUI Colours
EMPTY_COLOUR = "#2e3440"
PERSON_COLOUR = "#c0392b"
READY_COLOUR = "#27ae60"


# Same Folder-Name Rules As pc_agent.chooseRun, Returns An Error Message Or None
def recordingNameError(runName):
    if not runName or runName in (".", ".."):
        return "Recording name cannot be empty."
    if ("/" in runName or "\\" in runName or
            any(char in runName for char in '<>:"|?*') or
            any(ord(char) < 32 for char in runName) or
            runName.endswith((" ", "."))):
        return "Use a simple folder name without path separators or invalid filename characters."
    return None


# Toggle Presses -> (start, end) Intervals. An Unclosed Interval Ends At endTime.
def pressesToIntervals(presses, endTime):
    intervals = []
    for i in range(0, len(presses), 2):
        start = max(0.0, presses[i])
        end = presses[i + 1] if i + 1 < len(presses) else endTime
        if end > start:
            intervals.append((start, end))
    return intervals


# Same Format As The Playback Tool: start_s,end_s,human (Header Only = Nobody Present)
def writeLabels(path, intervals):
    with open(path, "w", newline="") as labelFile:
        writer = csv.writer(labelFile)
        writer.writerow(["start_s", "end_s", "human"])
        for start, end in intervals:
            writer.writerow([start, end, 1])


class RecordingSession:
    """Everything belonging to one recording (one ROCK connection = one calibration)."""

    def __init__(self, split, runName, runDir, host, port, useCamera, cameraIndex):
        self.split = split
        self.runName = runName
        self.runDir = runDir
        self.host = host
        self.port = port
        self.cameraRequested = useCamera
        self.useCamera = useCamera
        self.cameraIndex = cameraIndex

        self.pirPath = runDir / "pir.csv"
        self.thermalPath = runDir / "thermopile.csv"
        self.videoPath = runDir / "video.mp4"
        self.videoFramesPath = runDir / "video_frames.csv"
        self.metadataPath = runDir / "metadata.json"
        self.labelsPath = runDir / LABEL_FILE

        self.sock = None
        self.readyPayload = None
        self.pirFile = None
        self.thermalFile = None
        self.clockSync = None
        self.startPcTime = None
        self.startRockTime = None
        self.recordingStartMonotonic = None
        self.started = False

        self.cameraError = None
        self.cameraLost = False

        # Live Label Key Presses (Seconds Since T0), Alternating Enter / Leave
        self.presses = []
        self.lastPressMonotonic = 0.0

        self.cancelEvent = threading.Event()
        self.abortLock = threading.Lock()
        self.aborted = False

        self.state = {
            "pir_rows": 0,
            "thermal_rows": 0,
            "recording_status": None,
            "recordingEvent": threading.Event(),
            "stoppedEvent": threading.Event(),
            "sync_reply": None,
            "syncEvent": threading.Event(),
            "clock_offset_s": None,
            "start_pc_time": None,
            "connection_error": None,
            "last_pir_monotonic": None,
            "last_thermal_monotonic": None,
        }
        self.receiveStopEvent = threading.Event()
        self.receiveThread = None

        self.videoStopEvent = threading.Event()
        self.cameraReadyEvent = threading.Event()
        self.startConfiguredEvent = threading.Event()
        self.videoInfo = {}
        self.videoThread = None

    @property
    def personInside(self):
        return len(self.presses) % 2 == 1


class CollectorApp:
    def __init__(self, root, args):
        self.root = root
        self.session = None
        self.mode = "idle"  # idle / connecting / ready / starting / recording / stopping
        self.closeWhenDone = False
        self.uiQueue = queue.Queue()

        root.title("PIR + Thermopile Data Collector")
        root.minsize(760, 640)

        self.splitVar = tk.StringVar(value="training")
        self.nameVar = tk.StringVar()
        self.hostVar = tk.StringVar(value=args.host or "")
        self.portVar = tk.StringVar(value=str(args.port))
        self.useCameraVar = tk.BooleanVar(value=not args.no_camera)
        self.cameraIndexVar = tk.StringVar(value=str(args.camera))

        self.statusVar = tk.StringVar(value="Fill in the recording details, then press Connect.")
        self.elapsedVar = tk.StringVar(value="--")
        self.pirVar = tk.StringVar(value="0")
        self.thermalVar = tk.StringVar(value="0")
        self.cameraStateVar = tk.StringVar(value="--")
        self.cameraTestVar = tk.StringVar(value="")

        self.buildUi()
        self.setMode("idle")

        # SPACE Toggles The Inbound Label Anywhere In The Window
        root.bind_all("<KeyPress-space>", self.onSpace)
        root.protocol("WM_DELETE_WINDOW", self.onClose)
        root.after(50, self.pollUiQueue)

    # ------------------------------------------------------------------ UI
    def buildUi(self):
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=1)

        # Recording Details
        details = ttk.LabelFrame(outer, text="Recording", padding=8)
        details.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        ttk.Label(details, text="Data type").grid(row=0, column=0, sticky="w")
        self.splitBox = ttk.Combobox(details, textvariable=self.splitVar, values=SPLITS, state="readonly", width=14)
        self.splitBox.grid(row=0, column=1, sticky="ew", pady=2)
        ttk.Label(details, text="Recording name").grid(row=1, column=0, sticky="w")
        self.nameEntry = ttk.Entry(details, textvariable=self.nameVar, width=28)
        self.nameEntry.grid(row=1, column=1, sticky="ew", pady=2)
        ttk.Label(details, text="Saved to Datasets/<data type>/<name>", foreground="#666").grid(
            row=2, column=0, columnspan=2, sticky="w")
        details.columnconfigure(1, weight=1)

        # ROCK And Camera
        devices = ttk.LabelFrame(outer, text="Devices", padding=8)
        devices.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        ttk.Label(devices, text="ROCK IP").grid(row=0, column=0, sticky="w")
        self.hostEntry = ttk.Entry(devices, textvariable=self.hostVar, width=16)
        self.hostEntry.grid(row=0, column=1, sticky="ew", pady=2)
        ttk.Label(devices, text="Port").grid(row=0, column=2, sticky="w", padx=(6, 0))
        self.portEntry = ttk.Entry(devices, textvariable=self.portVar, width=6)
        self.portEntry.grid(row=0, column=3, sticky="w", pady=2)

        self.cameraCheck = ttk.Checkbutton(devices, text="Record camera", variable=self.useCameraVar,
                                           takefocus=False)
        self.cameraCheck.grid(row=1, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Label(devices, text="Index").grid(row=1, column=2, sticky="w", padx=(6, 0))
        self.cameraSpin = ttk.Spinbox(devices, from_=0, to=9, textvariable=self.cameraIndexVar, width=4)
        self.cameraSpin.grid(row=1, column=3, sticky="w", pady=2)
        self.testCameraButton = ttk.Button(devices, text="Test camera", command=self.onTestCamera,
                                           takefocus=False)
        self.testCameraButton.grid(row=2, column=0, sticky="w", pady=2)
        ttk.Label(devices, textvariable=self.cameraTestVar, foreground="#555").grid(
            row=2, column=1, columnspan=3, sticky="w")
        devices.columnconfigure(1, weight=1)

        # Control Buttons
        controls = ttk.Frame(outer, padding=(0, 8))
        controls.grid(row=1, column=0, columnspan=2, sticky="ew")
        for column in range(4):
            controls.columnconfigure(column, weight=1)
        self.connectButton = ttk.Button(controls, text="1. Connect + calibrate", command=self.onConnect,
                                        takefocus=False)
        self.startButton = ttk.Button(controls, text="2. Start recording", command=self.onStart,
                                      takefocus=False)
        self.stopButton = ttk.Button(controls, text="3. Stop + save", command=self.onStop,
                                     takefocus=False)
        self.cancelButton = ttk.Button(controls, text="Cancel", command=self.onCancel, takefocus=False)
        self.connectButton.grid(row=0, column=0, sticky="ew", padx=2)
        self.startButton.grid(row=0, column=1, sticky="ew", padx=2)
        self.stopButton.grid(row=0, column=2, sticky="ew", padx=2)
        self.cancelButton.grid(row=0, column=3, sticky="ew", padx=2)

        # Status
        status = ttk.LabelFrame(outer, text="Status", padding=8)
        status.grid(row=2, column=0, columnspan=2, sticky="ew")
        self.statusLabel = tk.Label(status, textvariable=self.statusVar, anchor="w", justify="left",
                                    wraplength=700, font=("Segoe UI", 11, "bold"))
        self.statusLabel.grid(row=0, column=0, columnspan=8, sticky="ew", pady=(0, 6))
        for column, (text, var) in enumerate((("Elapsed", self.elapsedVar), ("PIR samples", self.pirVar),
                                              ("Thermal frames", self.thermalVar),
                                              ("Camera", self.cameraStateVar))):
            ttk.Label(status, text=text + ":").grid(row=1, column=2 * column, sticky="w")
            ttk.Label(status, textvariable=var, width=14).grid(row=1, column=2 * column + 1, sticky="w")

        # Live Labelling
        labelling = ttk.LabelFrame(outer, text="Live inbound labels (SPACE = person enters / zone empty again)",
                                   padding=8)
        labelling.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=(8, 0))
        outer.rowconfigure(3, weight=1)
        labelling.columnconfigure(0, weight=3)
        labelling.columnconfigure(1, weight=2)
        labelling.rowconfigure(1, weight=1)

        self.indicator = tk.Label(labelling, text="NOT RECORDING", font=("Segoe UI", 26, "bold"),
                                  fg="white", bg=EMPTY_COLOUR, height=2)
        self.indicator.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        # Clickable Alternative To SPACE; takefocus=False So SPACE Never "Presses" A Focused Button
        self.toggleButton = tk.Button(labelling, text="Enter / Leave\n(SPACE)", font=("Segoe UI", 14, "bold"),
                                      command=self.toggleLabel, takefocus=False)
        self.toggleButton.grid(row=0, column=1, sticky="nsew")

        listFrame = ttk.Frame(labelling)
        listFrame.grid(row=1, column=0, sticky="nsew", padx=(0, 6), pady=(8, 0))
        listFrame.columnconfigure(0, weight=1)
        listFrame.rowconfigure(0, weight=1)
        self.intervalList = tk.Listbox(listFrame, height=6, takefocus=False, font=("Consolas", 10))
        self.intervalList.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(listFrame, orient="vertical", command=self.intervalList.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.intervalList.configure(yscrollcommand=scroll.set)

        sideFrame = ttk.Frame(labelling)
        sideFrame.grid(row=1, column=1, sticky="nsew", pady=(8, 0))
        self.undoButton = ttk.Button(sideFrame, text="Undo last press", command=self.undoPress, takefocus=False)
        self.undoButton.pack(fill="x")
        ttk.Label(sideFrame, text=("Times are seconds since T0 (same clock as\n"
                                   "pir.csv / thermopile.csv). Refine later in\n"
                                   "Tools.synchronised_playback if needed."),
                  foreground="#666", justify="left").pack(anchor="w", pady=(6, 0))

        # Log
        logFrame = ttk.LabelFrame(outer, text="Log", padding=4)
        logFrame.grid(row=4, column=0, columnspan=2, sticky="nsew", pady=(8, 0))
        logFrame.columnconfigure(0, weight=1)
        self.logText = tk.Text(logFrame, height=7, state="disabled", takefocus=False, font=("Consolas", 9))
        self.logText.grid(row=0, column=0, sticky="nsew")

    def log(self, message):
        self.logText.configure(state="normal")
        self.logText.insert("end", time.strftime("%H:%M:%S  ") + message + "\n")
        self.logText.see("end")
        self.logText.configure(state="disabled")

    def setStatus(self, message, colour="black"):
        self.statusVar.set(message)
        self.statusLabel.configure(fg=colour)

    def setMode(self, mode):
        self.mode = mode
        editable = mode == "idle"
        entryState = "normal" if editable else "disabled"
        self.splitBox.configure(state="readonly" if editable else "disabled")
        for widget in (self.nameEntry, self.hostEntry, self.portEntry, self.cameraSpin):
            widget.configure(state=entryState)
        self.cameraCheck.configure(state=entryState)
        self.testCameraButton.configure(state=entryState)
        self.connectButton.configure(state="normal" if mode == "idle" else "disabled")
        self.startButton.configure(state="normal" if mode == "ready" else "disabled")
        self.stopButton.configure(state="normal" if mode == "recording" else "disabled")
        self.cancelButton.configure(state="normal" if mode in ("connecting", "ready") else "disabled")
        recording = mode == "recording"
        self.toggleButton.configure(state="normal" if recording else "disabled")
        self.undoButton.configure(state="normal" if recording else "disabled")
        self.refreshLabelDisplay()

    def refreshLabelDisplay(self):
        session = self.session
        if self.mode != "recording" or session is None:
            text = {"ready": "READY", "starting": "STARTING..."}.get(self.mode, "NOT RECORDING")
            self.indicator.configure(text=text, bg=READY_COLOUR if self.mode == "ready" else EMPTY_COLOUR)
        elif session.personInside:
            self.indicator.configure(text="PERSON INBOUND", bg=PERSON_COLOUR)
        else:
            self.indicator.configure(text="ZONE EMPTY", bg=EMPTY_COLOUR)

        self.intervalList.delete(0, "end")
        if session is None:
            return
        presses = session.presses
        for index in range(0, len(presses), 2):
            start = presses[index]
            end = f"{presses[index + 1]:8.2f} s" if index + 1 < len(presses) else "   (open)"
            self.intervalList.insert("end", f"{index // 2 + 1:3d}:  {start:8.2f} s  ->  {end}")
        self.intervalList.see("end")

    # Worker Threads Never Touch Tk Directly; They Queue Calls For The GUI Thread
    def post(self, function, *args):
        self.uiQueue.put((function, args))

    def pollUiQueue(self):
        while True:
            try:
                function, args = self.uiQueue.get_nowait()
            except queue.Empty:
                break
            function(*args)
        self.root.after(50, self.pollUiQueue)

    # ------------------------------------------------------------- Camera test
    def onTestCamera(self):
        try:
            index = int(self.cameraIndexVar.get())
        except ValueError:
            self.cameraTestVar.set("Index must be a number")
            return
        self.testCameraButton.configure(state="disabled")
        self.cameraTestVar.set(f"Testing camera {index}...")

        def worker():
            capture = openCamera(index)
            result = f"Camera {index}: not found"
            if capture.isOpened():
                deadline = time.monotonic() + 5.0
                result = f"Camera {index}: opened but no frames"
                while time.monotonic() < deadline:
                    ok, frame = capture.read()
                    if ok and frame is not None and frame.size:
                        result = f"Camera {index}: OK ({frame.shape[1]}x{frame.shape[0]})"
                        break
                    time.sleep(0.05)
            capture.release()
            self.post(self.onCameraTested, result)

        threading.Thread(target=worker, daemon=True).start()

    def onCameraTested(self, result):
        self.cameraTestVar.set(result)
        self.log(result)
        if self.mode == "idle":
            self.testCameraButton.configure(state="normal")

    # ------------------------------------------------------------------ Connect
    def onConnect(self):
        split = self.splitVar.get()
        runName = self.nameVar.get().strip()
        host = self.hostVar.get().strip()

        error = recordingNameError(runName)
        if error:
            messagebox.showerror("Recording name", error)
            return
        if not host:
            messagebox.showerror("ROCK IP", "Enter the ROCK 4C+ IP address.")
            return
        try:
            port = int(self.portVar.get())
            cameraIndex = int(self.cameraIndexVar.get())
        except ValueError:
            messagebox.showerror("Settings", "Port and camera index must be whole numbers.")
            return

        splitDir = DATASET_ROOT / split
        splitDir.mkdir(parents=True, exist_ok=True)
        runDir = splitDir / runName
        try:
            runDir.mkdir(exist_ok=False)
        except FileExistsError:
            messagebox.showerror("Recording name",
                                 f"{split}/{runName} already exists. Recordings are never overwritten - "
                                 "choose a new name.")
            return

        self.session = RecordingSession(split, runName, runDir, host, port, self.useCameraVar.get(), cameraIndex)
        self.elapsedVar.set("--")
        self.pirVar.set("0")
        self.thermalVar.set("0")
        self.cameraStateVar.set("warming up" if self.session.useCamera else "not used")
        self.setMode("connecting")
        self.log(f"New recording {split}/{runName} (camera {'index ' + str(cameraIndex) if self.session.useCamera else 'off'})")
        self.setStatus(f"Connecting to ROCK at {host}:{port}...", "#1f4e79")
        threading.Thread(target=self.connectWorker, args=(self.session,), daemon=True).start()

    def connectWorker(self, session):
        try:
            # Keep Trying Until The ROCK Agent Answers Or The User Cancels
            attempt = 0
            while not session.cancelEvent.is_set():
                attempt += 1
                candidate = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                candidate.settimeout(CONNECT_TIMEOUT_SECONDS)
                try:
                    candidate.connect((session.host, session.port))
                    session.sock = candidate
                    break
                except OSError:
                    candidate.close()
                    self.post(self.setStatus, f"ROCK not reachable at {session.host}:{session.port} "
                                              f"(attempt {attempt}) - is rock_agent running? Retrying...", "#b9770e")
                    session.cancelEvent.wait(CONNECT_RETRY_SECONDS)
            if session.sock is None:
                self.abortSession(session)
                return

            self.post(self.log, "Connected to ROCK, sensors calibrating")
            self.post(self.setStatus, "Connected. ROCK is calibrating - KEEP THE AREA EMPTY (~10 s)...", "#b9770e")
            session.readyPayload = self.waitForReady(session)
            if session.readyPayload is None:
                self.abortSession(session)
                return
            # receiveLoop Needs A Blocking Socket (A Timeout Would Look Like A Lost Connection)
            session.sock.settimeout(None)

            self.openSensorFiles(session)
            session.receiveThread = threading.Thread(
                target=receiveLoop,
                args=(session.sock, session.pirWriter, session.thermalWriter, session.receiveStopEvent, session.state),
                daemon=True,
            )
            session.receiveThread.start()

            if session.useCamera:
                self.post(self.setStatus, f"Sensors calibrated. Warming up camera {session.cameraIndex}...", "#1f4e79")
                self.startCamera(session)

            if session.cancelEvent.is_set():
                self.abortSession(session)
                return
            self.post(self.onReady, session)
        except Exception as exc:
            # Closing The Socket On Cancel Raises Here; That Is Not An Error
            error = None if session.cancelEvent.is_set() else f"{type(exc).__name__}: {exc}"
            self.abortSession(session, error)

    # Like pc_agent.waitForReady, But Can Be Cancelled And Times Out
    def waitForReady(self, session):
        sock = session.sock
        sock.settimeout(0.5)
        buffer = ""
        deadline = time.monotonic() + READY_TIMEOUT_SECONDS
        while not session.cancelEvent.is_set():
            if time.monotonic() > deadline:
                raise TimeoutError(f"ROCK did not finish calibrating within {READY_TIMEOUT_SECONDS:.0f} s")
            try:
                chunk = sock.recv(4096)
            except socket.timeout:
                continue
            if not chunk:
                raise ConnectionError("ROCK closed the connection before sending ready status")
            buffer += chunk.decode("utf-8")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if payload.get(KEY_STATUS) == STATUS_READY:
                    return payload
        return None

    def openSensorFiles(self, session):
        frameShape = session.readyPayload.get(KEY_THERMAL_FRAME_SHAPE, [24, 32])
        frameLength = int(frameShape[0]) * int(frameShape[1])
        session.frameShape = frameShape

        session.pirFile = open(session.pirPath, "w", newline="", buffering=1)
        session.thermalFile = open(session.thermalPath, "w", newline="", buffering=1)
        session.pirWriter = csv.writer(session.pirFile)
        session.thermalWriter = csv.writer(session.thermalFile)
        session.pirWriter.writerow(["rock_timestamp", "pc_timestamp", "time_s", "seq", "voltage"])
        session.thermalWriter.writerow(
            ["rock_timestamp", "pc_timestamp", "time_s", "seq"]
            + [f"pixel_{i}" for i in range(frameLength)]
        )

    # Camera Failure Here Is Not Fatal: The Recording Continues Sensors-Only
    def startCamera(self, session):
        session.videoThread = threading.Thread(
            target=recordVideo,
            args=(
                session.cameraIndex,
                session.videoPath,
                session.videoFramesPath,
                session.videoStopEvent,
                session.cameraReadyEvent,
                session.startConfiguredEvent,
                session.videoInfo,
            ),
            daemon=True,
        )
        session.videoThread.start()

        deadline = time.monotonic() + CAMERA_READY_TIMEOUT_SECONDS
        while not session.cameraReadyEvent.is_set() and not session.cancelEvent.is_set():
            if time.monotonic() > deadline:
                break
            session.cameraReadyEvent.wait(0.25)

        if session.cancelEvent.is_set():
            return
        if not session.cameraReadyEvent.is_set() or session.videoInfo.get("error"):
            session.cameraError = session.videoInfo.get("error") or "Camera did not become ready"
            session.useCamera = False
            session.videoStopEvent.set()
            self.post(self.onCameraUnavailable, session)

    def onCameraUnavailable(self, session):
        self.cameraStateVar.set("unavailable")
        self.log(f"WARNING: {session.cameraError} - continuing WITHOUT video")

    def onReady(self, session):
        if session is not self.session:
            return
        if session.cancelEvent.is_set():
            threading.Thread(target=self.abortSession, args=(session,), daemon=True).start()
            return
        self.setMode("ready")
        if session.useCamera:
            self.cameraStateVar.set("ready")
        cameraNote = "" if session.useCamera else " (no camera)"
        self.setStatus(f"Ready{cameraNote}. Press 'Start recording', then SPACE each time someone "
                       "enters or the zone becomes empty.", READY_COLOUR)
        self.log("Sensors calibrated and ready")

    # ------------------------------------------------------------------- Cancel
    def onCancel(self):
        session = self.session
        if session is None or self.mode not in ("connecting", "ready"):
            return
        session.cancelEvent.set()
        self.setStatus("Cancelling...", "#b9770e")
        if self.mode == "ready":
            threading.Thread(target=self.abortSession, args=(session,), daemon=True).start()
        else:
            # Unblocks A Pending recv/connect In The Connect Worker
            self.closeSocket(session)

    @staticmethod
    def closeSocket(session):
        if session.sock is None:
            return
        try:
            session.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            session.sock.close()
        except OSError:
            pass

    # Tears Down A Session That Never Started Recording And Deletes Its Empty Folder
    def abortSession(self, session, error=None):
        with session.abortLock:
            if session.aborted:
                return
            session.aborted = True
        session.cancelEvent.set()
        session.videoStopEvent.set()
        session.receiveStopEvent.set()
        self.closeSocket(session)
        if session.videoThread is not None:
            session.videoThread.join(timeout=5.0)
        if session.receiveThread is not None:
            session.receiveThread.join(timeout=2.0)
        for handle in (session.pirFile, session.thermalFile):
            if handle is not None:
                handle.close()
        if not session.started:
            shutil.rmtree(session.runDir, ignore_errors=True)
        self.post(self.onAborted, session, error)

    def onAborted(self, session, error):
        if session is self.session:
            self.session = None
        self.setMode("idle")
        self.cameraStateVar.set("--")
        if error:
            self.setStatus(f"Recording not started: {error}", PERSON_COLOUR)
            self.log(f"ERROR: {error}")
        else:
            self.setStatus("Cancelled. Nothing was saved.", "black")
            self.log("Cancelled - recording folder removed")
        if self.closeWhenDone:
            self.root.destroy()

    # -------------------------------------------------------------------- Start
    def onStart(self):
        session = self.session
        if session is None or self.mode != "ready":
            return
        self.setMode("starting")
        self.setStatus("Synchronising clocks and scheduling T0...", "#1f4e79")
        threading.Thread(target=self.startWorker, args=(session,), daemon=True).start()

    def startWorker(self, session):
        try:
            if not session.receiveThread.is_alive():
                raise ConnectionError(session.state.get("connection_error") or "ROCK connection lost")

            # One Simple Clock Sync Immediately Before Recording (As In pc_agent)
            session.clockSync = measureClockOffset(session.sock, session.state)
            if session.clockSync is None:
                raise RuntimeError("ROCK did not answer clock-sync messages")
            clockOffset = float(session.clockSync["offset_s"])
            session.state["clock_offset_s"] = clockOffset

            # Give Both Machines Enough Time To Receive T0 Before It Occurs
            session.startPcTime = time.time() + START_DELAY_SECONDS
            session.startRockTime = session.startPcTime - clockOffset
            session.state["start_pc_time"] = session.startPcTime
            session.videoInfo["start_pc_time"] = session.startPcTime

            # Provisional Metadata Before Acquisition So A Crash Still Leaves Calibration On Disk
            writeMetadata(session.metadataPath, {
                "split": session.split, "run_name": session.runName,
                "pir_baseline_v": session.readyPayload.get(KEY_PIR_BASELINE),
                "pir_sample_rate_hz": session.readyPayload.get(KEY_PIR_SAMPLE_RATE),
                "thermal_rate_hz": session.readyPayload.get(KEY_THERMAL_RATE),
                "thermal_frame_shape": session.frameShape,
                "thermal_background": session.readyPayload.get(KEY_THERMAL_BACKGROUND),
                "thermal_sigma": session.readyPayload.get(KEY_THERMAL_SIGMA),
                "common_start_pc_time": session.startPcTime,
                "common_start_rock_time": session.startRockTime,
                "rock_to_pc_offset_s": clockOffset,
                "camera_used": session.useCamera,
                "recording_complete": False,
                "note": "Provisional metadata; CSV rows already on disk are recoverable",
            })
            sendCommand(session.sock, CMD_START_AT, **{KEY_ROCK_START_TIME: session.startRockTime})
            session.started = True
            session.startConfiguredEvent.set()

            self.post(self.setStatus, f"Recording starts in {START_DELAY_SECONDS:.1f} s...", "#1f4e79")
            if not session.state["recordingEvent"].wait(timeout=START_DELAY_SECONDS + 3.0):
                raise RuntimeError("ROCK did not start recording at the scheduled time")
            # recordingEvent Arrives Just After T0
            session.recordingStartMonotonic = time.monotonic() - max(0.0, time.time() - session.startPcTime)
            self.post(self.onRecording, session, clockOffset)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            if session.started:
                # Data May Already Be On Disk: Stop Properly And Keep It
                self.post(self.stopRecording, f"start failed: {message}")
            else:
                self.abortSession(session, message)

    def onRecording(self, session, clockOffset):
        if session is not self.session:
            return
        self.setMode("recording")
        self.root.focus_set()
        self.log(f"Recording {session.split}/{session.runName} "
                 f"(PC = ROCK {clockOffset:+.4f} s, round trip {session.clockSync['round_trip_s'] * 1000:.1f} ms)")
        self.setStatus("RECORDING - press SPACE when someone enters, SPACE again when the zone is empty.",
                       PERSON_COLOUR)
        self.root.after(MONITOR_PERIOD_MS, self.monitor)

    # ------------------------------------------------------------------ Labels
    def onSpace(self, event):
        if self.mode == "recording":
            self.toggleLabel()
            return "break"
        return None

    def toggleLabel(self):
        session = self.session
        if session is None or self.mode != "recording":
            return
        pressTime = time.time() - session.startPcTime
        nowMonotonic = time.monotonic()
        if nowMonotonic - session.lastPressMonotonic < KEY_DEBOUNCE_SECONDS:
            return
        session.lastPressMonotonic = nowMonotonic
        session.presses.append(pressTime)
        action = "ENTERED" if session.personInside else "zone EMPTY"
        self.log(f"t = {pressTime:7.2f} s  {action}")
        self.refreshLabelDisplay()

    def undoPress(self):
        session = self.session
        if session is None or self.mode != "recording" or not session.presses:
            return
        removed = session.presses.pop()
        self.log(f"Undo: removed press at t = {removed:.2f} s")
        self.refreshLabelDisplay()

    # ----------------------------------------------------------------- Monitor
    def monitor(self):
        session = self.session
        if session is None or self.mode != "recording":
            return
        now = time.monotonic()
        elapsed = time.time() - session.startPcTime
        self.elapsedVar.set(f"{int(elapsed // 60):02d}:{elapsed % 60:05.2f}")
        self.pirVar.set(str(session.state["pir_rows"]))
        self.thermalVar.set(str(session.state["thermal_rows"]))

        # Lost ROCK Or Silent Sensors Stop The Recording (Same Rules As pc_agent)
        if not session.receiveThread.is_alive():
            self.stopRecording(session.state.get("connection_error") or "ROCK connection lost")
            return
        for name, lastSeen in (("PIR", session.state.get("last_pir_monotonic")),
                               ("Thermopile", session.state.get("last_thermal_monotonic"))):
            reference = lastSeen if lastSeen is not None else session.recordingStartMonotonic
            if now - reference > SENSOR_STALL_SECONDS:
                self.stopRecording(f"{name} stopped responding for over {SENSOR_STALL_SECONDS:g} seconds")
                return

        # The Camera Is Optional: Losing It Only Gives A Warning
        if session.useCamera and not session.cameraLost:
            lastFrame = session.videoInfo.get("last_frame_monotonic") or session.recordingStartMonotonic
            if not session.videoThread.is_alive() or now - lastFrame > CAMERA_STALL_SECONDS:
                session.cameraLost = True
                self.cameraStateVar.set("LOST")
                reason = session.videoInfo.get("runtime_warning") or "no frames"
                self.log(f"WARNING: camera lost ({reason}) - sensors keep recording")
            else:
                self.cameraStateVar.set("recording")

        self.root.after(MONITOR_PERIOD_MS, self.monitor)

    # -------------------------------------------------------------------- Stop
    def onStop(self):
        self.stopRecording("manual_stop")

    def stopRecording(self, reason):
        session = self.session
        if session is None or self.mode not in ("recording", "starting"):
            return
        self.setMode("stopping")
        if reason != "manual_stop":
            self.log(f"Stopping automatically: {reason}")
        self.setStatus("Stopping and saving...", "#1f4e79")
        threading.Thread(target=self.stopWorker, args=(session, reason), daemon=True).start()

    def stopWorker(self, session, stopReason):
        stopPcTime = time.time()
        # The ROCK Closing The Socket After STOP Is Normal; Only Earlier Errors Count
        connectionError = session.state.get("connection_error")
        try:
            sendCommand(session.sock, CMD_STOP)
            session.state["stoppedEvent"].wait(timeout=1.0)
        except OSError:
            pass
        session.videoStopEvent.set()
        if session.videoThread is not None:
            session.videoThread.join(timeout=15.0)
        session.receiveStopEvent.set()
        self.closeSocket(session)
        if session.receiveThread is not None:
            session.receiveThread.join(timeout=2.0)
        session.pirFile.flush()
        session.thermalFile.flush()
        session.pirFile.close()
        session.thermalFile.close()

        # Labels: An Interval Still Open At Stop Ends At The Stop Time
        stopTime = stopPcTime - session.startPcTime
        intervals = pressesToIntervals(session.presses, stopTime)
        writeLabels(session.labelsPath, intervals)

        # Video Bookkeeping (Only If The Camera Actually Recorded Frames)
        frameCount = int(session.videoInfo.get("frame_count") or 0)
        videoSaved = session.cameraRequested and frameCount > 0
        if not videoSaved:
            for path in (session.videoPath, session.videoFramesPath):
                if path.exists():
                    path.unlink()
        measuredFps = None
        firstFrameTime = session.videoInfo.get("first_frame_pc_time")
        lastFrameTime = session.videoInfo.get("last_frame_pc_time")
        if frameCount > 1 and firstFrameTime is not None and lastFrameTime is not None:
            duration = float(lastFrameTime) - float(firstFrameTime)
            if duration > 0:
                measuredFps = (frameCount - 1) / duration

        recordingStatus = session.state.get("recording_status") or session.readyPayload
        metadata = {
            "split": session.split,
            "run_name": session.runName,
            "recording_name": session.runName,
            "collector": "pc_collector_gui",
            "rock_host": session.host,
            "rock_port": session.port,
            "pir_file": session.pirPath.name,
            "thermopile_file": session.thermalPath.name,
            "video_file": session.videoPath.name if videoSaved else None,
            "video_frames_file": session.videoFramesPath.name if videoSaved else None,
            "labels_file": LABEL_FILE,
            "labels_source": "live SPACE toggle on PC clock (inbound zone); press time = label time, no lag correction",
            "label_press_times_s": session.presses,
            "label_interval_count": len(intervals),
            "common_start_pc_time": session.startPcTime,
            "common_start_rock_time": session.startRockTime,
            "stop_pc_time": stopPcTime,
            "rock_to_pc_offset_s": session.state.get("clock_offset_s"),
            "clock_sync_round_trip_s": session.clockSync["round_trip_s"] if session.clockSync else None,
            "clock_sync_pings": session.clockSync["pings"] if session.clockSync else None,
            "pir_rows": session.state["pir_rows"],
            "thermopile_rows": session.state["thermal_rows"],
            "video_frame_count": frameCount,
            "connection_error": connectionError,
            "stop_reason": stopReason,
            "recording_complete": stopReason == "manual_stop" and not connectionError,
            "camera_requested": session.cameraRequested,
            "camera_used": videoSaved,
            "camera_error": session.cameraError,
            "camera_lost_during_recording": session.cameraLost,
            "first_video_frame_time_s": session.videoInfo.get("first_frame_time_s"),
            "video_measured_fps": measuredFps,
            "video_file_fps": VIDEO_FILE_FPS if videoSaved else None,
            "video_width": session.videoInfo.get("width"),
            "video_height": session.videoInfo.get("height"),
            "camera_index": session.cameraIndex if session.cameraRequested else None,
            "camera_backend": session.videoInfo.get("camera_backend"),
            "camera_runtime_warning": session.videoInfo.get("runtime_warning"),
            "pir_baseline_v": recordingStatus.get(KEY_PIR_BASELINE),
            "pir_sample_rate_hz": recordingStatus.get(KEY_PIR_SAMPLE_RATE),
            "thermal_rate_hz": recordingStatus.get(KEY_THERMAL_RATE),
            "thermal_frame_shape": recordingStatus.get(KEY_THERMAL_FRAME_SHAPE),
            "thermal_background": recordingStatus.get(KEY_THERMAL_BACKGROUND),
            "thermal_sigma": recordingStatus.get(KEY_THERMAL_SIGMA),
            "timing_method": (
                "One PC-ROCK clock offset measured immediately before recording; "
                "both devices start at a scheduled common T0; all CSV time_s values are seconds since T0."
            ),
        }
        writeMetadata(session.metadataPath, metadata)

        summary = (f"Saved {session.split}/{session.runName}: {session.state['pir_rows']} PIR samples, "
                   f"{session.state['thermal_rows']} thermal frames, "
                   f"{frameCount if videoSaved else 'no'} video frames, {len(intervals)} labelled intervals.")
        self.post(self.onFinished, session, stopReason, summary)

    def onFinished(self, session, stopReason, summary):
        if session is self.session:
            self.session = None
        self.setMode("idle")
        self.refreshFinishedLabels(session)
        self.cameraStateVar.set("--")
        self.log(summary)
        if stopReason == "manual_stop":
            self.setStatus(summary + " Enter a new name for the next recording.", READY_COLOUR)
        else:
            self.setStatus(f"Stopped early ({stopReason}). " + summary, PERSON_COLOUR)
        self.nameVar.set("")
        self.nameEntry.focus_set()
        if self.closeWhenDone:
            self.root.destroy()

    # Keeps The Finished Recording's Intervals Visible Until The Next One Starts
    def refreshFinishedLabels(self, session):
        self.intervalList.delete(0, "end")
        for index in range(0, len(session.presses), 2):
            start = session.presses[index]
            end = f"{session.presses[index + 1]:8.2f} s" if index + 1 < len(session.presses) else "  (to stop)"
            self.intervalList.insert("end", f"{index // 2 + 1:3d}:  {start:8.2f} s  ->  {end}")

    # -------------------------------------------------------------------- Close
    def onClose(self):
        if self.mode == "idle":
            self.root.destroy()
        elif self.mode in ("recording", "starting"):
            if messagebox.askyesno("Recording in progress", "Stop and save the recording, then close?"):
                self.closeWhenDone = True
                self.stopRecording("window_closed")
        elif self.mode in ("connecting", "ready"):
            self.closeWhenDone = True
            self.onCancel()
        # While "stopping" The Window Closes Itself Once Saving Is Done
        else:
            self.closeWhenDone = True


def main():
    parser = argparse.ArgumentParser(description="PC-side data-collection GUI (camera optional, live labels)")
    parser.add_argument("--host", default="", help="Pre-fill the ROCK 4C+ IP address")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--camera", type=int, default=0, help="Pre-fill the OpenCV camera index")
    parser.add_argument("--no-camera", action="store_true", help="Start with 'Record camera' unticked")
    args = parser.parse_args()

    root = tk.Tk()
    CollectorApp(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
