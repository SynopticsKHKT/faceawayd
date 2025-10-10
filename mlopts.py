import numpy as np
import time
import string
import torch
import os
import cv2
import mlutils
import face_recognition
from ultralytics import YOLO
import requests
from typing import Dict, Tuple


print(">>> Checking for GPUs...")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on device: {}".format(device))

print(">>> Preparing Machine Learning models...")
model = YOLO("yolo11n.pt")
door_model = YOLO("doors.pt")

known_face_encodings = []
door_boxes = []
known_face_names = ["khoa", "khoa2", "khoa3"]
face_caches = {}

# mapping of door index -> room name (hardcoded for now)
DOOR_ROOMS = {
    0: "phong_khach",
    1: "phong_ngu",
}

# per-track last known door index (None if not in door)
# per-track last known door index (None if not in door)
track_last_door: Dict[int, Tuple[int, float]] = {}
# per-track current room name (if known)
track_current_room: Dict[int, Tuple[str, float]] = {}
# dedupe TTL seconds per track-door event
DEDUP_TTL = 5.0

# backend endpoint to post events to
BACKEND_URL = os.environ.get("SYNOPTIC_BACKEND", "http://localhost:13578/api/rooms/event")

# Load door boxes from doors.json if present
try:
    import json
    with open("doors.json", "r") as f:
        dd = json.load(f)
        door_boxes = []
        DOOR_ROOMS = {}
        for i, entry in enumerate(dd.get("doors", [])):
            door_boxes.append(entry.get("box", []))
            DOOR_ROOMS[i] = entry.get("room", f"door_{i}")
        print(f"Loaded {len(door_boxes)} door boxes from doors.json")
except Exception:
    print("No doors.json found or failed to load; using default DOOR_ROOMS")

for name in known_face_names:
    pathtoimg = "./faces/" + name + ".jpg"
    if not os.path.exists(pathtoimg):
        pathtoimg = "./faces/" + name + ".png"
        if not os.path.exists(pathtoimg):
            continue

    image = face_recognition.load_image_file(pathtoimg)
    face_encodings = face_recognition.face_encodings(image)

    if face_encodings:
        # append encoding to known faces
        known_face_encodings.append(face_encodings[0])
        print("> Loaded face for", name)


def process_cam_frame(frame):
    print("Processing frame with AI...")
    start_time = time.perf_counter()
    # Run YOLO11 tracking on the frame, persisting tracks between frames
    results = model.track(frame, persist=True, classes=[0], tracker="botsort.yaml")

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)

        seen_tracks = set()
        for box, track_id in zip(boxes, track_ids):
            identity = "Unknown"
            x1, y1, x2, y2 = box
            if track_id in face_caches:
                identity = face_caches[track_id]
            else:
                person_crop = frame[y1:y2, x1:x2]

                # prevent stupid
                if person_crop.size > 0:
                    rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_crop)
                    face_encodings = face_recognition.face_encodings(
                        rgb_crop, face_locations
                    )

                    if face_encodings:
                        # only match 1st
                        face_encoding = face_encodings[0]

                        matches = face_recognition.compare_faces(
                            known_face_encodings, face_encoding, tolerance=0.9
                        )
                        print("Match?> ", matches)
                        # sleep(10000)
                        if True in matches:
                            first_match_index = matches.index(True)
                            identity = known_face_names[first_match_index]
                            face_caches[track_id] = identity

            identity = face_caches.get(track_id, "Unknown")
            label = f"{identity} (id: {track_id})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - h - 15), (x1 + w, y1 - 10), (0, 255, 0), -1)

            cv2.putText(
                frame,
                label,
                (x1, y1 - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )
            # Check door intersections
            # door_boxes elements expected as [x1,y1,x2,y2]
            for di, db in enumerate(door_boxes):
                dx1, dy1, dx2, dy2 = db
                # compute intersection over area (simple bbox overlap)
                ix1 = max(x1, dx1)
                iy1 = max(y1, dy1)
                ix2 = min(x2, dx2)
                iy2 = min(y2, dy2)
                if ix2 > ix1 and iy2 > iy1:
                    # there's an intersection (door crossing)
                    now = time.time()
                    last = track_last_door.get(track_id)
                    # only process if new door or past dedupe TTL
                    if last is None or last[0] != di or (now - last[1]) > DEDUP_TTL:
                        door_room = DOOR_ROOMS.get(di, f"door_{di}")

                        # check if this track was in a different room before
                        prev_room_entry = track_current_room.get(track_id)
                        prev_room = prev_room_entry[0] if prev_room_entry is not None else None

                        # if changed room, send exit for previous room then enter for new
                        try:
                            if prev_room is not None and prev_room != door_room:
                                # send exit for previous room
                                payload_exit = {"room": prev_room, "label": label, "action": "exit"}
                                r = requests.post(BACKEND_URL, json=payload_exit, timeout=1.0)
                                print(f"Posted EXIT to backend {payload_exit} -> {r.status_code}")

                            # send enter for new room
                            payload_enter = {"room": door_room, "label": label, "action": "enter"}
                            r2 = requests.post(BACKEND_URL, json=payload_enter, timeout=1.0)
                            print(f"Posted ENTER to backend {payload_enter} -> {r2.status_code}")

                            # update trackers
                            track_last_door[track_id] = (di, now)
                            track_current_room[track_id] = (door_room, now)
                        except Exception as e:
                            print("Failed to post transition event:", e)
                    break
        seen_tracks.add(int(track_id))
    duration = time.perf_counter() - start_time
    print(f"Processing AI frame operation took {duration:.4f} seconds.")
    # cv2.imshow("Test", frame)

    # cleanup stale tracks: if a track was last seen longer than STALE_TTL, send exit
    STALE_TTL = 5.0
    now = time.time()
    to_remove = []
    for tid, (room, ts) in track_current_room.items():
        if int(tid) not in locals().get('seen_tracks', set()) and (now - ts) > STALE_TTL:
            # send exit event
            try:
                payload_exit = {"room": room, "label": f"unknown (id: {tid})", "action": "exit"}
                r = requests.post(BACKEND_URL, json=payload_exit, timeout=1.0)
                print(f"Posted STALE EXIT to backend {payload_exit} -> {r.status_code}")
            except Exception as e:
                print("Failed to post stale exit:", e)
            to_remove.append(tid)

    for tid in to_remove:
        track_current_room.pop(tid, None)


def process_door_frame(frame):
    # process the cur door frame to return boxes of doors
    results = door_model.track(frame)
    cv2.imshow("oputput", results[0].plot())

    if cv2.waitKey(1) & 0xFF == ord("q"):
        return False
