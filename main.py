import io
from time import sleep
import numpy as np
import torch
from PIL import Image
import os
import cv2
import ssl
import sys
from aiohttp import web
from PIL import Image, ImageDraw
from ultralytics import YOLO
import face_recognition

print(">>> Checking for GPUs...")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on device: {}".format(device))

print(">>> Preparing Machine Learning models...")
model = YOLO("yolo11n.pt")

known_face_encodings = []
known_face_names = ["khoa", "khoa2", "khoa3"]
face_caches = {}

for name in known_face_names:
    pathtoimg = "./faces/" + name + ".jpg"
    if not os.path.exists(pathtoimg): 
        pathtoimg = "./faces/" + name + ".png"
        if not os.path.exists(pathtoimg): continue

    image = face_recognition.load_image_file(pathtoimg)
    face_encodings = face_recognition.face_encodings(image)

    if face_encodings:
        # append encoding to known faces
        known_face_encodings.append(face_encodings[0])
        print("> Loaded face for", name)


# websocket
async def recv_images(r):
    ws = web.WebSocketResponse()
    await ws.prepare(r)

    print("ws connected: ", r.remote)

    async for msg in ws:
        if msg.type == web.WSMsgType.BINARY:
            img_bytes = io.BytesIO(msg.data)
            np_array = np.frombuffer(img_bytes.getvalue(), np.uint8)
            frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            # Run YOLO11 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True, classes=[0], tracker="botsort.yaml")

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLO Tracking", annotated_frame)

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)

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
                            face_encodings = face_recognition.face_encodings(rgb_crop, face_locations)

                            if face_encodings:
                                # only match 1st
                                face_encoding = face_encodings[0]

                                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.9)
                                print("Match?> ", matches)
                                #sleep(10000)
                                if True in matches:
                                    first_match_index = matches.index(True)
                                    identity = known_face_names[first_match_index]
                                    face_caches[track_id] = identity

                    
                    identity = face_caches.get(track_id, "Unknown")
                    label = f"{identity} (id: {track_id})"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(frame, (x1, y1 - h - 15), (x1 + w, y1 - 10), (0, 255, 0), -1)

                    cv2.putText(frame, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


            cv2.imshow('people', frame)
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    return ws



async def index(_):
    return web.FileResponse("./index.html")



def main():
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain("cert.crt", "cert.key")

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/ws", recv_images)

    web.run_app(app, host="0.0.0.0", port=13577, ssl_context=ssl_context)


def cli():
    face_in = sys.argv[1]
    if face_in is None:
        return
    face_out = sys.argv[2]
    if face_out is None:
        return

    if not os.path.isfile(face_in):
        print("E: File not found")
        sys.exit(1)


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            cli()
        else:
            main()
    except:
        cv2.destroyAllWindows()