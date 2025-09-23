from aiohttp import web
from av import VideoFrame
import io
import time
from cv2.gapi import video
import numpy as np
import string
import torch
import os
import cv2
import ssl
import random
import concurrent.futures
from aiortc import (
    RTCRtpSender,
    VideoStreamTrack,
    RTCPeerConnection,
    RTCSessionDescription,
)

import asyncio
import mlopts

q = asyncio.Queue


routes = web.RouteTableDef()
latest_frame = None
frame_lock = asyncio.Lock()


# WebRTC Handlers
class ProcessedTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()

    async def recv(self):
        global latest_frame
        pts, time_base = await self.next_timestamp()

        async with frame_lock:
            frame = (
                latest_frame.copy()
                if latest_frame is not None
                else np.full((480, 640, 3), (0, 0, 255), np.uint8)
            )

        new_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        new_frame.pts = pts
        new_frame.time_base = time_base
        return new_frame


pcs = set()
auth_toks = set()


@routes.get("/get_cam_token")
async def get_camera_token(_: web.Request):
    tok = "".join(
        random.choice(string.ascii_letters + string.digits) for _ in range(24)
    )
    auth_toks.add(tok)
    return web.json_response(tok)


@routes.post("/offer")
async def offer(request: web.Request):
    # Bearer [AuthKey]
    auth = request.headers.get("Authorization")
    print(request.headers)
    if auth is None:
        return web.json_response(
            {"message": "No authentication headers provided."}, status=401
        )
    try:
        auth_toks.remove(auth.split(" ")[1])
    except:
        return web.json_response(
            {"message": "Invalid authentication token."}, status=401
        )

    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    # add processed track
    pc.addTrack(ProcessedTrack())

    @pc.on("iceconnectionstatechange")
    def on_state_change():
        print("ICE state:", pc.iceConnectionState)
        if pc.iceConnectionState == "failed":
            asyncio.ensure_future(pc.close())
            pcs.discard(pc)

    await pc.setRemoteDescription(offer)

    video_trans = next((t for t in pc.getTransceivers() if t.kind == "video"), None)
    if video_trans:
        all_codecs = RTCRtpSender.getCapabilities("video").codecs
        print("Available codecs: ")
        print(all_codecs)

        # Create a list of preferred codecs, with H.264 first, then VP9
        preferred_codecs = []
        for codec in all_codecs:
            if codec.mimeType.lower() == "video/h264":
                preferred_codecs.insert(0, codec)  # Add H.264 to the front
            elif codec.mimeType.lower() == "video/vp9":
                preferred_codecs.append(codec)  # Add VP9 after H.264

        if preferred_codecs:
            print("Setting preferred codecs: ", preferred_codecs)
            video_trans.setCodecPreferences(preferred_codecs)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response(
        {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    )


executor = concurrent.futures.ThreadPoolExecutor()


@routes.get("/ws/camera")
async def recv_images(r):
    ws = web.WebSocketResponse()
    await ws.prepare(r)

    print("ws connected: ", r.remote)

    global latest_frame
    loop = asyncio.get_running_loop()
    i = 0
    async for msg in ws:
        if msg.type == web.WSMsgType.BINARY:
            start_time = time.perf_counter()
            img_bytes = io.BytesIO(msg.data)
            np_array = np.frombuffer(img_bytes.getvalue(), np.uint8)
            frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            if frame is None:
                continue  # Receive next frame

            async with frame_lock:
                latest_frame = frame

            print(i)
            if i < 6:
                i += 1
            else:
                print("Running in executor")
                loop.run_in_executor(executor, mlopts.process_cam_frame, frame)
                i = 0

            duration = time.perf_counter() - start_time
            print(f"Processing operation took {duration:.4f} seconds.")

    return ws


# Static Routes
@routes.get("/webcam")
async def webcam_http(_: web.Request):
    return web.FileResponse("static/webcam_stream.html")


@routes.get("/")
async def recver_http(_: web.Request):
    return web.FileResponse("static/receiver.html")


app = web.Application()
app.add_routes(routes)

if __name__ == "__main__":
    print("Server running at http://localhost:13577")
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain("cert.crt", "cert.key")

    web.run_app(app, host="0.0.0.0", port=13577, ssl_context=ssl_context)
