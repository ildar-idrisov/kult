import subprocess
import pygetwindow as gw

SERVER_IP = '192.168.10.13'
SERVER_PORT = 5000
PASSPHRASE = 'SnSu72im4o12'
WINDOW_NAME = 'Google Chrome'

def record_with_ffmpeg(region, audio_device, duration=10, server_url=None):
    if server_url is None:
        server_url = f"srt://{SERVER_IP}:{SERVER_PORT}?mode=caller&encryption=encrypt&pbkeylen=16&passphrase={PASSPHRASE}"

    chunk_duration = 2.4
    command = [
        "ffmpeg",
        "-y",
        "-f", "gdigrab",
        "-draw_mouse", "0",
        "-framerate", "10",
        "-offset_x", str(region[0]),
        "-offset_y", str(region[1]),
        "-video_size", f"{region[2]}x{region[3]}",
        "-i", "desktop",
        "-f", "dshow",
        "-i", f"audio={audio_device}",
        #"-t", str(duration),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-profile:v", "baseline",
        "-force_key_frames", f"expr:gte(t,n_forced*{chunk_duration})",
        "-max_delay", "0",
        "-flags", "+low_delay",
        "-f", "mpegts",
        server_url
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Streaming started! Streaming to {server_url}")
    except subprocess.CalledProcessError as e:
        print(f"Streaming error: {e}")

titles = gw.getAllTitles()
window_title = None
for title in titles:
    if WINDOW_NAME in title:
        window_title = title
        break

if window_title:
    window = gw.getWindowsWithTitle(window_title)[0]
    region = (window.left, window.top, window.width, window.height)
    print(window)
    record_with_ffmpeg(
        region,
        "Microphone Array (Realtek High Definition Audio)",
        duration=30
    )
else:
    print("Window not found.")