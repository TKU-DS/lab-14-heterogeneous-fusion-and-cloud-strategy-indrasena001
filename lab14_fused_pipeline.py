import json
import os
import queue
import random
import threading
import time

try:
    import paho.mqtt.client as mqtt

    HAS_MQTT = True
except ImportError:
    HAS_MQTT = False

# =================================================================
# Course: Data Engineering (CSIE, Tamkang University)
# Lab 14: Heterogeneous Fusion & MQTT Strategy
# =================================================================

SENSOR_RATE_HZ = 20
VISION_RATE_HZ = 4
TOLERANCE_MS = 100
CACHE_FILE = "local.jsonl"
MQTT_BROKER = os.environ.get("LAB14_MQTT_BROKER", "localhost")
MQTT_PORT = int(os.environ.get("LAB14_MQTT_PORT", "1883"))
MAX_RUNTIME_SECONDS = float(os.environ.get("LAB14_RUNTIME_SECONDS", "8"))

sensor_queue = queue.Queue(maxsize=100)
vision_queue = queue.Queue(maxsize=5)
stop_event = threading.Event()


def sensor_producer():
    """Generate high-frequency water-level samples."""
    print(f"[Sensor Thread] Started ({SENSOR_RATE_HZ}Hz).")
    while not stop_event.is_set():
        data = {
            "ts": time.time(),
            "val": round(random.uniform(2.0, 4.2), 2),
        }

        try:
            sensor_queue.put_nowait(data)
        except queue.Full:
            try:
                sensor_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                sensor_queue.put_nowait(data)
            except queue.Full:
                pass

        time.sleep(1.0 / SENSOR_RATE_HZ)


def vision_producer():
    """Generate low-frequency YOLO-style detections."""
    print(f"[Vision Thread] Started ({VISION_RATE_HZ}Hz).")
    while not stop_event.is_set():
        data = {
            "ts": time.time(),
            "count": random.randint(0, 10),
        }

        try:
            vision_queue.put_nowait(data)
        except queue.Full:
            try:
                vision_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                vision_queue.put_nowait(data)
            except queue.Full:
                pass

        time.sleep(1.0 / VISION_RATE_HZ)


def find_nearest_sensor(vision_ts):
    """Use nearest-neighbor join over the recent sensor buffer."""
    best_match = None
    min_diff = float("inf")

    for sample in list(sensor_queue.queue):
        diff = abs(vision_ts - sample["ts"])
        if diff < min_diff:
            min_diff = diff
            best_match = sample

    return best_match, min_diff


def build_payload(vision, best_match, min_diff_ms):
    """Pack fused data into a structured JSON record."""
    return {
        "schema_version": "1.0",
        "event_id": f"lab14-{int(time.time() * 1000)}",
        "vision_ts": round(vision["ts"], 6),
        "sensor_ts": round(best_match["ts"], 6),
        "sync_error_ms": round(min_diff_ms, 2),
        "data": {
            "water_level_m": best_match["val"],
            "debris_count": vision["count"],
        },
    }


def publish_data(payload):
    """Publish to MQTT if available; otherwise cache to local.jsonl."""
    if HAS_MQTT:
        try:
            client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                client_id="lab14-fusion",
            )
            client.connect(MQTT_BROKER, MQTT_PORT, 5)
            client.publish("lab14/fused", json.dumps(payload), qos=1)
            client.disconnect()
            print("[CLOUD] Published to MQTT broker.")
            return
        except Exception as exc:
            print(f"[CLOUD] MQTT unavailable ({exc}); falling back to local cache.")

    print("[CACHE] Saving record to local.jsonl...")
    with open(CACHE_FILE, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


if __name__ == "__main__":
    threading.Thread(target=sensor_producer, daemon=True).start()
    threading.Thread(target=vision_producer, daemon=True).start()

    print("[*] Starting Fusion Pipeline...")
    print(f"[*] Running for {MAX_RUNTIME_SECONDS:.0f}s (tolerance: {TOLERANCE_MS}ms).")

    start_time = time.time()

    try:
        while time.time() - start_time < MAX_RUNTIME_SECONDS:
            try:
                vision = vision_queue.get(timeout=0.25)
            except queue.Empty:
                continue

            best_match, min_diff = find_nearest_sensor(vision["ts"])
            min_diff_ms = min_diff * 1000.0

            if best_match and min_diff_ms < TOLERANCE_MS:
                payload = build_payload(vision, best_match, min_diff_ms)
                print(f"\n[FUSION] Aligned with {payload['sync_error_ms']}ms error.")
                publish_data(payload)
            else:
                print(f"[!] Sync failed: time error {min_diff_ms:.2f}ms exceeds {TOLERANCE_MS}ms.")

    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        print("\n[*] System Shutdown initiated.")
