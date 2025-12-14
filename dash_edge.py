# dash_edge.py
import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import queue
import threading
from datetime import datetime, timezone, timedelta
import plotly.graph_objs as go
import paho.mqtt.client as mqtt

# Optional: lightweight auto-refresh helper (install in requirements). If you don't want it, remove next import and the st_autorefresh call below.
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except Exception:
    HAS_AUTOREFRESH = False

# ---------------------------
# Config (edit if needed)
# ---------------------------
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
TOPIC_SENSOR = "iot/class/session7/sensor_predIlham"
TOPIC_OUTPUT = "iot/class/session7/outputIlham"

# timezone GMT+7 helper
TZ = timezone(timedelta(hours=7))
def now_str():
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")

# ---------------------------
# module-level queue used by MQTT thread (do NOT replace this with st.session_state inside callbacks)
# ---------------------------
GLOBAL_MQ = queue.Queue()

# ---------------------------
# Streamlit page setup
# ---------------------------

st.set_page_config(page_title="IoT ML Realtime Dashboard — Stable", layout="wide")
st.title("IoT ML Realtime Dashboard — Stable")

# ---------------------------
# session_state init (must be done before starting worker)
# ---------------------------
if "msg_queue" not in st.session_state:
    # expose the global queue in session_state so UI can read it
    st.session_state.msg_queue = GLOBAL_MQ

if "logs" not in st.session_state:
    st.session_state.logs = []         # list of dict rows

if "last" not in st.session_state:
    st.session_state.last = None

if "mqtt_thread_started" not in st.session_state:
    st.session_state.mqtt_thread_started = False

# ---------------------------
# MQTT callbacks (use GLOBAL_MQ, NOT st.session_state inside callbacks)
# ---------------------------
def _on_connect(client, userdata, flags, rc):
    try:
        # SUBSCRIBE KE TOPIK YANG KAMU BUTUHKAN
        client.subscribe([
            (TOPIC_SENSOR, 0),
            ("data/status/kontrol", 0),
            ("data/ldr/kontrol", 0),
            ("data/pir/kontrol", 0),
            ("/ai/face/result", 0),
            ("/ai/voice/result", 0),
            ("/ai/face/accuracy", 0),
            ("/ai/voice/accuracy", 0),
            ("/iot/camera/photo", 0)
        ])
    except Exception:
        pass
    # push connection status into queue
    GLOBAL_MQ.put({"_type": "status", "connected": (rc == 0), "ts": time.time()})

def _on_message(client, userdata, msg):
    topic = msg.topic
    payload = msg.payload.decode(errors="ignore")
    try:
        data = json.loads(payload)
    except Exception:
        # push raw payload if JSON parse fails
        GLOBAL_MQ.put({"_type": "raw", "payload": payload, "topic": topic, "ts": time.time()})
        return

    # push structured sensor message
    GLOBAL_MQ.put({"_type": "sensor", "data": data, "topic": topic, "ts": time.time()})

# ---------------------------
# Start MQTT thread (worker)
# ---------------------------
def start_mqtt_thread_once():
    def worker():
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.on_connect = _on_connect
        client.on_message = _on_message
        # optional: configure username/password if needed:
        # client.username_pw_set(USER, PASS)
        while True:
            try:
                client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
                client.loop_forever()
            except Exception as e:
                # push error into queue so UI can show it
                GLOBAL_MQ.put({"_type": "error", "msg": f"MQTT worker error: {e}", "ts": time.time()})
                time.sleep(5)  # backoff then retry

    if not st.session_state.mqtt_thread_started:
        t = threading.Thread(target=worker, daemon=True, name="mqtt_worker")
        t.start()
        st.session_state.mqtt_thread_started = True
        time.sleep(0.05)

# start thread
start_mqtt_thread_once()

# ---------------------------
# Drain queue (process incoming msgs)
# ---------------------------
def process_queue():
    updated = False
    q = st.session_state.msg_queue
    while not q.empty():
        item = q.get()
        ttype = item.get("_type")
        topic = item.get("topic", "")
        if ttype == "status":
            # status - connection
            st.session_state.last_status = item.get("connected", False)
            updated = True
        elif ttype == "error":
            # show error
            st.error(item.get("msg"))
            updated = True
        elif ttype == "raw":
            # JANGAN SIMPAN RAW KE LOG UTAMA, TAPI BISA DITAMPILKAN JIKA DIBUTUHKAN
            pass
        elif ttype == "sensor":
            d = item.get("data", {})
            ts_str = datetime.fromtimestamp(item.get("ts", time.time()), TZ).strftime("%Y-%m-%d %H:%M:%S")

            # PROSES BERDASARKAN TOPIC
            if topic == "data/status/kontrol":
                new_row = {
                    "ts": ts_str,
                    "status_br": d.get("value"),
                    "temp": None,
                    "hum": None,
                    "pred": None,
                    "conf": None
                }
                st.session_state.logs.append(new_row)
                st.session_state.last = new_row
            elif topic == "data/ldr/kontrol":
                # UPDATE ROW TERAKHIR DENGAN DATA INI
                if st.session_state.logs:
                    st.session_state.logs[-1]["temp"] = d.get("value")
            elif topic == "/ai/face/result":
                # UPDATE ROW TERAKHIR DENGAN DATA INI
                if st.session_state.logs:
                    st.session_state.logs[-1]["pred"] = d.get("value")
            elif topic == "/ai/face/accuracy":
                if st.session_state.logs:
                    st.session_state.logs[-1]["conf"] = d.get("value")
            # Tambahkan elif lain untuk topic lain sesuai kebutuhan

            updated = True
    return updated

# run once here to pick up immediately available messages
_ = process_queue()

# ---------------------------
# UI layout
# ---------------------------
# optionally auto refresh UI; requires streamlit-autorefresh in requirements
if HAS_AUTOREFRESH:
    st_autorefresh(interval=2000, limit=None, key="autorefresh")  # 2s refresh

left, right = st.columns([1, 2])

with left:
    st.header("Connection Status")
    st.write("Broker:", f"{MQTT_BROKER}:{MQTT_PORT}")
    connected = getattr(st.session_state, "last_status", None)
    st.metric("MQTT Connected", "Yes" if connected else "No")
    st.write("Topics: sensor, face/voice result")
    st.markdown("---")

    st.header("Last Reading")
    if st.session_state.last:
        last = st.session_state.last
        st.write(f"Time: {last.get('ts')}")
        st.write(f"Temp: {last.get('temp')} °C")
        st.write(f"Hum : {last.get('hum')} %")
        st.write(f"Prediction: {last.get('pred')}")
        st.write(f"Confidence: {last.get('conf')}")
    else:
        st.info("Waiting for data...")

    st.markdown("---")
    st.header("Manual Output Control")
    col1, col2 = st.columns(2)
    if col1.button("ALARM ON MANUALY"):
        try:
            pubc = mqtt.Client()
            pubc.connect(MQTT_BROKER, MQTT_PORT, 60)
            pubc.publish(TOPIC_OUTPUT, "ALERT_ON")
            pubc.disconnect()
            st.success("Published ALERT_ON")
        except Exception as e:
            st.error(f"Send failed: {e}")
    if col2.button("ALARM OFF MANUALY"):
        try:
            pubc = mqtt.Client()
            pubc.connect(MQTT_BROKER, MQTT_PORT, 60)
            pubc.publish(TOPIC_OUTPUT, "ALERT_OFF")
            pubc.disconnect()
            st.success("Sended...")
        except Exception as e:
            st.error(f"Send failed: {e}")

    st.markdown("---")
    st.header("Download Logs")
    if st.button("Download CSV"):
        if st.session_state.logs:
            df_dl = pd.DataFrame(st.session_state.logs)
            csv = df_dl.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV file", data=csv, file_name=f"iot_logs_{int(time.time())}.csv")
        else:
            st.info("No logs to download")

with right:
    st.header("Live Chart (last 200 points)")
    df_plot = pd.DataFrame(st.session_state.logs[-200:])
    if (not df_plot.empty) and {"temp", "hum"}.issubset(df_plot.columns):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_plot["ts"], y=df_plot["temp"], mode="lines+markers", name="Temp (°C)"))
        fig.add_trace(go.Scatter(x=df_plot["ts"], y=df_plot["hum"], mode="lines+markers", name="Hum (%)", yaxis="y2"))
        fig.update_layout(
            yaxis=dict(title="Temp (°C)"),
            yaxis2=dict(title="Humidity (%)", overlaying="y", side="right", showgrid=False),
            height=520
        )
        # color markers by anomaly / label
        colors = []
        for _, r in df_plot.iterrows():
            lab = r.get("pred", "")
            if lab == "Panas":
                colors.append("red")
            elif lab == "Normal":
                colors.append("green")
            elif lab == "Dingin":
                colors.append("blue")
            else:
                colors.append("gray")
        fig.update_traces(marker=dict(size=8, color=colors), selector=dict(mode="lines+markers"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet. Make sure ESP32 publishes to correct topic.")

    st.markdown("### Recent Logs")
    if st.session_state.logs:
        st.dataframe(pd.DataFrame(st.session_state.logs)[::-1].head(100))
    else:
        st.write("—")

# after UI render, drain queue (so next rerun shows fresh data)
process_queue()