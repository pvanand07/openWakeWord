import streamlit as st
import plotly.graph_objects as go
import numpy as np
import collections
from functools import partial
import pyaudio
import threading
import time
from openwakeword.model import Model

# Page config
st.set_page_config(
    page_title="openWakeWord Live Demo",
    page_icon="🎤",
    layout="wide"
)

# Initialize session state
if 'model' not in st.session_state:
    import openwakeword
    openwakeword.utils.download_models()
    st.session_state.model = Model(inference_framework="onnx")
    st.session_state.scores = collections.defaultdict(partial(collections.deque, maxlen=10))
    st.session_state.recording = False
    st.session_state.audio_thread = None

# Audio settings
CHUNK = 1280
RATE = 16000
FORMAT = pyaudio.paInt16
CHANNELS = 1

def audio_callback(in_data, frame_count, time_info, status):
    """Callback for audio stream"""
    if st.session_state.recording:
        audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Get prediction
        prediction = st.session_state.model.predict(audio_data)
        
        # Update scores
        for key in prediction:
            st.session_state.scores[key].append(prediction[key])
    
    return (in_data, pyaudio.paContinue)

def start_recording():
    """Start audio recording"""
    st.session_state.recording = True
    p = pyaudio.PyAudio()
    
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        stream_callback=audio_callback
    )
    
    stream.start_stream()
    
    # Keep stream alive
    while st.session_state.recording:
        time.sleep(0.1)
    
    stream.stop_stream()
    stream.close()
    p.terminate()

def create_chart():
    """Create Plotly bar chart"""
    if not st.session_state.scores:
        # Empty chart
        fig = go.Figure()
        fig.update_layout(
            title="Real-time Wake Word Detection",
            xaxis_title="Detection Score",
            yaxis_title="Model",
            height=500,
            xaxis_range=[0, 1],
            template="plotly_white"
        )
        return fig
    
    # Calculate average scores
    model_names = []
    scores = []
    
    for key in st.session_state.scores.keys():
        if len(st.session_state.scores[key]) > 0:
            model_names.append(key.replace('_', ' ').title())
            scores.append(np.mean(list(st.session_state.scores[key])))
    
    # Sort by score
    sorted_indices = np.argsort(scores)
    model_names = [model_names[i] for i in sorted_indices]
    scores = [scores[i] for i in sorted_indices]
    
    # Create bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=scores,
        y=model_names,
        orientation='h',
        marker=dict(
            color=scores,
            colorscale='Blues',
            cmin=0,
            cmax=1,
            line=dict(color='rgb(8,48,107)', width=1.5)
        ),
        text=[f'{score:.3f}' for score in scores],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Score: %{x:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text="Real-time Wake Word Detection",
            font=dict(size=20, color='#1f77b4')
        ),
        xaxis=dict(
            title="Detection Score",
            range=[0, 1.15],
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=12)
        ),
        height=max(400, len(model_names) * 40),
        template="plotly_white",
        showlegend=False,
        margin=dict(l=20, r=100, t=60, b=40)
    )
    
    return fig

# Header
st.title("🎤 openWakeWord Live Demo")

st.markdown("""
This demo uses the pre-trained models from the [openWakeWord](https://github.com/dscripka/openWakeWord) library.

**Try these phrases:**
- **alexa** - "alexa"
- **hey mycroft** - "hey mycroft"
- **hey jarvis** - "hey jarvis"
- **hey rhasspy** - "hey rhasspy"
- **weather** - "what's the weather", "tell me today's weather"
- **timer** - "set a timer for 1 minute", "create 1 hour alarm"
""")

# Control buttons
col1, col2, col3 = st.columns([1, 1, 4])

with col1:
    if st.button("🎙️ Start Recording", disabled=st.session_state.recording):
        if st.session_state.audio_thread is None or not st.session_state.audio_thread.is_alive():
            st.session_state.audio_thread = threading.Thread(target=start_recording, daemon=True)
            st.session_state.audio_thread.start()
            st.rerun()

with col2:
    if st.button("⏹️ Stop Recording", disabled=not st.session_state.recording):
        st.session_state.recording = False
        time.sleep(0.2)
        st.rerun()

with col3:
    if st.session_state.recording:
        st.success("🔴 Recording...")
    else:
        st.info("⚪ Not recording")

# Chart container
chart_placeholder = st.empty()

# Auto-refresh when recording
if st.session_state.recording:
    while st.session_state.recording:
        with chart_placeholder:
            st.plotly_chart(create_chart(), use_container_width=True)
        time.sleep(0.1)  # Update every 100ms
else:
    with chart_placeholder:
        st.plotly_chart(create_chart(), use_container_width=True)