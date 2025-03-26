import streamlit as st
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import librosa
import os
import tempfile
import time
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.graph_objects as go
import json

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize NLTK resources if not already downloaded
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Setting page config
st.set_page_config(
    page_title="ToneCoach: Emotional Intelligence for Public Speaking",
    page_icon="🎤",
    layout="wide",
)

# App title and description
st.title("ToneCoach: Emotional Intelligence for Public Speaking")
st.markdown("""
    Improve your speech delivery with AI-powered feedback on tone, pitch, 
    pacing, and emotional expressiveness. Get coached in your own voice!
""")

# Sidebar for settings
st.sidebar.header("Settings")

# Diction type selection
diction_types = {
    "Professional Speaking": "Clear, authoritative tone with precise articulation for business presentations.",
    "Casual Conversation": "Relaxed, natural speech patterns suitable for everyday interactions.",
    "Public Speech": "Formal, engaging delivery with varied pacing and emphasis for audience engagement.",
    "Storytelling": "Expressive, dynamic narration with emotional variations to captivate listeners.",
    "News Anchoring": "Neutral, clear delivery with consistent pacing and enunciation."
}

selected_diction = st.sidebar.selectbox(
    "Select Diction Type to Learn:",
    list(diction_types.keys())
)

st.sidebar.markdown(f"**Description:** {diction_types[selected_diction]}")

# Sample phrases based on diction type
sample_phrases = {
    "Professional Speaking": [
        "I'm confident we can achieve our quarterly targets with this strategy.",
        "Let me outline the key benefits of this proposal for all stakeholders.",
        "The data clearly indicates a significant opportunity for market expansion."
    ],
    "Casual Conversation": [
        "Hey, how's it going? I haven't seen you in ages!",
        "That movie was pretty good, but the ending surprised me.",
        "I was thinking we could grab coffee sometime next week?"
    ],
    "Public Speech": [
        "Today, we stand at the threshold of a new era of innovation.",
        "Let me share with you three powerful insights that changed my perspective.",
        "We must ask ourselves: what legacy will we leave for future generations?"
    ],
    "Storytelling": [
        "The old mansion stood silent against the stormy night sky.",
        "Her eyes widened as she realized the true meaning of the ancient riddle.",
        "With trembling hands, he opened the mysterious letter from his long-lost brother."
    ],
    "News Anchoring": [
        "Breaking news: Officials have announced a major policy change effective immediately.",
        "Experts are warning of severe weather conditions expected throughout the weekend.",
        "In international news, diplomatic talks have resumed between the two nations."
    ]
}

selected_phrase = st.sidebar.selectbox(
    "Select a sample phrase to practice (or speak your own):",
    sample_phrases[selected_diction]
)

# Advanced settings
st.sidebar.header("Advanced Settings")
pitch_sensitivity = st.sidebar.slider("Pitch Analysis Sensitivity", 0.5, 2.0, 1.0, 0.1)
tempo_sensitivity = st.sidebar.slider("Tempo Analysis Sensitivity", 0.5, 2.0, 1.0, 0.1)
voice_clone_option = st.sidebar.checkbox("Enable Voice Cloning", False)
export_option = st.sidebar.checkbox("Enable Export of Analysis", False)

# Helper functions
def analyze_pitch(y, sr):
    """Extract pitch information from audio."""
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = []
    
    for i in range(pitches.shape[1]):
        index = magnitudes[:, i].argmax()
        pitch = pitches[index, i]
        if pitch > 0:  # Filter out zero pitches
            pitch_values.append(pitch)
    
    if not pitch_values:
        return {
            "values": [],
            "mean": 0,
            "std": 0,
            "is_monotone": True
        }
    
    pitch_std = np.std(pitch_values)
    # Higher sensitivity means we're more likely to detect monotone speech
    monotone_threshold = 15 / pitch_sensitivity
    
    return {
        "values": pitch_values,
        "mean": float(np.mean(pitch_values)) if pitch_values else 0,  # Convert to Python float
        "std": float(pitch_std) if pitch_values else 0,               # Convert to Python float
        "is_monotone": bool(pitch_std < monotone_threshold) if pitch_values else True  # Convert to Python bool
    }

def analyze_energy(y, sr):
    """Calculate energy (volume) over time."""
    # Calculate energy in short windows
    hop_length = int(sr / 10)  # 100ms windows
    energy = []
    
    for i in range(0, len(y), hop_length):
        if i + hop_length < len(y):
            energy.append(np.sum(np.abs(y[i:i+hop_length])))
        else:
            energy.append(np.sum(np.abs(y[i:])))
    
    if not energy:
        return {
            "values": [],
            "mean": 0,
            "std": 0,
            "low_variation": True
        }
    
    energy_std = np.std(energy)
    energy_mean = np.mean(energy)
    variation_ratio = energy_std / energy_mean if energy_mean > 0 else 0
    
    return {
        "values": energy,
        "mean": energy_mean,
        "std": energy_std,
        "variation_ratio": variation_ratio,
        "low_variation": variation_ratio < 0.2
    }

def analyze_tempo(y, sr):
    """Analyze speech tempo and rhythm."""
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    # Adjust tempo by sensitivity
    tempo = tempo * tempo_sensitivity
    
    # Determine if pacing is appropriate for the selected diction type
    tempo_ranges = {
        "Professional Speaking": (120, 160),
        "Casual Conversation": (130, 180),
        "Public Speech": (100, 140),
        "Storytelling": (90, 150),
        "News Anchoring": (140, 180)
    }
    
    min_tempo, max_tempo = tempo_ranges[selected_diction]
    pacing_status = "good"
    
    if tempo < min_tempo:
        pacing_status = "too_slow"
    elif tempo > max_tempo:
        pacing_status = "too_fast"
    
    return {
        "bpm": float(tempo),  # Convert to Python float
        "beat_frames": beat_frames,
        "pacing_status": pacing_status
    }

def generate_feedback(pitch_analysis, energy_analysis, tempo_analysis, text_analysis=None):
    """Generate personalized feedback based on speech analysis."""
    feedback = []
    
    # Pitch feedback
    if pitch_analysis["is_monotone"]:
        if selected_diction == "Professional Speaking":
            feedback.append("Your pitch is slightly monotone. Try varying your tone more for emphasis on key points.")
        elif selected_diction == "Casual Conversation":
            feedback.append("Your speech has limited pitch variation. Try sounding more animated to express interest.")
        elif selected_diction == "Public Speech":
            feedback.append("Try emphasizing key words with slight pitch changes to maintain audience engagement.")
        elif selected_diction == "Storytelling":
            feedback.append("Add more dramatic pitch variations to bring characters to life and create emotion.")
        elif selected_diction == "News Anchoring":
            feedback.append("Maintain consistent pitch with strategic variations to highlight important information.")
    
    # Energy/volume feedback
    if energy_analysis["low_variation"]:
        if selected_diction in ["Public Speech", "Storytelling"]:
            feedback.append("Increase your dynamic range by emphasizing certain words with more energy and others with less.")
        else:
            feedback.append("Try varying your volume slightly for more engaging delivery.")
    
    # Tempo feedback
    if tempo_analysis["pacing_status"] == "too_slow":
        if selected_diction == "News Anchoring":
            feedback.append("Your pacing is a bit slow for news delivery. Aim for a slightly faster, more consistent pace.")
        elif selected_diction == "Professional Speaking":
            feedback.append("Consider a slightly faster pace to maintain audience engagement.")
        else:
            feedback.append("Try increasing your pace slightly to maintain listener interest.")
    elif tempo_analysis["pacing_status"] == "too_fast":
        if selected_diction == "Storytelling":
            feedback.append("Slow down your storytelling to allow listeners to absorb the narrative.")
        elif selected_diction == "Public Speech":
            feedback.append("You're speaking quite rapidly. Consider adding strategic pauses after important points.")
        else:
            feedback.append("Try slowing down your pace slightly for better clarity.")
    
    # Add general feedback based on diction type if we don't have many specific points
    if len(feedback) < 2:
        if selected_diction == "Professional Speaking":
            feedback.append("Good pacing overall, but consider slowing down for key points.")
            feedback.append("Increase energy slightly to convey more authority.")
        elif selected_diction == "Casual Conversation":
            feedback.append("Your speech sounds slightly formal. Try using more varied intonation.")
            feedback.append("Consider adding more emotional coloring to sound more engaged.")
        elif selected_diction == "Public Speech":
            feedback.append("Great pacing, but add more strategic pauses for impact.")
            feedback.append("Your energy builds well, but start stronger to grab attention.")
        elif selected_diction == "Storytelling":
            feedback.append("Slow down during descriptive passages for greater effect.")
            feedback.append("Exaggerate emotional contrasts more for story impact.")
        elif selected_diction == "News Anchoring":
            feedback.append("Excellent clear diction, but slightly increase your pace.")
            feedback.append("Great neutral tone, perfect for news delivery.")
    
    return feedback

def transcribe_speech(audio_file_path):
    """Placeholder for speech-to-text functionality."""
    # In a full implementation, this would use Whisper API or another STT service
    # For now, we'll return a placeholder message
    return "Speech transcription would appear here in the full implementation."

def export_analysis_data(pitch_analysis, energy_analysis, tempo_analysis, 
                         selected_diction, selected_phrase, expressiveness_score,
                         feedback_items):
    """Generate and export analysis data in JSON format with proper type handling."""
    import json
    import time
    import streamlit as st
    import numpy as np
    
    # Create a custom JSON encoder to handle special data types
    class AnalysisJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            # Handle NumPy data types
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            if isinstance(obj, (np.bool_)):
                return bool(obj)
            # Handle other custom objects
            return super().default(obj)
    
    try:
        # Prepare the export data with explicit type conversions
        export_data = {
            "analysis_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "diction_type": str(selected_diction),
            "phrase": str(selected_phrase),
            "pitch_analysis": {
                "mean_pitch": float(pitch_analysis["mean"]) if "mean" in pitch_analysis else 0.0,
                "pitch_std": float(pitch_analysis["std"]) if "std" in pitch_analysis else 0.0,
                "is_monotone": bool(pitch_analysis["is_monotone"]) if "is_monotone" in pitch_analysis else False
            },
            "energy_analysis": {
                "mean_energy": float(energy_analysis["mean"]) if "mean" in energy_analysis else 0.0,
                "energy_variation": float(energy_analysis["variation_ratio"]) 
                    if "variation_ratio" in energy_analysis else 0.0,
                "low_variation": bool(energy_analysis["low_variation"]) 
                    if "low_variation" in energy_analysis else False
            },
            "tempo_analysis": {
                "bpm": float(tempo_analysis["bpm"]) if "bpm" in tempo_analysis else 0.0,
                "pacing_status": str(tempo_analysis["pacing_status"]) 
                    if "pacing_status" in tempo_analysis else "unknown"
            },
            "expressiveness_score": float(expressiveness_score),
            "feedback": [str(item) for item in feedback_items]
        }
        
        # Convert to JSON using the custom encoder
        json_data = json.dumps(export_data, indent=2, cls=AnalysisJSONEncoder)
        
        # Create download button
        st.download_button(
            label="Download Analysis Report (JSON)",
            data=json_data,
            file_name=f"tonecoach_analysis_{time.strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        return True
    
    except Exception as e:
        st.error(f"Error generating JSON export: {str(e)}")
        st.info("Unable to generate JSON export. Your analysis is still available on screen.")
        return False
    
# Function for voice cloning and enhancement
def clone_voice_and_enhance(audio_file_path, text, selected_diction):
    """Clone user's voice and enhance speech expressiveness."""
    try:
        # Import required libraries
        import os
        import requests
        import tempfile
        import time
        import json
        import streamlit as st
        
        # Step 1: Set API configuration - this should be stored in environment variables
        api_key = os.environ.get("ELEVENLABS_API_KEY") 
        
        if not api_key:
            st.warning("Voice cloning requires an ElevenLabs API key. Please set it in your environment variables.")
            st.info("For testing purposes, you can still proceed with the analysis without voice cloning.")
            return None
            
        # Step 2: Create a voice model from the user's audio
        with open(audio_file_path, "rb") as audio_file:
            # Prepare files and headers for the API request
            files = {
                "file": (f"tonecoach_recording_{int(time.time())}.wav", audio_file, "audio/wav")
            }
            
            # Include metadata for the voice
            data = {
                "name": f"ToneCoach_User_{int(time.time())}",
                "description": f"Voice created for ToneCoach {selected_diction} style enhancement"
            }
            
            headers = {
                "xi-api-key": api_key
            }
            
            # Make API request to create voice
            voice_response = requests.post(
                "https://api.elevenlabs.io/v1/voices/add",
                headers=headers,
                files=files,
                data=data
            )
            
            # Handle API response
            if voice_response.status_code != 200:
                st.error(f"Error creating voice model: {voice_response.text}")
                return None
                
            # Extract voice ID from response
            voice_data = voice_response.json()
            voice_id = voice_data.get("voice_id")
            if not voice_id:
                st.error("Failed to retrieve voice ID from API response")
                return None

        # Step 3: Configure voice settings based on diction type
        stability = 0.5  # Default: balanced stability
        similarity_boost = 0.75  # Default: maintain voice identity
        
        # Adjust settings based on selected diction type
        if selected_diction == "Storytelling":
            stability = 0.3  # Less stability for more variation in storytelling
            similarity_boost = 0.8  # Higher similarity to maintain character
        elif selected_diction == "Professional Speaking":
            stability = 0.7  # More stability for professional speaking
            similarity_boost = 0.7  # Balance between identity and clarity
        elif selected_diction == "News Anchoring":
            stability = 0.8  # High stability for consistent delivery
            similarity_boost = 0.6  # Less similarity for clarity
        elif selected_diction == "Casual Conversation":
            stability = 0.4  # Less stability for natural variation
            similarity_boost = 0.8  # Higher similarity for authenticity
            
        # Step 4: Generate enhanced speech with the voice model
        tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        
        # Payload for the TTS request
        tts_payload = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity_boost
            }
        }
        
        # Make API request for text-to-speech
        tts_response = requests.post(
            tts_url,
            headers=headers,
            json=tts_payload
        )
        
        # Handle API response
        if tts_response.status_code != 200:
            st.error(f"Error generating enhanced speech: {tts_response.text}")
            return None
            
        # Save the enhanced audio to a temporary file
        enhanced_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        enhanced_temp_file.write(tts_response.content)
        enhanced_temp_file.close()
        
        # Return the path to the temporary file
        return enhanced_temp_file.name
        
    except Exception as e:
        st.error(f"Error in voice cloning process: {str(e)}")
        return None

# Main content area
col1, col2 = st.columns([3, 2])

with col1:
    st.header("Record Your Speech")
    st.markdown(f"**Selected phrase:** *{selected_phrase}*")
    
    # Create a button to start recording
    if 'recording' not in st.session_state:
        st.session_state.recording = False
        st.session_state.audio_data = None
        st.session_state.sample_rate = 44100  # Standard sample rate
        st.session_state.duration = 10  # Max recording duration in seconds
        st.session_state.temp_file_path = None
        st.session_state.analysis_complete = False
        st.session_state.transcription = None
        st.session_state.enhanced_audio = None
    
    # Recording function
    def record_audio():
        st.session_state.recording = True
        st.session_state.audio_data = sd.rec(
            int(st.session_state.duration * st.session_state.sample_rate),
            samplerate=st.session_state.sample_rate,
            channels=1
        )
    
    # Stop recording function
    def stop_recording():
        sd.stop()
        st.session_state.recording = False
        
        # Save to temporary file
        if st.session_state.audio_data is not None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            wav.write(temp_file.name, st.session_state.sample_rate, st.session_state.audio_data)
            st.session_state.temp_file_path = temp_file.name

    # UI for recording
    if not st.session_state.recording and not st.session_state.temp_file_path:
        if st.button("Start Recording", key="start_recording"):
            record_audio()
    
    if st.session_state.recording:
        st.warning("Recording in progress... Speak now!")
        stop_button = st.button("Stop Recording")
        if stop_button:
            stop_recording()
    
    # Display recorded audio if available
    if st.session_state.temp_file_path:
        st.success("Recording complete! Playback:")
        st.audio(st.session_state.temp_file_path)
        
        if st.button("Analyze My Speech"):
            with st.spinner("Analyzing your speech patterns..."):
                # In a full implementation, this would call the speech-to-text API
                st.session_state.transcription = selected_phrase  # Using selected phrase as a placeholder
                
                # Voice cloning and enhancement (if enabled)
                if voice_clone_option:
                    st.session_state.enhanced_audio = clone_voice_and_enhance(
                        st.session_state.temp_file_path, 
                        st.session_state.transcription or selected_phrase,
                        selected_diction
                    )
                
                # Simulate analysis with a delay
                time.sleep(2)
                
                # Set flag for completed analysis
                st.session_state.analysis_complete = True
                st.rerun()
        
        if st.button("Record Again"):
            # Reset the session state
            st.session_state.recording = False
            st.session_state.audio_data = None
            if st.session_state.temp_file_path and os.path.exists(st.session_state.temp_file_path):
                os.unlink(st.session_state.temp_file_path)
            st.session_state.temp_file_path = None
            st.session_state.analysis_complete = False
            st.session_state.transcription = None
            st.session_state.enhanced_audio = None
            st.rerun()

# Analysis results display
with col2:
    st.header("Speech Analysis")
    
    if st.session_state.analysis_complete and st.session_state.temp_file_path:
        # Load audio file for analysis
        try:
            y, sr = librosa.load(st.session_state.temp_file_path, sr=None)
            
            # Perform analyses
            pitch_analysis = analyze_pitch(y, sr)
            energy_analysis = analyze_energy(y, sr)
            tempo_analysis = analyze_tempo(y, sr)
            
            # Generate feedback
            feedback_items = generate_feedback(pitch_analysis, energy_analysis, tempo_analysis)
            
            # Display transcription if available
            if st.session_state.transcription:
                st.subheader("Transcription")
                st.text(st.session_state.transcription)
            
            # Create visualizations
            # Pitch visualization
            if pitch_analysis["values"]:
                fig_pitch = go.Figure()
                fig_pitch.add_trace(go.Scatter(
                    y=pitch_analysis["values"], 
                    mode='lines',
                    name='Pitch',
                    line=dict(color='blue', width=2)
                ))
                fig_pitch.update_layout(
                    title="Pitch Variation Over Time",
                    xaxis_title="Time",
                    yaxis_title="Frequency (Hz)",
                    height=250
                )
                st.plotly_chart(fig_pitch, use_container_width=True)
            
            # Energy visualization
            if energy_analysis["values"]:
                fig_energy = go.Figure()
                fig_energy.add_trace(go.Scatter(
                    y=energy_analysis["values"], 
                    mode='lines',
                    name='Energy',
                    line=dict(color='green', width=2)
                ))
                fig_energy.update_layout(
                    title="Speech Energy Over Time",
                    xaxis_title="Time",
                    yaxis_title="Energy",
                    height=250
                )
                st.plotly_chart(fig_energy, use_container_width=True)
            
            # Display metrics
            col_met1, col_met2, col_met3 = st.columns(3)
            
            # Calculate expressiveness score (simplified formula)
            pitch_score = min(1.0, pitch_analysis["std"] / 30.0) if pitch_analysis["values"] else 0.4
            energy_score = min(1.0, energy_analysis["variation_ratio"] * 3.0) if energy_analysis["values"] else 0.5
            pacing_score = 0.8 if tempo_analysis["pacing_status"] == "good" else 0.5
            expressiveness_score = 0.75 * pitch_score + 0.15 * energy_score + 0.1 * pacing_score
            
            with col_met1:
                st.metric("Avg. Pitch (Hz)", f"{pitch_analysis['mean']:.1f}" if pitch_analysis['values'] else "N/A")
            with col_met2:
                st.metric("Speech Tempo", f"{tempo_analysis['bpm']:.1f} BPM")
            with col_met3:
                st.metric("Expressiveness", f"{expressiveness_score:.2f}")
            
            # Feedback based on diction type
            st.subheader("Your Speech Feedback")
            
            # Show personalized feedback
            for item in feedback_items:
                st.info(item)
            
            # Enhanced speech playback section
            if voice_clone_option:
                st.subheader("Enhanced Speech Example")
                
                if st.session_state.enhanced_audio:
                    st.markdown("Listen to how your phrase could sound with improved expressiveness:")
                    st.text("AI-enhanced version in your voice:")
                    st.audio(st.session_state.enhanced_audio)
                    st.info("Compare this enhanced version with your original recording to hear the differences in expressiveness.")
                else:
                    st.info("Voice cloning is enabled but couldn't generate an enhanced speech sample. This could be due to API limitations or network issues.")
            
            # Export option
            if export_option:
                st.subheader("Export Analysis Data")
                if export_analysis_data(pitch_analysis, energy_analysis, tempo_analysis, 
                                        selected_diction, selected_phrase, 
                                        expressiveness_score, feedback_items):
                    st.success("Analysis data exported successfully!")
                else:
                    st.error("Failed to export analysis data.")
            
        except Exception as e:
            st.error(f"Error analyzing audio: {str(e)}")

# Extra information in expander
with st.expander("Tips for Improving Your Speech"):
    st.markdown("""
    ### General Tips for Better Speech Delivery
    
    1. **Practice breath control**: Proper breathing supports your voice and helps control pacing.
    
    2. **Record yourself regularly**: Compare your progress over time to see improvement.
    
    3. **Focus on one aspect at a time**: First work on pitch, then pacing, then expressiveness.
    
    4. **Listen to experts**: Study speakers who excel in your chosen diction style.
    
    5. **Stay hydrated**: Drink water before speaking to keep your vocal cords in good condition.
    """)

# Footer
st.markdown("---")
st.markdown(
    "ToneCoach: Emotional Intelligence for Public Speaking | Developed by Speech AI Team"
)