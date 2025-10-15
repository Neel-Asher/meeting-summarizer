import streamlit as st
import whisper
import google.generativeai as genai
import dotenv
import os
import datetime
import tempfile
import io
import subprocess

dotenv.load_dotenv()

try:
    subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    print("✅ FFmpeg is available")
except Exception as e:
    raise RuntimeError(f"FFmpeg is not found or not working: {e}")


st.set_page_config(
    page_title="Meeting Summarizer",
    page_icon="🎙️",
    layout="wide"
)

if 'transcript' not in st.session_state:
    st.session_state.transcript = ""
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'processing' not in st.session_state:
    st.session_state.processing = False

@st.cache_resource
def load_whisper_model():
    """Load Whisper model and cache it"""
    return whisper.load_model("base")

@st.cache_resource
def load_gemini_model():
    """Load Gemini model and cache it"""
    API_key = os.getenv('GEMINI_API_KEY')
    if not API_key:
        st.error("GEMINI_API_KEY not found in environment variables")
        return None
    genai.configure(api_key=API_key)
    return genai.GenerativeModel('gemini-2.5-flash')

def transcribe_audio(audio_file):
    """Transcribe audio file using Whisper"""
    try:
        model = load_whisper_model()
        
        audio_file.seek(0)
        audio_data = audio_file.read()
        
        if len(audio_data) == 0:
            raise ValueError("Audio file is empty")
        if len(audio_data) < 1024:
            raise ValueError("Audio file is too small (less than 1KB). Please check if the file is valid.")
        
        file_extension = os.path.splitext(audio_file.name)[1] if audio_file.name else ".wav"
        if not file_extension:
            file_extension = ".wav"
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(audio_data)
            tmp_file_path = tmp_file.name
        
        if not os.path.exists(tmp_file_path):
            raise FileNotFoundError(f"Temporary file not created: {tmp_file_path}")
        if os.path.getsize(tmp_file_path) == 0:
            raise ValueError("Temporary file is empty")
        
        try:
            subprocess.run([
                'ffmpeg', '-v', 'quiet', '-i', tmp_file_path, '-f', 'null', '-'
            ], capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Invalid audio file format or corrupted file. FFmpeg error: {e}")
        
        result = model.transcribe(tmp_file_path, fp16=False, verbose=False)
        transcript = result['text'].strip()
        if not transcript or len(transcript) < 3:
            raise ValueError("Transcript is too short. Audio may not contain clear speech.")
        
        return transcript
        
    except Exception as e:
        error_msg = str(e)
        if "reshape tensor" in error_msg:
            raise Exception(
                "Audio file processing failed. The file may be corrupted, empty, or in an unsupported format. Please try:\n"
                "1. Converting to WAV format\n"
                "2. Checking if the file contains actual audio content\n"
                "3. Using a different audio file"
            )
        else:
            raise Exception(f"Transcription failed: {error_msg}")
    finally:
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except:
                pass

def generate_summary(transcript):
    """Generate summary using Gemini AI"""
    model = load_gemini_model()
    if model is None:
        return "Error: Could not load Gemini model"
    
    prompt = f"""You are an expert note taker. Using the following transcript of a meeting, generate a bullet point formatted summary. Do not speculate, use details from the given transcript alone.

Transcript: [{transcript}]"""
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def download_output(transcript, summary, filename):
    """Create downloadable output file"""
    output = f"""Meeting Transcript and Summary
Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 60}

ORIGINAL TRANSCRIPT:
{'-' * 30}
{transcript}

AI-GENERATED SUMMARY:
{'-' * 30}
{summary}
"""
    return output

st.title("🎙️ Meeting Summarizer Dashboard")
st.markdown("Upload an audio file to get an AI-generated transcript and summary")

with st.sidebar:
    st.header("📋 Instructions")
    st.markdown("""
    1. **Upload** an audio file (MP3, WAV, M4A, etc.)
    2. **Wait** for transcription to complete
    3. **Review** the transcript and summary
    4. **Download** the results if needed
    
    ### Supported Formats:
    - MP3, WAV, M4A, FLAC
    - Maximum file size: 200MB
    """)
    
    st.header("⚙️ Settings")
    show_progress = st.checkbox("Show detailed progress", value=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.header("📁 Upload Audio File")
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['mp3', 'wav', 'm4a', 'flac', 'ogg'],
        help="Upload your meeting audio file for transcription"
    )
    
    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")
        st.audio(uploaded_file, format='audio/wav')
        
        if st.button("🚀 Process Audio", type="primary"):
            st.session_state.processing = True
            st.session_state.transcript = ""
            st.session_state.summary = ""
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                if show_progress:
                    status_text.text("🎯 Loading Whisper model...")
                progress_bar.progress(10)
                
                if show_progress:
                    status_text.text("🎙️ Transcribing audio...")
                progress_bar.progress(30)
                
                transcript = transcribe_audio(uploaded_file)
                st.session_state.transcript = transcript
                progress_bar.progress(60)
                
                if show_progress:
                    status_text.text("🤖 Generating AI summary...")
                progress_bar.progress(80)
                
                summary = generate_summary(transcript)
                st.session_state.summary = summary
                progress_bar.progress(100)
                
                if show_progress:
                    status_text.text("✅ Processing complete!")
                
                st.success("Processing completed successfully!")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            finally:
                st.session_state.processing = False

with col2:
    st.header("📊 Results")
    
    if st.session_state.transcript:
        st.subheader("📝 Transcript")
        with st.expander("View Transcript", expanded=True):
            st.text_area(
                "Transcript Content",
                value=st.session_state.transcript,
                height=200,
                label_visibility="collapsed"
            )
        
        if st.session_state.summary:
            st.subheader("📋 AI Summary")
            with st.expander("View Summary", expanded=True):
                st.markdown(st.session_state.summary)
            
            st.subheader("💾 Download Results")
            col_download1, col_download2 = st.columns(2)
            
            with col_download1:
                st.download_button(
                    label="📄 Download Transcript",
                    data=st.session_state.transcript,
                    file_name=f"transcript_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            
            with col_download2:
                combined_output = download_output(
                    st.session_state.transcript, 
                    st.session_state.summary,
                    uploaded_file.name if uploaded_file else "audio_file"
                )
                st.download_button(
                    label="📦 Download Combined",
                    data=combined_output,
                    file_name=f"meeting_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
    
    elif not st.session_state.processing:
        st.info("👆 Upload an audio file and click 'Process Audio' to get started!")

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        Built with ❤️ using Streamlit, OpenAI Whisper, and Google Gemini
    </div>
    """, 
    unsafe_allow_html=True
)