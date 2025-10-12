# Meeting Summarizer 🎙️

A complete AI-powered solution for transcribing audio files and generating intelligent summaries using OpenAI Whisper and Google Gemini.

## Features

- 🎯 **Audio Transcription** - Convert speech to text using OpenAI Whisper
- 🤖 **AI Summarization** - Generate bullet-point summaries using Google Gemini
- 🖥️ **Command Line Interface** - Simple script execution
- 🌐 **Web Dashboard** - Interactive Streamlit interface
- 📁 **Multiple Formats** - Support for MP3, WAV, M4A, FLAC, OGG
- 💾 **Export Options** - Download transcript and summary as text files

## Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/HarshithMandi/meeting_summarizer.git
cd meeting_summarizer
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env and add your Google Gemini API key
```

### 3. Install FFmpeg
- **Windows**: `winget install ffmpeg`
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg`

### 4. Run the Application

#### Option A: Command Line
```bash
python meeting_summarizer.py
```

#### Option B: Web Dashboard
```bash
streamlit run streamlit_app.py
```
Then open http://localhost:8501 in your browser.

## Files Structure

### Core Files
- **`meeting_summarizer.py`** - Complete standalone script
- **`streamlit_app.py`** - Interactive web dashboard
- **`requirements.txt`** - Python dependencies
- **`.env.example`** - Environment variables template

### Configuration
- **`.gitignore`** - Git ignore rules
- **`README.md`** - This documentation

## Requirements

### Python Packages
- `openai-whisper` - Audio transcription
- `google-generativeai` - AI summarization
- `python-dotenv` - Environment variables
- `streamlit` - Web interface
- `torch` - Machine learning backend

### External Dependencies
- **FFmpeg** - Audio processing
- **Google Gemini API Key** - Get from [Google AI Studio](https://aistudio.google.com/)

## API Setup

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Create a new API key
3. Copy the key to your `.env` file:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```

## Usage Examples

### Command Line
Place your audio file in the project directory and update the filename in `meeting_summarizer.py`, then run:
```bash
python meeting_summarizer.py
```

### Web Interface
1. Start the dashboard: `streamlit run streamlit_app.py`
2. Upload an audio file through the web interface
3. Click "Process Audio"
4. View results and download files

## Troubleshooting

### FFmpeg Issues
If you get "FFmpeg not found" errors:
1. Install FFmpeg using the instructions above
2. Restart your terminal/command prompt
3. Verify installation: `ffmpeg -version`

### API Key Issues
- Ensure your `.env` file exists and contains the correct API key
- Verify the API key is valid in Google AI Studio
- Check for any trailing spaces or quotes in the `.env` file

## License

MIT License - see LICENSE file for details

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request