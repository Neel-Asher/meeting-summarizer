import whisper
import google.generativeai as genai
import dotenv
import os
import datetime

dotenv.load_dotenv()

print("=== MEETING SUMMARIZER ===")
print("Step 1: Audio Transcription")
print("-" * 40)

# Initialize Whisper
print('Loading Whisper model...')
print('Whisper version ok, torch version:', __import__('torch').__version__)
model = whisper.load_model('base')
print('Loaded Whisper model:', type(model))

# Transcribe audio file
audio_file = "sample-meeting.mp3"
print(f"Transcribing audio file: {audio_file}")
transcript_result = model.transcribe(audio_file, fp16=False)
transcript_text = transcript_result['text']
print(f"Transcription complete: {transcript_text}")

# Save transcript to file
transcript_filename = "transcript.txt"
with open(transcript_filename, 'w', encoding='utf-8') as file:
    file.write(f"Meeting Transcript\n")
    file.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    file.write(f"Audio file: {audio_file}\n")
    file.write("-" * 50 + "\n\n")
    file.write(transcript_text)
    file.write("\n")

print(f"Transcript saved to: {transcript_filename}")

print("\nStep 2: AI Summary Generation")
print("-" * 40)

# Get API key from environment variable
API_key = os.getenv('GEMINI_API_KEY')
if not API_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

# Configure Gemini AI
genai.configure(api_key=API_key)
ai_model = genai.GenerativeModel('gemini-2.5-flash')
print("Gemini AI model loaded successfully")

# Create prompt for summary
prompt = f"""You are an expert note taker, using the following transcript of a meeting generate a bullet point formatted summary. Do not speculate, use details from the given transcript alone. 

Transcript: [{transcript_text}]"""

# Generate summary
print("Generating AI summary...")
response = ai_model.generate_content(prompt)
summary_text = response.text
print(f"Summary generated: {summary_text}")

print("\nStep 3: Creating Combined Output")
print("-" * 40)

# Create output.txt with both transcript and summary
output_filename = "output.txt"
with open(output_filename, 'w', encoding='utf-8') as file:
    # Write header with timestamp
    file.write(f"Meeting Transcript and Summary\n")
    file.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    file.write(f"Audio file: {audio_file}\n")
    file.write("=" * 60 + "\n\n")
    
    # Write original transcript section
    file.write("ORIGINAL TRANSCRIPT:\n")
    file.write("-" * 30 + "\n")
    file.write(transcript_text)
    file.write("\n\n")
    
    # Write summary section
    file.write("AI-GENERATED SUMMARY:\n")
    file.write("-" * 30 + "\n")
    file.write(summary_text)
    file.write("\n")

print(f"✅ Complete! Both transcript and summary saved to: {output_filename}")
print(f"✅ Individual transcript also saved to: {transcript_filename}")
print("\n=== PROCESS COMPLETED SUCCESSFULLY ===")