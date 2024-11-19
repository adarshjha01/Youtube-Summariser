
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
import re
import torch
import gradio as gr
from transformers import pipeline

# Initialize summarization pipeline
text_summary = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6",
                        torch_dtype=torch.bfloat16, device=0)

def summarize_text(input_text):
    """Summarizes the given text."""
    if not input_text.strip():
        return "No transcript available to summarize."
    output = text_summary(input_text, max_length=200, min_length=30, do_sample=False)
    return output[0]['summary_text']

def get_video_id(url):
    """Extracts video ID from a YouTube URL."""
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
    if not match:
        raise ValueError("Invalid YouTube URL")
    return match.group(1)

def fetch_transcript(video_id):
    """
    Fetches the transcript of the YouTube video in English or Hindi.
    Returns the transcript text and the language.
    """
    languages_to_try = [["en"], ["hi"]]
    for language_codes in languages_to_try:
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=language_codes)
            formatter = TextFormatter()
            return formatter.format_transcript(transcript), language_codes[0]
        except Exception as e:
            continue  # Try the next language in the list
    raise ValueError("Transcript not available in English or Hindi.")

def process_youtube_url(url):
    """Fetches the transcript from the YouTube URL and summarizes it."""
    try:
        video_id = get_video_id(url)
        transcript, language = fetch_transcript(video_id)
        language_name = "English" if language == "en" else "Hindi"
        summary = summarize_text(transcript)
        return f"Transcript Language: {language_name}\n\nSummary:\n{summary}"
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"An unexpected error occurred: {e}"

# Gradio Interface
demo = gr.Interface(
    fn=process_youtube_url,
    inputs=[gr.Textbox(label="Enter YouTube URL", placeholder="https://www.youtube.com/watch?v=example")],
    outputs=[gr.Textbox(label="Summary")],
    title="YouTube Video Transcript Summarizer",
    description="Enter a YouTube video URL to generate it's summary. The system will first attempt to fetch the transcript in English, then Hindi if English is unavailable."
)

if __name__ == "__main__":
    demo.launch()