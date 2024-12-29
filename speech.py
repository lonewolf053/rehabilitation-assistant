import os
import random
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import speech_recognition as sr
from difflib import SequenceMatcher
from gtts import gTTS
import streamlit as st

# Words for different difficulty levels
easy_words = ["cat", "dog", "sun", "tree", "fish", "car", "ball", "hat", "pen", "book", "milk", "apple", "chair", "clock", "key"]
medium_words = ["table", "garden", "yellow", "basket", "window", "chicken", "purple", "silver", "orange", "flower", "banana", "turtle", "market", "bottle", "candle"]
hard_words = ["evaluate", "practice", "triangle", "intelligent", "umbrella", "elephant", "photograph", "important", "calendar", "technology", "butterfly", "university", "laboratory", "chocolate", "experiment"]

# Function to select one word from each difficulty level
def select_random_words():
    return [
        (random.choice(easy_words), "Easy"),
        (random.choice(medium_words), "Medium"),
        (random.choice(hard_words), "Hard"),
    ]

# Function to record audio
def record_audio(filename, duration=5, sample_rate=16000):
    st.info("🎙️ Recording... Please speak into the microphone.")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()  # Wait until the recording is finished
    write(filename, sample_rate, (audio_data * 32767).astype(np.int16))  # Save as 16-bit WAV file
    st.success("✅ Recording saved!")

# Function to transcribe audio using SpeechRecognition and Google Web Speech API
def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
    try:
        transcription = recognizer.recognize_google(audio_data)
        return transcription
    except sr.UnknownValueError:
        return "Could not understand the audio."
    except sr.RequestError as e:
        return f"Speech recognition error: {e}"

# Function to calculate percentage correctness
def calculate_percentage_correctness(target_sequence, patient_sequence):
    target_sequence = target_sequence.lower().strip()
    patient_sequence = patient_sequence.lower().strip()
    matcher = SequenceMatcher(None, target_sequence, patient_sequence)
    return matcher.ratio() * 100

# Function to provide sound-specific feedback
def analyze_sounds(target_word, patient_transcription):
    target_word = target_word.lower().strip()
    patient_transcription = patient_transcription.lower().strip()
    mismatched_sounds = [
        f"'{t_char}'" for t_char, p_char in zip(target_word, patient_transcription) if t_char != p_char
    ]
    if not mismatched_sounds:
        return "No specific sounds to improve. Great work!"
    feedback = "Focus on improving the following sounds: " + ", ".join(set(mismatched_sounds))
    return feedback

# Function to generate and play audio for the target word
def play_word(word):
    audio_file = f"{word}.mp3"
    if not os.path.exists(audio_file):
        tts = gTTS(text=word, lang="en")
        tts.save(audio_file)
    st.audio(audio_file, format="audio/mp3", start_time=0)

# Function to play the patient's attempt
def play_attempt(difficulty):
    attempt_file = f"{difficulty}_attempt.wav"
    if os.path.exists(attempt_file):
        st.audio(attempt_file, format="audio/wav", start_time=0)
    else:
        st.error(f"No recording found for {difficulty} attempt.")

# Function to generate and play audio for feedback
def play_feedback(feedback, word):
    feedback_audio = f"{word}_feedback.mp3"
    tts = gTTS(text=feedback, lang="en")
    tts.save(feedback_audio)
    st.audio(feedback_audio, format="audio/mp3", start_time=0)

# Initialize Streamlit session state
if "page" not in st.session_state:
    st.session_state.page = "main"
if "current_words" not in st.session_state:
    st.session_state.current_words = select_random_words()
if "current_word_index" not in st.session_state:
    st.session_state.current_word_index = 0
if "feedback_summary" not in st.session_state:
    st.session_state.feedback_summary = []

# Navigation between pages
if st.session_state.page == "main":
   

    current_word_index = st.session_state.current_word_index
    current_words = st.session_state.current_words
    feedback_summary = st.session_state.feedback_summary

    if current_word_index < len(current_words):
        target_word, difficulty = current_words[current_word_index]
        st.markdown(f"### 🎯 Target Word: **{target_word}**")
        st.markdown(f"**Difficulty Level:** `{difficulty}`")
    else:
        st.markdown("### ✅ Set Completed! Click 'View Feedback' to see how you can improve.")

    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        if current_word_index < len(current_words):
            if st.button("🔊 Listen to Word"):
                play_word(target_word)

    with col2:
        if st.button("🎤 Start Recording and Evaluate") and current_word_index < len(current_words):
            target_word, difficulty = current_words[current_word_index]
            attempt_file = f"{difficulty}_attempt.wav"
            record_audio(attempt_file)
            transcription = transcribe_audio(attempt_file)
            st.markdown(f"**Your Pronunciation:** `{transcription}`")
            correctness = calculate_percentage_correctness(target_word, transcription)
            st.markdown(f"**Pronunciation Correctness:** `{correctness:.2f}%`")
            feedback_summary.append({
                "word": target_word,
                "difficulty": difficulty,
                "correctness": correctness,
                "transcription": transcription,
                "feedback": analyze_sounds(target_word, transcription)
            })
            st.session_state.current_word_index += 1

    with col3:
        if st.button("📜 View Feedback") and current_word_index >= len(current_words):
            st.session_state.page = "feedback"

    st.divider()
    if current_word_index >= len(current_words):
        if st.button("🔄 Next Set"):
            st.session_state.current_words = select_random_words()
            st.session_state.current_word_index = 0
            st.session_state.feedback_summary = []

elif st.session_state.page == "feedback":
    st.title("📝 Feedback Summary")
    feedback_summary = st.session_state.feedback_summary
    if feedback_summary:
        for entry in feedback_summary:
            st.markdown(f"- **Word**: `{entry['word']}` (**{entry['difficulty']}**)")
            st.markdown(f"  - **Your Pronunciation**: `{entry['transcription']}`")
            st.markdown(f"  - **Correctness**: `{entry['correctness']:.2f}%`")
            st.markdown(f"  - **Feedback**: {entry['feedback']}")
            if st.button(f"🔊 Listen to Word - {entry['word']}", key=f"word_{entry['word']}"):
                play_word(entry['word'])
            if st.button(f"🔊 Listen to Your Attempt - {entry['difficulty']}", key=f"attempt_{entry['difficulty']}"):
                play_attempt(entry['difficulty'])
            if st.button(f"🔊 Listen to Feedback for {entry['word']}", key=f"feedback_{entry['word']}"):
                play_feedback(entry['feedback'], entry['word'])
    else:
        st.markdown("No feedback available yet.")

    if st.button("🔙 Back to Practice"):
        st.session_state.page = "main"
