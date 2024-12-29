# rehab_tool.py

import streamlit as st
import os

# -- Set Page Config BEFORE any other Streamlit calls --
st.set_page_config(
    page_title="Comprehensive AI-Enhanced Rehabilitation Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🩺 Comprehensive AI-Enhanced Rehabilitation Tool for Stroke Patients")
    st.write(
        """
        **Welcome!** This integrated application assists stroke patients with both 
        **motor therapy** (hand gesture recognition), **speech therapy** (pronunciation training),
        and **telemedicine** (real-time consultations with therapists).
        
        Please select the **Therapy Module** you'd like to use below.
        """
    )

    # --- Main Page Navigation ---
    st.markdown("## Select Therapy Module")

    # Using radio buttons for navigation on the main page
    therapy_option = st.radio(
        "Choose a module:",
        ("Motor Therapy (Hand Gestures)", "Speech Therapy", "Telemedicine")
    )

    # Display instructions related to the selected module
    if therapy_option == "Motor Therapy (Hand Gestures)":
        st.markdown(
            """
            **Instructions for Motor Therapy:**
            1. Use your **webcam** to track your hand gestures.
            2. **Record** a new gesture or **use** an existing one.
            3. Receive real-time, color-coded feedback on your progress.
            
            **Tip**: Watch the rehabilitation exercise videos available in the resources section to guide you!
            """
        )
    
    elif therapy_option == "Speech Therapy":
        st.markdown(
            """
            **Instructions for Speech Therapy:**
            1. Select words from the easy, medium, or hard lists.
            2. **Record** your voice pronouncing the selected words.
            3. Receive **feedback** on correct sounds and areas for improvement.
            """
        )
    
    else:  # Telemedicine
        st.markdown(
            """
            **Instructions for Telemedicine:**
            1. Click the button below to start a video conference with your therapist.
            2. Ensure your browser has access to the webcam and microphone.
            3. Communicate with your therapist to discuss your progress and receive guidance.
            """
        )

    st.markdown("---")

    # Dynamically load motor.py, speech.py, or telemedicine module based on selection
    if therapy_option == "Motor Therapy (Hand Gestures)":
        st.subheader("🤚 Motor Therapy (Hand Gestures)")
        st.markdown(
            """
            This module uses your **webcam** to track your hand and compare it to 
            predefined gestures. **Record** a new gesture or **use** an existing one, 
            and receive real-time, color-coded feedback on your progress.
            """
        )
        if os.path.exists("motor.py"):
            with open("motor.py", "r", encoding="utf-8") as f:
                code = f.read()
            exec(code, {"__name__": "__main__"})
        else:
            st.error("`motor.py` not found in this directory.")

    elif therapy_option == "Speech Therapy":
        st.subheader("🗣️ Speech Therapy")
        st.markdown(
            """
            This module helps you practice **pronunciation** by offering words from 
            easy, medium, and hard lists. 
            **Record** your voice, then receive **feedback** on correct sounds 
            and areas for improvement.
            """
        )
        if os.path.exists("speech.py"):
            with open("speech.py", "r", encoding="utf-8") as f:
                code = f.read()
            exec(code, {"__name__": "__main__"})
        else:
            st.error("`speech.py` not found in this directory.")

    else:  # Telemedicine
        st.subheader("💬 Telemedicine")
        st.markdown(
            """
            Connect with your therapist in real-time through secure video consultations.
            Follow these steps to initiate a session:
            1. **Start** the video conference below.
            2. **Communicate** with your therapist to discuss your progress and receive guidance.
            """
        )
        # Telemedicine Video Conference
        st.markdown("### 📹 Start Your Telemedicine Session")
        st.markdown(
            """
            Click the button below to start a video conference with your therapist. 
            Ensure you have granted your browser access to the webcam and microphone.
            """
        )

        # Generate a unique room name or use a fixed one
        # For demonstration, we'll use a fixed room name. In production, consider dynamic room names.
        room_name = "RehabToolTelemedicine"
        jitsi_url = f"https://meet.jit.si/{room_name}"

        # Embed Jitsi Meet via iframe
        st.components.v1.iframe(jitsi_url, width=800, height=600)

        st.markdown(
            """
            **Note**: Ensure that both you and your therapist join the same room name to connect.
            """
        )

   
   

if __name__ == "__main__":
    main()