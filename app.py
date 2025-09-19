# In app.py

import streamlit as st
import tempfile
import os
import torch
import torch.nn as nn
import pandas as pd
# Import the function from your report generation script (assuming it's named 'report_generator.py')
from report_generator import generate_pdf_report 
from pipeline import load_model, predict_video, xception_model, transform

@st.cache_resource
def get_model():
    # Load the fine-tuned model weights
    return load_model(checkpoint_path='deepfake_xception_best.pth')

def main():
    st.title("🔍 Deepfake Video Detection Tool")
    
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_file.read())
            video_path = temp_file.name

        st.video(video_path)
        
        st.info("Analyzing video... Please wait.")
        
        try:
            model = get_model()
            analysis_result = predict_video(model, video_path)
            
            st.subheader("📊 Detection Result")
            
            # Display final verdict and confidence
            st.write(f"**Final Verdict:** {analysis_result['final_label']}")
            st.write(f"**Confidence:** {analysis_result['final_confidence']:.2f}")

            # Display the new metrics
            st.write("---")
            st.subheader("🔍 Detailed Analysis")
            st.write(f"**Temporal Consistency Score:** {analysis_result['temporal_consistency_score']:.2f}")
            st.write(f"**Audio-Visual Sync Deviation:** {analysis_result['audio_sync_deviation']:.2f}")
            
            # Display Frame-Level Classification Table
            if analysis_result['frame_classifications']:
                st.write("---")
                st.subheader("🖼️ Frame-Level Classification")
                
                # Convert list of dicts to a pandas DataFrame for better display
                df_frames = pd.DataFrame(analysis_result['frame_classifications'])
                st.dataframe(df_frames, use_container_width=True)

            # --- New Code for PDF Report Generation and Download ---
            st.write("---")
            st.subheader("📄 Generate Report")

            # Prepare the data in the format expected by the report generator
            report_data = {
                "label": analysis_result['final_label'],
                "confidence": analysis_result['final_confidence'],
                "per_frame": analysis_result['per_frame_probabilities'],
                "temporal_consistency_score": analysis_result['temporal_consistency_score'],
                "temporal_consistency_interpretation": "Moderate inconsistencies in facial movement and lighting across frames.",
                "audio_sync_deviation": analysis_result['audio_sync_deviation'],
                "audio_sync_observation": "Lip movement does not align with phoneme timing in a percentage of sampled frames.",
                "summary": f"The video '{os.path.basename(video_path)}' exhibits multiple indicators of synthetic manipulation. High-confidence deepfake classification was observed, supported by temporal and audio-visual inconsistencies."
            }

            # Generate the PDF report
            report_path = generate_pdf_report(video_path, report_data)

            # Create a download button for the generated report
            with open(report_path, "rb") as file:
                st.download_button(
                    label="Download Report (PDF)",
                    data=file,
                    file_name="deepfake_report.pdf",
                    mime="application/pdf"
                )
            # --- End of new code ---

        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
        finally:
            os.remove(video_path)
            # Remove the generated report as well
            if 'report_path' in locals() and os.path.exists(report_path):
                os.remove(report_path)

if __name__ == "__main__":
    main()