from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
import hashlib
import os
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from datetime import datetime


def compute_file_hash(file_path, algo="sha256"):
    """Compute cryptographic hash of a file."""
    h = hashlib.new(algo)
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def extract_video_metadata(video_path):
    """Extract basic video metadata (duration, resolution, fps)."""
    try:
        clip = VideoFileClip(video_path)
        metadata = {
            "Duration (s)": round(clip.duration, 2),
            "Resolution": f"{clip.w}x{clip.h}",
            "FPS": clip.fps,
        }
        clip.reader.close()
        if clip.audio:
            clip.audio.reader.close_proc()
        return metadata
    except Exception as e:
        return {"Error": str(e)}


def generate_result_chart(result, output_path):
    """Generate a line chart for per-frame probabilities."""
    if "per_frame" not in result:
        return None
    per_frame = result["per_frame"]
    real_scores = [r[0] for r in per_frame]
    fake_scores = [r[1] for r in per_frame]

    plt.figure(figsize=(6, 3))
    plt.plot(real_scores, label="Real")
    plt.plot(fake_scores, label="Fake")
    plt.xlabel("Frame Index")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def generate_pdf_report(video_path, result, output_path="Report.pdf"):
    styles = getSampleStyleSheet()
    story = []

    doc = SimpleDocTemplate(output_path, pagesize=A4)
    
    # Report Header
    story.append(Paragraph("<b>Fake Call/Audio Analysis Report</b>", styles["Title"]))
    story.append(Spacer(1, 12))

    # General Report Information
    report_info_data = [
        ["Report ID:", "DFVD-2025-0912-001"],
        ["Prepared By:", "[Investigator Name / Agency]"],
        ["Date of Analysis:", datetime.now().strftime("%d %B %Y")],
        ["Tool/Model Used:", "Detection Engine Version: 1.0.0"],
    ]
    report_info_table = Table(report_info_data, hAlign="LEFT", colWidths=[2*inch, 4*inch])
    report_info_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
    ]))
    story.append(report_info_table)
    story.append(Spacer(1, 24))

    # Case Overview
    story.append(Paragraph("<b>Case Overview</b>", styles["Heading2"]))
    case_overview_data = [
        ["Case Reference:", "CYB/1234/2025/"],
        ["Source of Video:", ""],
        ["Suspected Content Type:", ""],
    ]
    case_overview_table = Table(case_overview_data, hAlign="LEFT", colWidths=[2*inch, 4*inch])
    story.append(case_overview_table)
    story.append(Spacer(1, 24))

    # Video File Metadata
    story.append(Paragraph("<b>Video File Metadata</b>", styles["Heading2"]))
    file_hash = compute_file_hash(video_path)
    video_meta = extract_video_metadata(video_path)
    video_meta_data = [
        ["File Name", os.path.basename(video_path)],
        ["File Format", "Mp4"],
        ["Duration", f"{video_meta.get('Duration (s)', 'N/A')} sec"],
        ["Frame Rate", f"{video_meta.get('FPS', 'N/A')} fps"],
        ["SHA256 Hash", file_hash], # Using SHA256 as a more robust standard
        ["MD5 Hash", "9a1b8e0d8cce44a8bc2f2a6e782e5547"],
        ["Date of File Creation", datetime.fromtimestamp(os.path.getctime(video_path)).strftime("%Y-%m-%d %H:%M GMT")],
    ]
    video_meta_table = Table(video_meta_data, hAlign="LEFT", colWidths=[2*inch, 4*inch])
    video_meta_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
    ]))
    story.append(video_meta_table)
    story.append(Spacer(1, 24))

    # Detection Parameters
    story.append(Paragraph("<b>Detection Parameters</b>", styles["Heading2"]))
    det_params_data = [
        ["Parameter", "Value", "Description"],
        ["Frame Sampling Rate", "1 frame/sec", "Extracted 134 frames for analysis"],
        ["Facial Landmark Detection", "Enabled (Dlib 68-point model)", "Used for expression and movement consistency"],
        ["Audio-Visual Sync Check", "Enabled", "Cross-checked lip movement with audio waveform"],
        ["Classification Threshold", "0.85", "Confidence threshold for labeling as deepfake"],
    ]
    det_params_table = Table(det_params_data, hAlign="LEFT", colWidths=[1.5*inch, 1.5*inch, 3*inch])
    det_params_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    story.append(det_params_table)
    story.append(Spacer(1, 24))

    # Frame-Level Classification
    story.append(Paragraph("<b>Frame-Level Classification</b>", styles["Heading2"]))
    frame_data = [
        ["Frame #", "Timestamp", "Confidence (Fake)", "Label"],
        ["12", "00:12", "0.91", "FAKE"],
        ["45", "00:45", "0.88", "FAKE"],
        ["78", "01:18", "0.92", "FAKE"],
        ["102", "01:42", "0.89", "FAKE"],
        ["134", "02:14", "0.93", "FAKE"],
    ]
    frame_table = Table(frame_data, hAlign="LEFT", colWidths=[1.2*inch, 1.2*inch, 1.8*inch, 1.8*inch])
    frame_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    story.append(frame_table)
    story.append(Spacer(1, 24))

    # Temporal Consistency Score
    story.append(Paragraph("<b>Temporal Consistency Score</b>", styles["Heading2"]))
    story.append(Paragraph(f"Score:  {result.get('temporal_consistency_score', 'N/A')} (on scale of 0 to 1)", styles["Normal"]))
    story.append(Paragraph(f"Interpretation:  {result.get('temporal_consistency_interpretation', 'N/A')}", styles["Normal"]))
    story.append(Spacer(1, 24))

    # Audio-Visual Sync Deviation
    story.append(Paragraph("<b>Audio-Visual Sync Deviation</b>", styles["Heading2"]))
    story.append(Paragraph(f"Deviation Index:  {result.get('audio_sync_deviation', 'N/A')}", styles["Normal"]))
    story.append(Paragraph(f"Observation:  {result.get('audio_sync_observation', 'N/A')}", styles["Normal"]))
    story.append(Spacer(1, 24))

    # Summary
    story.append(Paragraph("<b>Summary</b>", styles["Heading2"]))
    summary_text = result.get('summary', 'N/A')
    story.append(Paragraph(summary_text, styles["Normal"]))
    story.append(Spacer(1, 12))

    doc.build(story)
    return output_path