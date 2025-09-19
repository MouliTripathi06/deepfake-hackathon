# Modified data_processor.py for proper data split
import cv2
import os
import random
from pipeline import extract_faces_from_video

def process_video_dataset_with_split(source_folder, train_output, val_output, is_fake, split_ratio=0.8):
    """
    Processes videos, extracts faces, and splits them into training and validation sets.
    """
    video_files = [f for f in os.listdir(source_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
    random.shuffle(video_files)  # Shuffle the files for a random split

    train_videos = video_files[:int(len(video_files) * split_ratio)]
    val_videos = video_files[int(len(video_files) * split_ratio):]

    # Process training videos
    _process_and_save(train_videos, source_folder, train_output, is_fake)

    # Process validation videos
    _process_and_save(val_videos, source_folder, val_output, is_fake)

def _process_and_save(video_list, source_folder, output_base_folder, is_fake):
    label_folder = 'fake' if is_fake else 'real'
    output_path = os.path.join(output_base_folder, label_folder)
    os.makedirs(output_path, exist_ok=True)

    for video_name in video_list:
        video_path = os.path.join(source_folder, video_name)
        print(f"Processing {video_path}...")
        face_images = extract_faces_from_video(video_path)
        
        if face_images:
            for i, face in enumerate(face_images):
                image_name = f"{os.path.splitext(video_name)[0]}_{i:04d}.jpg"
                image_path = os.path.join(output_path, image_name)
                cv2.imwrite(image_path, face)
        else:
            print(f"No faces found in {video_name}. Skipping...")

if __name__ == '__main__':
    # Set your actual paths here
    real_videos_source = 'C:/Users/compu/Downloads/SDFVD/SDFVD/videos_real'
    fake_videos_source = 'C:/Users/compu/Downloads/SDFVD/SDFVD/videos_fake'
    
    train_output = 'C:/Users/compu/OneDrive/doc/deepfake_video/deepfake-hack/data/train'
    val_output = 'C:/Users/compu/OneDrive/doc/deepfake_video/deepfake-hack/data/val'

    # Process and save with an 80/20 train/val split
    process_video_dataset_with_split(real_videos_source, train_output, val_output, is_fake=False)
    process_video_dataset_with_split(fake_videos_source, train_output, val_output, is_fake=True)

    print("Data processing complete. Image dataset created.")
    
    train_output = 'C:/Users/compu/OneDrive/doc/deepfake_video/deepfake-hack/data/train'
    val_output = 'C:/Users/compu/OneDrive/doc/deepfake_video/deepfake-hack/data/val'

    # Process and save with an 80/20 train/val split
    process_video_dataset_with_split(real_videos_source, train_output, val_output, is_fake=False)
    process_video_dataset_with_split(fake_videos_source, train_output, val_output, is_fake=True)

    print("Data processing complete. Image dataset created.")