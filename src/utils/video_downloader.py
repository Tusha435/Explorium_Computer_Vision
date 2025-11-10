"""
Utility to download sample test videos.
"""
import os
import requests
from tqdm import tqdm
import gdown


class VideoDownloader:
    """Download sample videos for testing."""

    # Sample video URLs (royalty-free traffic videos)
    SAMPLE_VIDEOS = {
        'traffic1': {
            'url': 'https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4',
            'filename': 'traffic_sample_1.mp4',
            'description': 'Traffic scene with multiple vehicles'
        },
        'traffic2': {
            'url': 'https://www.sample-videos.com/video123/mp4/720/big_buck_bunny_720p_2mb.mp4',
            'filename': 'traffic_sample_2.mp4',
            'description': 'Highway traffic scene'
        },
        'pedestrian': {
            'url': 'https://www.sample-videos.com/video123/mp4/720/big_buck_bunny_720p_5mb.mp4',
            'filename': 'pedestrian_sample.mp4',
            'description': 'Pedestrians and vehicles'
        }
    }

    def __init__(self, output_dir='sample_videos'):
        """
        Initialize video downloader.

        Args:
            output_dir (str): Directory to save downloaded videos
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def download_from_url(self, url, filename):
        """
        Download a video from URL.

        Args:
            url (str): Video URL
            filename (str): Output filename

        Returns:
            str: Path to downloaded video
        """
        filepath = os.path.join(self.output_dir, filename)

        if os.path.exists(filepath):
            print(f"✓ Video already exists: {filepath}")
            return filepath

        print(f"Downloading {filename}...")

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(filepath, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            print(f"✓ Downloaded: {filepath}")
            return filepath

        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return None

    def download_from_gdrive(self, file_id, filename):
        """
        Download a video from Google Drive.

        Args:
            file_id (str): Google Drive file ID
            filename (str): Output filename

        Returns:
            str: Path to downloaded video
        """
        filepath = os.path.join(self.output_dir, filename)

        if os.path.exists(filepath):
            print(f"✓ Video already exists: {filepath}")
            return filepath

        print(f"Downloading {filename} from Google Drive...")

        try:
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, filepath, quiet=False)
            print(f"✓ Downloaded: {filepath}")
            return filepath

        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return None

    def download_sample_video(self, video_key):
        """
        Download a predefined sample video.

        Args:
            video_key (str): Key from SAMPLE_VIDEOS dict

        Returns:
            str: Path to downloaded video
        """
        if video_key not in self.SAMPLE_VIDEOS:
            print(f"✗ Unknown video key: {video_key}")
            print(f"Available keys: {list(self.SAMPLE_VIDEOS.keys())}")
            return None

        video_info = self.SAMPLE_VIDEOS[video_key]
        print(f"Description: {video_info['description']}")

        return self.download_from_url(video_info['url'], video_info['filename'])

    def list_sample_videos(self):
        """List all available sample videos."""
        print("\nAvailable Sample Videos:")
        print("=" * 60)
        for key, info in self.SAMPLE_VIDEOS.items():
            print(f"\nKey: {key}")
            print(f"  File: {info['filename']}")
            print(f"  Description: {info['description']}")
        print("=" * 60)

    def generate_synthetic_video(self, filename='synthetic_traffic.mp4', duration=10, fps=30):
        """
        Generate a synthetic test video with moving objects.

        Args:
            filename (str): Output filename
            duration (int): Video duration in seconds
            fps (int): Frames per second

        Returns:
            str: Path to generated video
        """
        import cv2
        import numpy as np

        filepath = os.path.join(self.output_dir, filename)

        print(f"Generating synthetic video: {filename}")

        width, height = 1280, 720
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))

        num_frames = duration * fps

        for frame_idx in range(num_frames):
            # Create frame
            frame = np.ones((height, width, 3), dtype=np.uint8) * 50

            # Draw road
            cv2.rectangle(frame, (0, height//2 - 100), (width, height//2 + 100), (70, 70, 70), -1)

            # Draw lane markings
            for x in range(0, width, 100):
                cv2.rectangle(frame, (x, height//2 - 5), (x + 50, height//2 + 5), (255, 255, 255), -1)

            # Draw moving car
            car_x = int((frame_idx / num_frames) * width)
            car_y = height // 2
            cv2.rectangle(frame, (car_x, car_y - 30), (car_x + 80, car_y + 30), (0, 0, 255), -1)

            # Draw moving person
            person_x = int((frame_idx / num_frames) * width * 0.7)
            person_y = height // 2 + 150
            cv2.circle(frame, (person_x, person_y), 20, (255, 0, 0), -1)

            # Add frame number
            cv2.putText(frame, f"Frame: {frame_idx}/{num_frames}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            writer.write(frame)

        writer.release()
        print(f"✓ Generated synthetic video: {filepath}")

        return filepath
