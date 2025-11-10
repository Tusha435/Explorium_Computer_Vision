"""
Download sample videos and images for testing object detection models.
Includes 10 videos and 10 images with traffic, people, vehicles, and urban scenes.
"""
import os
import requests
from pathlib import Path
from tqdm import tqdm
import argparse


class SampleDownloader:
    """Download sample videos and images for object detection testing."""

    # 10 Sample Videos - Traffic, vehicles, people, urban scenes
    SAMPLE_VIDEOS = {
        'traffic_highway_1': {
            'url': 'https://www.pexels.com/download/video/2103099/',
            'filename': 'traffic_highway_busy.mp4',
            'description': 'Busy highway traffic with multiple vehicles'
        },
        'city_street_1': {
            'url': 'https://www.pexels.com/download/video/3044127/',
            'filename': 'city_street_pedestrians.mp4',
            'description': 'City street with pedestrians and cars'
        },
        'parking_lot_1': {
            'url': 'https://www.pexels.com/download/video/4009538/',
            'filename': 'parking_lot_vehicles.mp4',
            'description': 'Parking lot with various vehicles'
        },
        'bicycle_riders_1': {
            'url': 'https://www.pexels.com/download/video/4728688/',
            'filename': 'bicycle_riders_street.mp4',
            'description': 'Bicycle riders on city street'
        },
        'motorcycle_traffic_1': {
            'url': 'https://www.pexels.com/download/video/3571264/',
            'filename': 'motorcycle_traffic.mp4',
            'description': 'Motorcycles in traffic'
        },
        'bus_station_1': {
            'url': 'https://www.pexels.com/download/video/4009550/',
            'filename': 'bus_station_people.mp4',
            'description': 'Bus station with people and buses'
        },
        'crosswalk_1': {
            'url': 'https://www.pexels.com/download/video/2103096/',
            'filename': 'crosswalk_pedestrians.mp4',
            'description': 'Pedestrians crossing street'
        },
        'traffic_intersection_1': {
            'url': 'https://www.pexels.com/download/video/1721294/',
            'filename': 'traffic_intersection.mp4',
            'description': 'Busy traffic intersection'
        },
        'highway_aerial_1': {
            'url': 'https://www.pexels.com/download/video/2491284/',
            'filename': 'highway_aerial_view.mp4',
            'description': 'Aerial view of highway traffic'
        },
        'urban_traffic_1': {
            'url': 'https://www.pexels.com/download/video/4033121/',
            'filename': 'urban_traffic_mixed.mp4',
            'description': 'Urban traffic with mixed vehicles'
        }
    }

    # 10 Sample Images - Various traffic and urban scenes
    SAMPLE_IMAGES = {
        'street_traffic_1': {
            'url': 'https://images.pexels.com/photos/290595/pexels-photo-290595.jpeg',
            'filename': 'street_traffic_cars.jpg',
            'description': 'Street with multiple cars'
        },
        'pedestrian_crossing_1': {
            'url': 'https://images.pexels.com/photos/450062/pexels-photo-450062.jpeg',
            'filename': 'pedestrian_crossing.jpg',
            'description': 'People crossing street'
        },
        'parking_cars_1': {
            'url': 'https://images.pexels.com/photos/164634/pexels-photo-164634.jpeg',
            'filename': 'parking_lot_cars.jpg',
            'description': 'Cars in parking lot'
        },
        'bicycle_street_1': {
            'url': 'https://images.pexels.com/photos/100582/pexels-photo-100582.jpeg',
            'filename': 'bicycle_on_street.jpg',
            'description': 'Bicycle on city street'
        },
        'motorcycle_road_1': {
            'url': 'https://images.pexels.com/photos/1130880/pexels-photo-1130880.jpeg',
            'filename': 'motorcycle_rider.jpg',
            'description': 'Motorcycle on road'
        },
        'bus_city_1': {
            'url': 'https://images.pexels.com/photos/1098365/pexels-photo-1098365.jpeg',
            'filename': 'city_bus.jpg',
            'description': 'City bus on street'
        },
        'truck_highway_1': {
            'url': 'https://images.pexels.com/photos/1335077/pexels-photo-1335077.jpeg',
            'filename': 'truck_on_highway.jpg',
            'description': 'Truck on highway'
        },
        'people_sidewalk_1': {
            'url': 'https://images.pexels.com/photos/1519088/pexels-photo-1519088.jpeg',
            'filename': 'people_walking_sidewalk.jpg',
            'description': 'People walking on sidewalk'
        },
        'intersection_aerial_1': {
            'url': 'https://images.pexels.com/photos/313782/pexels-photo-313782.jpeg',
            'filename': 'intersection_aerial.jpg',
            'description': 'Aerial view of intersection'
        },
        'mixed_traffic_1': {
            'url': 'https://images.pexels.com/photos/2664216/pexels-photo-2664216.jpeg',
            'filename': 'mixed_traffic_scene.jpg',
            'description': 'Mixed traffic with various vehicles'
        }
    }

    def __init__(self, video_dir='sample_videos', image_dir='sample_images'):
        """
        Initialize sample downloader.

        Args:
            video_dir (str): Directory for sample videos
            image_dir (str): Directory for sample images
        """
        self.video_dir = Path(video_dir)
        self.image_dir = Path(image_dir)

        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)

    def download_file(self, url, filepath, description=''):
        """
        Download a file from URL with progress bar.

        Args:
            url (str): Download URL
            filepath (Path): Destination file path
            description (str): File description

        Returns:
            bool: True if successful, False otherwise
        """
        if filepath.exists():
            print(f"✓ Already exists: {filepath.name}")
            return True

        print(f"\nDownloading: {filepath.name}")
        if description:
            print(f"  Description: {description}")

        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(filepath, 'wb') as f, tqdm(
                desc=filepath.name,
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
            return True

        except Exception as e:
            print(f"✗ Failed to download {filepath.name}: {e}")
            if filepath.exists():
                filepath.unlink()
            return False

    def download_videos(self, video_keys=None):
        """
        Download sample videos.

        Args:
            video_keys (list): List of video keys to download (None for all)

        Returns:
            dict: Download results
        """
        print(f"\n{'='*70}")
        print("DOWNLOADING SAMPLE VIDEOS")
        print(f"{'='*70}")

        if video_keys is None:
            video_keys = list(self.SAMPLE_VIDEOS.keys())

        results = {'successful': [], 'failed': []}

        for key in video_keys:
            if key not in self.SAMPLE_VIDEOS:
                print(f"✗ Unknown video key: {key}")
                continue

            video_info = self.SAMPLE_VIDEOS[key]
            filepath = self.video_dir / video_info['filename']

            success = self.download_file(
                video_info['url'],
                filepath,
                video_info['description']
            )

            if success:
                results['successful'].append(filepath)
            else:
                results['failed'].append(key)

        self._print_summary("VIDEOS", results)
        return results

    def download_images(self, image_keys=None):
        """
        Download sample images.

        Args:
            image_keys (list): List of image keys to download (None for all)

        Returns:
            dict: Download results
        """
        print(f"\n{'='*70}")
        print("DOWNLOADING SAMPLE IMAGES")
        print(f"{'='*70}")

        if image_keys is None:
            image_keys = list(self.SAMPLE_IMAGES.keys())

        results = {'successful': [], 'failed': []}

        for key in image_keys:
            if key not in self.SAMPLE_IMAGES:
                print(f"✗ Unknown image key: {key}")
                continue

            image_info = self.SAMPLE_IMAGES[key]
            filepath = self.image_dir / image_info['filename']

            success = self.download_file(
                image_info['url'],
                filepath,
                image_info['description']
            )

            if success:
                results['successful'].append(filepath)
            else:
                results['failed'].append(key)

        self._print_summary("IMAGES", results)
        return results

    def download_all(self):
        """Download all sample videos and images."""
        video_results = self.download_videos()
        image_results = self.download_images()

        print(f"\n{'='*70}")
        print("DOWNLOAD COMPLETE!")
        print(f"{'='*70}")
        print(f"Videos downloaded: {len(video_results['successful'])}/10")
        print(f"Images downloaded: {len(image_results['successful'])}/10")
        print(f"{'='*70}\n")

        return video_results, image_results

    def list_available(self):
        """List all available sample media."""
        print(f"\n{'='*70}")
        print("AVAILABLE SAMPLE VIDEOS (10)")
        print(f"{'='*70}")
        for i, (key, info) in enumerate(self.SAMPLE_VIDEOS.items(), 1):
            print(f"\n{i}. {key}")
            print(f"   File: {info['filename']}")
            print(f"   Description: {info['description']}")

        print(f"\n{'='*70}")
        print("AVAILABLE SAMPLE IMAGES (10)")
        print(f"{'='*70}")
        for i, (key, info) in enumerate(self.SAMPLE_IMAGES.items(), 1):
            print(f"\n{i}. {key}")
            print(f"   File: {info['filename']}")
            print(f"   Description: {info['description']}")

        print(f"\n{'='*70}\n")

    def _print_summary(self, media_type, results):
        """Print download summary."""
        print(f"\n{'='*70}")
        print(f"{media_type} DOWNLOAD SUMMARY")
        print(f"{'='*70}")
        print(f"Successful: {len(results['successful'])}")
        print(f"Failed: {len(results['failed'])}")

        if results['successful']:
            print(f"\nDownloaded {media_type.lower()}:")
            for path in results['successful']:
                print(f"  ✓ {path}")

        if results['failed']:
            print(f"\nFailed {media_type.lower()}:")
            for key in results['failed']:
                print(f"  ✗ {key}")

        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Download sample videos and images for object detection testing'
    )
    parser.add_argument('--videos-only', action='store_true',
                       help='Download only videos')
    parser.add_argument('--images-only', action='store_true',
                       help='Download only images')
    parser.add_argument('--list', action='store_true',
                       help='List available samples without downloading')
    parser.add_argument('--video-dir', type=str, default='sample_videos',
                       help='Directory for videos (default: sample_videos)')
    parser.add_argument('--image-dir', type=str, default='sample_images',
                       help='Directory for images (default: sample_images)')

    args = parser.parse_args()

    downloader = SampleDownloader(
        video_dir=args.video_dir,
        image_dir=args.image_dir
    )

    if args.list:
        downloader.list_available()
        return

    print(f"\n{'='*70}")
    print("SAMPLE MEDIA DOWNLOADER")
    print("Object Detection Testing Datasets")
    print(f"{'='*70}")
    print("\nThis will download:")
    print("  • 10 sample videos with traffic, vehicles, and people")
    print("  • 10 sample images with various urban scenes")
    print(f"{'='*70}\n")

    if args.videos_only:
        downloader.download_videos()
    elif args.images_only:
        downloader.download_images()
    else:
        downloader.download_all()

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Test on a video:")
    print(f"   python detect.py --video {args.video_dir}/traffic_highway_busy.mp4 --model yolov8n")
    print("\n2. Test on an image:")
    print(f"   python detect.py --image {args.image_dir}/street_traffic_cars.jpg --model yolov11n")
    print("\n3. Compare models on a video:")
    print(f"   python compare_models.py --video {args.video_dir}/city_street_pedestrians.mp4")
    print("\n4. Batch process images:")
    print(f"   python detect.py --image-dir {args.image_dir} --model yolov8s")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
