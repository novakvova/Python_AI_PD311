import yt_dlp

def download_soundcloud_audio(url, output_path='.'):
    """
    Downloads audio from a given SoundCloud URL.

    Args:
        url (str): The URL of the SoundCloud track or playlist.
        output_path (str): The directory to save the downloaded audio.
    """
    ydl_opts = {
        'format': 'bestaudio/best',  # Prioritize best audio quality
        'outtmpl': f'{output_path}/%(title)s.%(ext)s',  # Output template
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',  # Convert to MP3
            'preferredquality': '192', # Set MP3 quality
        }],
        'noplaylist': True,  # Download only the single track if it's a playlist URL
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    print(f"Downloaded audio from {url} to {output_path}")

if __name__ == "__main__":
    soundcloud_url = "https://soundcloud.com/user-277637898/yaktak-ne-pitai-mene" # Replace with your SoundCloud URL
    download_directory = "./downloads" # Specify your desired output directory

    download_soundcloud_audio(soundcloud_url, download_directory)