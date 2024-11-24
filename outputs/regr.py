from moviepy.editor import VideoFileClip, concatenate_videoclips
output_path = "IMAGINE_GEN_GENERATED_VIDEO.mp4"

def stitch_videos():
    """
    Merges an array of video files into a single video.

    Parameters:
    video_paths (list): List of paths to the video files to merge.
    output_path (str): Path to save the merged video.
    """
    try:
        # Load all video clips
        clips = [VideoFileClip(video) for video in ["output_0_cfgs_0.mp4"]*3]
        
        # Concatenate the video clips
        final_clip = concatenate_videoclips(clips, method="compose")
        
        # Write the result to a file
        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
        
        print(f"Video successfully merged and saved to {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

stitch_videos()