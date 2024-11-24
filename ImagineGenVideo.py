import sys

import streamlit as st
from temporal_decoder_runner import run_temporal_decoder_inference
from pixelspace_runner import run_pixelspace_inference
from latent_mmnist_runner import run_latentspace_mmnist_inference
from dummy_runner import dummy_runner_inference
from simple_description_runner import run_simple_desc_inference
from moviepy import VideoFileClip, concatenate_videoclips

st.title("Welcome to ImagineGen Video generator !")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

import base64

runners_list = ["Pixel space(no vae)","Latent Space Moving MNIST","Temporal Decoder","FLUX VAE","DUMMY runner", "AE","Simple Description"]


selected_runner = st.selectbox("Select a runner based on VAE used.", tuple(runners_list))

if selected_runner == "Pixel space(no vae)":
    default_model_file = "pytorch_model_unet_cross_attn_192_temp_decoder_ddim_initial_frames_res-reduction_bw_eph_2_mmnist_easy_pixelspace.bin"
if selected_runner == "Latent Space Moving MNIST":
    default_model_file = "pytorch_model_unet_cross_attn_192_temp_decoder_ddim_initial_frames_eph_2_mmnist_easy.bin"
elif selected_runner == "Temporal Decoder":
    default_model_file = "pytorch_model_unet_cross_attn_192_temp_decoder_ddim_optical_flow_eph_8_sakuga.bin"
elif selected_runner == "Simple Description":
    default_model_file = "pytorch_model_unet_cross_attn_192_tmp_dcd_bridge_ae_simple_desc_ddim_optical_flow_eph_1_sakuga.bin"
else:
    default_model_file = "placeholder.bin"
    

inference_steps = int(st.text_input("Enter Inference steps",value = 1))
model_file_name = (st.text_input("Enter Model file name",value=default_model_file))
cfg_scale = float(st.text_input("Enter Unconditional Guidance Scale",value = 1.0))

output_path = "/kaggle/working/Imagine-Gen-Inference-UI/outputs/IMAGINE_GEN_GENERATED_VIDEO.mp4"

if "path_array" not in st.session_state:
    st.session_state.path_array=[]

def stitch_videos():
    """
    Merges an array of video files into a single video.

    Parameters:
    video_paths (list): List of paths to the video files to merge.
    output_path (str): Path to save the merged video.
    """
    st.session_state.stitching = True
    try:
        # Load all video clips
        clips = [VideoFileClip(video) for video in st.session_state.path_array]
        
        # Concatenate the video clips
        final_clip = concatenate_videoclips(clips, method="compose")
        
        # Write the result to a file
        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
        
        print(f"Video successfully merged and saved to {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

def convert_to_base64():
    bytes_Data = uploaded_file.read()
    imgb64data = base64.b64encode(bytes_Data).decode('utf-8')
    print(imgb64data)
    return imgb64data

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def read_video(path):
    video_file = open(path, "rb")
    video_bytes = video_file.read()

    st.video(video_bytes)

progress_bar = st.progress(0)
# React to user input
if prompt := st.chat_input("Please enter a text prompt"):
    
    if selected_runner == "Pixel space(no vae)":
    	    for i in run_pixelspace_inference(prompt, "Hemabhushan/capstone_model", model_file_name, inference_steps, unconditional_guidance_scale=cfg_scale):
                if type(i) == str:
                    progress_bar.progress(1)
                    video_str = i
                    break
                # Calculate progress percentage
                progress_percentage = (i + 1) / inference_steps
                # Update the progress bar
                progress_bar.progress(progress_percentage)
    elif selected_runner == "DUMMY runner" :
        video_str = dummy_runner_inference(prompt, "Hemabhushan/capstone_model", model_file_name, inference_steps, unconditional_guidance_scale=cfg_scale)
    elif selected_runner == "Temporal Decoder" :
        video_str = run_temporal_decoder_inference(prompt, "amrithagk/capstone_model", model_file_name, inference_steps, unconditional_guidance_scale=cfg_scale)
    elif selected_runner == "Simple Description" :
        video_str = run_simple_desc_inference(prompt, "amrithagk/capstone_model", model_file_name, inference_steps, unconditional_guidance_scale=cfg_scale)
    elif selected_runner == "Latent Space Moving MNIST" :
        video_str = run_latentspace_mmnist_inference(prompt, "Hemabhushan/capstone_model", model_file_name, inference_steps, unconditional_guidance_scale=cfg_scale)
    
    st.session_state.path_array.append(video_str)

    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = f"Video generated : {video_str}"
    st.success('Completed!')
    read_video(video_str)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

if st.button('Stitch Videos', on_click=stitch_videos):
    st.success('Stitching Completed!')
    summary = f"""Video Summary :$\\newline$ """
    for i,ele in enumerate(st.session_state.messages):
        if i&1:continue
        summary+=f"Prompt {i//2 +1} : "+ele["content"]+"$\\newline$"
    read_video(output_path)
    with st.chat_message("assistant"):
        st.markdown(summary)

import os

if __name__ =="__main__":
    os.environ['HF_HUB_ENABLE_HF_TRANSFER']='1'