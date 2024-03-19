# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import json
import os
import shutil
import re
import tempfile
import subprocess
from cog import BasePredictor, Input, Path
from animatediff.utils.tagger import get_labels
import time
from google.cloud import storage

FAKE_PROMPT_TRAVEL_JSON = """
{{
  "name": "sample",
  "domain_lora_scale": 1.0,
  "adapter_lora_path": "models/motion-module/v3_sd15_adapter.ckpt",
  "dreambooth_path":   "share/Stable-diffusion/realisticVisionV60B1_v51VAE.safetensors",
  "path": "share/Stable-diffusion/realisticVisionV60B1_v51VAE.safetensors",
  "inference_config": "configs/inference/inference-v3.yaml",
  "motion_module": "models/motion-module/mm_sd_v3_fp16.safetensors",
  "compile": false,
  "seed": [
    {seed}
  ],
  "scheduler": "{scheduler}",
  "steps": {steps},
  "guidance_scale": {guidance_scale},
  "clip_skip": {clip_skip},
  "prompt_fixed_ratio": {prompt_fixed_ratio},
  "head_prompt": "{head_prompt}",
  "prompt_map": {{
    {prompt_map}
  }},
  "tail_prompt": "{tail_prompt}",
  "n_prompt": [
    "{negative_prompt}"
  ],
  "output":{{
    "format" : "{output_format}",
    "fps" : {fps},
    "encode_param":{{
      "crf": 10
    }}
  }}
}}
"""


def download_public_file(bucket_name, source_blob_name, destination_file_name):

    storage_client = storage.Client.create_anonymous_client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Downloaded public blob {} from bucket {} to {}.".format(
            source_blob_name, bucket.name, destination_file_name
        )
    )


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        pass

    def download_custom_model(self, custom_base_model_url: str):
        # Validate the custom_base_model_url to ensure it's from "civitai.com"
        if not re.match(r"^https://civitai\.com/api/download/models/\d+$", custom_base_model_url):
            raise ValueError(
                "Invalid URL. Only downloads from 'https://civitai.com/api/download/models/' are allowed."
            )

        cmd = ["wget", "-O", "data/share/Stable-diffusion/custom.safetensors", custom_base_model_url]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout_output, stderr_output = process.communicate()

        print("Output from wget command:")
        print(stdout_output)
        if stderr_output:
            print("Errors from wget command:")
            print(stderr_output)

        if process.returncode:
            raise ValueError(f"Failed to download the custom model. Wget returned code: {process.returncode}")
        return "custom"

    def transform_prompt_map(self, prompt_map_string: str):
        """
        Transform the given prompt_map string into a formatted string suitable for JSON injection.

        Parameters
        ----------
        prompt_map_string : str
            A string containing animation prompts in the format 'frame number : prompt at this frame',
            separated by '|'. Colons inside the prompt description are allowed.

        Returns
        -------
        str
            A formatted string where each prompt is represented as '"frame": "description"'.
        """

        segments = prompt_map_string.split("|")

        formatted_segments = []
        for segment in segments:
            frame, prompt = segment.split(":", 1)
            frame = frame.strip()
            prompt = prompt.strip()

            formatted_segment = f'"{frame}": "{prompt}"'
            formatted_segments.append(formatted_segment)

        return ", ".join(formatted_segments)

    def predict(
        self,
        prompt: str = Input(
            description="Primary animation prompt. If a prompt map is provided, this will be prefixed at the start of every individual prompt in the map",
            default="masterpiece, best quality, a haunting and detailed depiction of a ship at sea, battered by waves, ominous,((dark clouds:1.3)),distant lightning, rough seas, rain, silhouette of the ship against the stormy sky",
        ),
        prompt_map: str = Input(
            description="Prompt for changes in animation. Provide 'frame number : prompt at this frame', separate different prompts with '|'. Make sure the frame number does not exceed the length of video (frames)",
            default="0:",
            # default="0:ing, waves rising higher | 96: ship navigating through the storm, rain easing off",
        ),
        tail_prompt: str = Input(
            description="Additional prompt that will be appended at the end of the main prompt or individual prompts in the map",
            default="",
        ),
        negative_prompt: str = Input(
            default="",
        ),
        video_length: int = Input(
            description="Length of the video in frames (playback is at 8 fps e.g. 16 frames @ 8 fps is 2 seconds)",
            default=24,
            ge=1,
            le=1024,
        ),
        width: int = Input(
            description="Width of generated video in pixels, must be divisable by 8",
            default=512,
            ge=64,
            le=2160,
        ),
        height: int = Input(
            description="Height of generated video in pixels, must be divisable by 8",
            default=512,
            ge=64,
            le=2160,
        ),
        path: str = Input(
            description="Choose the base model for animation generation. If 'CUSTOM' is selected, provide a custom model URL in the next parameter",
            default="toonyou_beta3.safetensors",
        ),
        custom_base_model_url: str = Input(
            description="Only used when base model is set to 'CUSTOM'. URL of the custom model to download if 'CUSTOM' is selected in the base model. Only downloads from 'https://civitai.com/api/download/models/' are allowed",
            default="",
        ),
        prompt_fixed_ratio: float = Input(
            description="Defines the ratio of adherence to the fixed part of the prompt versus the dynamic part (from prompt map). Value should be between 0 (only dynamic) to 1 (only fixed).",
            default=0.5,
            ge=0,
            le=1,
        ),
        scheduler: str = Input(
            description="Diffusion scheduler",
            default="k_dpmpp_sde",
            choices=[
                "ddim",
                "pndm",
                "heun",
                "unipc",
                "euler",
                "euler_a",
                "lms",
                "k_lms",
                "dpm_2",
                "k_dpm_2",
                "dpm_2_a",
                "k_dpm_2_a",
                "dpmpp_2m",
                "k_dpmpp_2m",
                "dpmpp_sde",
                "k_dpmpp_sde",
                "dpmpp_2m_sde",
                "k_dpmpp_2m_sde",
            ],
        ),
        steps: int = Input(
            description="Number of inference steps",
            ge=1,
            le=100,
            default=25,
        ),
        guidance_scale: float = Input(
            description="Guidance Scale. How closely do we want to adhere to the prompt and its contents",
            ge=0.0,
            le=20,
            default=7.5,
        ),
        clip_skip: int = Input(
            description="Skip the last N-1 layers of the CLIP text encoder (lower values follow prompt more closely)",
            default=2,
            ge=1,
            le=6,
        ),
        context: int = Input(
            description="Number of frames to condition on (default: max of <length> or 32). max for motion module v1 is 24",
            default=16,
            ge=1,
            le=32,
        ),
        output_format: str = Input(
            description="Output format of the video. Can be 'mp4' or 'gif'",
            default="mp4",
            choices=["mp4", "gif"],
        ),
        fps: int = Input(default=8, ge=1, le=60),
        controlnetStrength: float = Input(default=0.1, ge=0.0, le=1.0),
        ipAdapterStrength: float = Input(default=0.5, ge=0.0, le=1.0),
        face: bool = Input(default=False),
        referenceImg: str = Input(default=None),
        seed: int = Input(
            description="Seed for different images and reproducibility. Use -1 to randomise seed",
            default=-1,
        ),
    ) -> Path:
        """
        Run a single prediction on the model
        NOTE: lora_map, motion_lora_map, and controlnets are NOT supported (cut scope)
        """
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if path.upper() == "CUSTOM":
            path = self.download_custom_model(custom_base_model_url)

        start_time = time.time()

        # print(f"{'-'*80}")
        # print(prompt_travel_json)
        # print(f"{'-'*80}")

        # file_path = "config/prompts/custom_prompt_travel.json"
        file_path = "input/prompt.json"

        if referenceImg is not None and referenceImg != "":
            img2video = True
            os.system("mkdir input")
            # os.system("cp brian512.png input/00000000.png") #temp
            download_public_file("bright-live-ai.appspot.com", referenceImg, "input/00000000.png")
            os.system("mkdir input/controlnet_normalbae")
            for f in range(0, video_length):
                os.system("cp input/00000000.png input/controlnet_normalbae/000000" + f"{f:02d}" + ".png")

            # Tagging the input image
            prompt_map = get_labels(
                frame_dir="input",
                interval=1,
                general_threshold=0.35,
                character_threshold=0.85,
                ignore_tokens=[],
                with_confidence=True,
                is_danbooru_format=False,
                is_cpu=False,
            )
            tags = str(prompt_map["0"])
            print("prompt_map is " + tags)

            # Parsing the input string into tuples
            parsed_data = [
                tuple(item.replace("(", "").replace(")", "").split(":")) for item in tags.split("),(")
            ]

            # Converting the value part of each tuple from string to float
            parsed_data = [(label, float(value)) for label, value in parsed_data]

            # Sorting the list by value in descending order and selecting the first four items
            # top = sorted(parsed_data, key=lambda x: x[1], reverse=True)[:4]
            # Filter out all values less than 0.6
            top = [item for item in parsed_data if item[1] >= 0.6]

            # Combining the top items back into a string
            tags_modified = ",".join([f"({label}:{value})" for label, value in top])

            print("tags modified is " + tags_modified)

            # ALWAYS SET CONTROLNET STRENGTH TO 0 FOR NOW, IP ADAPTER STRENGTH TO 0.7
            controlnetStrength = 0.0
            ipAdapterStrength = 0.7

            with open("stylize.json", "r", encoding="utf-8") as file:
                data = json.load(file)
                data["path"] = "share/Stable-diffusion/" + path
                data["tail_prompt"] = tags_modified
                data["prompt_map"]["0"] = prompt
                data["guidance_scale"] = guidance_scale
                data["seed"] = [seed]  # Need to fix this, causes error
                data["steps"] = steps
                data["ip_adapter_map"]["is_face"] = face
                if face:
                    controlnetStrength = 0.0
                    ipAdapterStrength = 0.7
                else:
                    controlnetStrength = 0.4
                    ipAdapterStrength = 0.5
                data["controlnet_map"]["controlnet_normalbae"][
                    "controlnet_conditioning_scale"
                ] = controlnetStrength
                data["ip_adapter_map"]["scale"] = ipAdapterStrength
                if ipAdapterStrength == 0.0:
                    data["ip_adapter_map"]["enable"] = False
                if controlnetStrength == 0.0:
                    data["controlnet_map"]["controlnet_normalbae"]["enable"] = False
                with open(file_path, "w", encoding="utf-8") as file:
                    json.dump(data, file, indent=4)  # indent=4 for pretty printing

        else:
            img2video = False
            print("In non-img2video and steps is " + str(steps))
            prompt_travel_json = FAKE_PROMPT_TRAVEL_JSON.format(
                dreambooth_path=f"share/Stable-diffusion/{path}",
                output_format=output_format,
                seed=seed,
                steps=steps,
                guidance_scale=guidance_scale,
                prompt_fixed_ratio=prompt_fixed_ratio,
                head_prompt=prompt,
                tail_prompt=tail_prompt,
                negative_prompt=negative_prompt,
                fps=8,  # Always generate at 8 fps, interpolate later based on fps value handed in
                prompt_map=self.transform_prompt_map(prompt_map),
                scheduler=scheduler,
                clip_skip=clip_skip,
            )
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            with open(file_path, "w") as file:
                file.write(prompt_travel_json)

        fpsMultipler = int(fps / 8)

        cmd = [
            "animatediff",
            "generate",
            "-c",
            str(file_path),
            "-W",
            str(width),
            "-H",
            str(height),
            "-L",
            str(int(video_length / fpsMultipler)),
            "-C",
            str(context),
        ]
        print(f"Running command: {' '.join(cmd)}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        (
            stdout_output,
            stderr_output,
        ) = process.communicate()

        print(stdout_output)
        if stderr_output:
            print(f"Error: {stderr_output}")

        if process.returncode:
            raise ValueError(f"Command exited with code: {process.returncode}")

        print("Identifying the GIF path from the generated outputs...")
        recent_dir = max(
            (
                os.path.join("output", d)
                for d in os.listdir("output")
                if os.path.isdir(os.path.join("output", d))
            ),
            key=os.path.getmtime,
        )

        print(f"Identified directory: {recent_dir}")

        # Get the first subdirectory of recent_dir
        directories = [
            d
            for d in os.listdir(recent_dir)
            if os.path.isdir(os.path.join(recent_dir, d)) and d.startswith("00-")
        ]

        if directories:
            source_images_path = os.path.join(recent_dir, directories[0])
            print("source_images_path is " + str(source_images_path))
        else:
            source_images_path = None  # or some other fallback in case there are no directories

        out_path = Path(tempfile.mkdtemp()) / "output.mp4"

        interpolate = fps > 8
        if interpolate:
            # rife_path = os.path.join("data", "rife")
            # rife_path = os.path.abspath(rife_path)
            # print("rife_path is " + str(rife_path))

            # os.environ["PATH"] += os.pathsep + rife_path

            rife_command = (
                "animatediff rife interpolate -M " + str(fpsMultipler) + " -c h264 " + str(source_images_path)
            )
            print("rife_command is " + str(rife_command))
            os.system(rife_command)
            media_files = [f for f in os.listdir(recent_dir) if f.endswith((".gif", ".mp4")) and "rife" in f]
        else:
            media_files = [f for f in os.listdir(recent_dir) if f.endswith((".gif", ".mp4"))]

        if not media_files:
            raise ValueError(f"No GIF or MP4 files found in directory: {recent_dir}")

        media_path = os.path.join(recent_dir, media_files[0])
        print(f"Identified Media Path: {media_path}")

        # Convert away from hev1 to more widely recognized codec
        os.system(
            "ffmpeg -i " + str(media_path) + " -movflags faststart -pix_fmt yuv420p -qp 17 " + str(out_path)
        )

        parent_dir = os.path.dirname(media_path)
        grandparent_dir = os.path.dirname(parent_dir)

        delete = False
        if delete == True:
            # Delete everything in output folder (including any prior gens that may be hanging around)
            for item in os.listdir(grandparent_dir):
                item_path = os.path.join(grandparent_dir, item)
                print(f"Deleting item at path: {item_path}")
                # Check if it's a file or directory and delete accordingly
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)

        end_time = time.time()

        # Calculate the execution time
        execution_time = end_time - start_time

        # Print the result
        if referenceImg != None:
            print("referenceImg was " + str(referenceImg))
        else:
            print("No referenceImg")
        print(
            "controlnetStrength was "
            + str(controlnetStrength)
            + " and ipAdapterStrength was "
            + str(ipAdapterStrength)
        )
        print("img2video was " + str(img2video) + " and interpolate was " + str(interpolate))
        print(f"Script execution time: {execution_time:.2f} seconds")

        return Path(out_path)
