# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import subprocess
from cog import BasePredictor, Input, Path

FAKE_PROMPT_TRAVEL_JSON = """{
  "name": "sample",
  "path": "{dreambooth_path}",
  "motion_module": "models/motion-module/mm_sd_v15_v2.ckpt",
  "compile": false,
  "seed": [
    341774366206100
  ],
  "scheduler": "k_dpmpp_sde",
  "steps": 20,
  "guidance_scale": 10,
  "clip_skip": 2,
  "prompt_fixed_ratio": 0.5,
  "head_prompt": "masterpiece, best quality, a beautiful and detailed portriat of muffet, monster girl,((purple body:1.3)),humanoid, arachnid, anthro,((fangs)),pigtails,hair bows,5 eyes,spider girl,6 arms,solo",
  "prompt_map": {
    "0": "smile standing,((spider webs:1.0))",
    "32": "(((walking))),((spider webs:1.0))",
    "64": "(((running))),((spider webs:2.0)),wide angle lens, fish eye effect",
    "96": "(((sitting))),((spider webs:1.0))"
  },
  "tail_prompt": "clothed, open mouth, awesome and detailed background, holding teapot, holding teacup, 6 hands,detailed hands,storefront that sells pastries and tea,bloomers,(red and black clothing),inside,pouring into teacup,muffetwear",
  "n_prompt": [
    "(worst quality, low quality:1.4),nudity,simple background,border,mouth closed,text, patreon,bed,bedroom,white background,((monochrome)),sketch,(pink body:1.4),7 arms,8 arms,4 arms"
  ],
  "lora_map": {
    "share/Lora/muffet_v2.safetensors": 1.0,
    "share/Lora/add_detail.safetensors": 1.0
  },
  "motion_lora_map": {
    "models/motion_lora/v2_lora_PanLeft.ckpt": 1.0
  },
  "ip_adapter_map": {
    "enable": true,
    "input_image_dir": "ip_adapter_image/test",
    "save_input_image": true,
    "resized_to_square": false,
    "scale": 0.5,
    "is_plus_face": true,
    "is_plus": true
  },
  "controlnet_map": {
    "input_image_dir": "controlnet_image/test",
    "max_samples_on_vram": 200,
    "max_models_on_vram": 3,
    "save_detectmap": true,
    "preprocess_on_gpu": true,
    "is_loop": true,
    "controlnet_tile": {
      "enable": true,
      "use_preprocessor": true,
      "preprocessor": {
        "type": "none",
        "param": {}
      },
      "guess_mode": false,
      "controlnet_conditioning_scale": 1.0,
      "control_guidance_start": 0.0,
      "control_guidance_end": 1.0,
      "control_scale_list": [
        0.5,
        0.4,
        0.3,
        0.2,
        0.1
      ]
    },
    "controlnet_ip2p": {
      "enable": true,
      "use_preprocessor": true,
      "guess_mode": false,
      "controlnet_conditioning_scale": 1.0,
      "control_guidance_start": 0.0,
      "control_guidance_end": 1.0,
      "control_scale_list": [
        0.5,
        0.4,
        0.3,
        0.2,
        0.1
      ]
    },
    "controlnet_lineart_anime": {
      "enable": true,
      "use_preprocessor": true,
      "guess_mode": false,
      "controlnet_conditioning_scale": 1.0,
      "control_guidance_start": 0.0,
      "control_guidance_end": 1.0,
      "control_scale_list": [
        0.5,
        0.4,
        0.3,
        0.2,
        0.1
      ]
    },
    "controlnet_openpose": {
      "enable": true,
      "use_preprocessor": true,
      "guess_mode": false,
      "controlnet_conditioning_scale": 1.0,
      "control_guidance_start": 0.0,
      "control_guidance_end": 1.0,
      "control_scale_list": [
        0.5,
        0.4,
        0.3,
        0.2,
        0.1
      ]
    },
    "controlnet_softedge": {
      "enable": true,
      "use_preprocessor": true,
      "preprocessor": {
        "type": "softedge_pidsafe",
        "param": {}
      },
      "guess_mode": false,
      "controlnet_conditioning_scale": 1.0,
      "control_guidance_start": 0.0,
      "control_guidance_end": 1.0,
      "control_scale_list": [
        0.5,
        0.4,
        0.3,
        0.2,
        0.1
      ]
    },
    "controlnet_shuffle": {
      "enable": true,
      "use_preprocessor": true,
      "guess_mode": false,
      "controlnet_conditioning_scale": 1.0,
      "control_guidance_start": 0.0,
      "control_guidance_end": 1.0,
      "control_scale_list": [
        0.5,
        0.4,
        0.3,
        0.2,
        0.1
      ]
    },
    "controlnet_depth": {
      "enable": true,
      "use_preprocessor": true,
      "guess_mode": false,
      "controlnet_conditioning_scale": 1.0,
      "control_guidance_start": 0.0,
      "control_guidance_end": 1.0,
      "control_scale_list": [
        0.5,
        0.4,
        0.3,
        0.2,
        0.1
      ]
    },
    "controlnet_canny": {
      "enable": true,
      "use_preprocessor": true,
      "guess_mode": false,
      "controlnet_conditioning_scale": 1.0,
      "control_guidance_start": 0.0,
      "control_guidance_end": 1.0,
      "control_scale_list": [
        0.5,
        0.4,
        0.3,
        0.2,
        0.1
      ]
    },
    "controlnet_inpaint": {
      "enable": true,
      "use_preprocessor": true,
      "guess_mode": false,
      "controlnet_conditioning_scale": 1.0,
      "control_guidance_start": 0.0,
      "control_guidance_end": 1.0,
      "control_scale_list": [
        0.5,
        0.4,
        0.3,
        0.2,
        0.1
      ]
    },
    "controlnet_lineart": {
      "enable": true,
      "use_preprocessor": true,
      "guess_mode": false,
      "controlnet_conditioning_scale": 1.0,
      "control_guidance_start": 0.0,
      "control_guidance_end": 1.0,
      "control_scale_list": [
        0.5,
        0.4,
        0.3,
        0.2,
        0.1
      ]
    },
    "controlnet_mlsd": {
      "enable": true,
      "use_preprocessor": true,
      "guess_mode": false,
      "controlnet_conditioning_scale": 1.0,
      "control_guidance_start": 0.0,
      "control_guidance_end": 1.0,
      "control_scale_list": [
        0.5,
        0.4,
        0.3,
        0.2,
        0.1
      ]
    },
    "controlnet_normalbae": {
      "enable": true,
      "use_preprocessor": true,
      "guess_mode": false,
      "controlnet_conditioning_scale": 1.0,
      "control_guidance_start": 0.0,
      "control_guidance_end": 1.0,
      "control_scale_list": [
        0.5,
        0.4,
        0.3,
        0.2,
        0.1
      ]
    },
    "controlnet_scribble": {
      "enable": true,
      "use_preprocessor": true,
      "guess_mode": false,
      "controlnet_conditioning_scale": 1.0,
      "control_guidance_start": 0.0,
      "control_guidance_end": 1.0,
      "control_scale_list": [
        0.5,
        0.4,
        0.3,
        0.2,
        0.1
      ]
    },
    "controlnet_seg": {
      "enable": true,
      "use_preprocessor": true,
      "guess_mode": false,
      "controlnet_conditioning_scale": 1.0,
      "control_guidance_start": 0.0,
      "control_guidance_end": 1.0,
      "control_scale_list": [
        0.5,
        0.4,
        0.3,
        0.2,
        0.1
      ]
    },
    "controlnet_ref": {
      "enable": false,
      "ref_image": "ref_image/ref_sample.png",
      "attention_auto_machine_weight": 0.3,
      "gn_auto_machine_weight": 0.3,
      "style_fidelity": 0.5,
      "reference_attn": true,
      "reference_adain": false,
      "scale_pattern": [
        1.0
      ]
    }
  },
  "upscale_config": {
    "scheduler": "k_dpmpp_sde",
    "steps": 20,
    "strength": 0.5,
    "guidance_scale": 10,
    "controlnet_tile": {
      "enable": true,
      "controlnet_conditioning_scale": 1.0,
      "guess_mode": false,
      "control_guidance_start": 0.0,
      "control_guidance_end": 1.0
    },
    "controlnet_line_anime": {
      "enable": false,
      "controlnet_conditioning_scale": 1.0,
      "guess_mode": false,
      "control_guidance_start": 0.0,
      "control_guidance_end": 1.0
    },
    "controlnet_ip2p": {
      "enable": false,
      "controlnet_conditioning_scale": 0.5,
      "guess_mode": false,
      "control_guidance_start": 0.0,
      "control_guidance_end": 1.0
    },
    "controlnet_ref": {
      "enable": false,
      "use_frame_as_ref_image": false,
      "use_1st_frame_as_ref_image": false,
      "ref_image": "ref_image/path_to_your_ref_img.jpg",
      "attention_auto_machine_weight": 1.0,
      "gn_auto_machine_weight": 1.0,
      "style_fidelity": 0.25,
      "reference_attn": true,
      "reference_adain": false
    }
  },
  "output": {
    "format": "gif",
    "fps": 8,
    "encode_param": {
      "crf": 10
    }
  }
}"""


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        pass

    # def set_defaults(self):
    # name = sample
    # path = share/Stable-diffusion/toonyou_beta3.safetensors
    # motion_module = models/motion-module/mm_sd_v15_v2.ckpt
    # compile = False
    # scheduler = k_dpmpp_sde
    # steps = 20
    # guidance_scale = 10
    # clip_skip = 2
    # prompt_fixed_ratio = 0.5
    # head_prompt = masterpiece, best quality, a beautiful and detailed portriat of muffet, monster girl,((purple body:1.3)),humanoid, arachnid, anthro,((fangs)),pigtails,hair bows,5 eyes,spider girl,6 arms,solo
    # prompt_map.0 = smile standing,((spider webs:1.0))
    # prompt_map.32 = (((walking))),((spider webs:1.0))
    # prompt_map.64 = (((running))),((spider webs:2.0)),wide angle lens, fish eye effect
    # prompt_map.96 = (((sitting))),((spider webs:1.0))
    # tail_prompt = clothed, open mouth, awesome and detailed background, holding teapot, holding teacup, 6 hands,detailed hands,storefront that sells pastries and tea,bloomers,(red and black clothing),inside,pouring into teacup,muffetwear
    # lora_map.share_Lora_muffet_v2.safetensors = 1.0
    # lora_map.share_Lora_add_detail.safetensors = 1.0
    # motion_lora_map.models_motion_lora_v2_lora_PanLeft.ckpt = 1.0
    # ip_adapter_map.enable = True
    # ip_adapter_map.input_image_dir = ip_adapter_image/test
    # ip_adapter_map.save_input_image = True
    # ip_adapter_map.resized_to_square = False
    # ip_adapter_map.scale = 0.5
    # ip_adapter_map.is_plus_face = True
    # ip_adapter_map.is_plus = True
    # controlnet_map.input_image_dir = controlnet_image/test
    # controlnet_map.max_samples_on_vram = 200
    # controlnet_map.max_models_on_vram = 3
    # controlnet_map.save_detectmap = True
    # controlnet_map.preprocess_on_gpu = True
    # controlnet_map.is_loop = True
    # controlnet_map.controlnet_tile.enable = True
    # controlnet_map.controlnet_tile.use_preprocessor = True
    # controlnet_map.controlnet_tile.preprocessor.type = none
    # controlnet_map.controlnet_tile.guess_mode = False
    # controlnet_map.controlnet_tile.controlnet_conditioning_scale = 1.0
    # controlnet_map.controlnet_tile.control_guidance_start = 0.0
    # controlnet_map.controlnet_tile.control_guidance_end = 1.0
    # controlnet_map.controlnet_ip2p.enable = True
    # controlnet_map.controlnet_ip2p.use_preprocessor = True
    # controlnet_map.controlnet_ip2p.guess_mode = False
    # controlnet_map.controlnet_ip2p.controlnet_conditioning_scale = 1.0
    # controlnet_map.controlnet_ip2p.control_guidance_start = 0.0
    # controlnet_map.controlnet_ip2p.control_guidance_end = 1.0
    # controlnet_map.controlnet_lineart_anime.enable = True
    # controlnet_map.controlnet_lineart_anime.use_preprocessor = True
    # controlnet_map.controlnet_lineart_anime.guess_mode = False
    # controlnet_map.controlnet_lineart_anime.controlnet_conditioning_scale = 1.0
    # controlnet_map.controlnet_lineart_anime.control_guidance_start = 0.0
    # controlnet_map.controlnet_lineart_anime.control_guidance_end = 1.0
    # controlnet_map.controlnet_openpose.enable = True
    # controlnet_map.controlnet_openpose.use_preprocessor = True
    # controlnet_map.controlnet_openpose.guess_mode = False
    # controlnet_map.controlnet_openpose.controlnet_conditioning_scale = 1.0
    # controlnet_map.controlnet_openpose.control_guidance_start = 0.0
    # controlnet_map.controlnet_openpose.control_guidance_end = 1.0
    # controlnet_map.controlnet_softedge.enable = True
    # controlnet_map.controlnet_softedge.use_preprocessor = True
    # controlnet_map.controlnet_softedge.preprocessor.type = softedge_pidsafe
    # controlnet_map.controlnet_softedge.guess_mode = False
    # controlnet_map.controlnet_softedge.controlnet_conditioning_scale = 1.0
    # controlnet_map.controlnet_softedge.control_guidance_start = 0.0
    # controlnet_map.controlnet_softedge.control_guidance_end = 1.0
    # controlnet_map.controlnet_shuffle.enable = True
    # controlnet_map.controlnet_shuffle.use_preprocessor = True
    # controlnet_map.controlnet_shuffle.guess_mode = False
    # controlnet_map.controlnet_shuffle.controlnet_conditioning_scale = 1.0
    # controlnet_map.controlnet_shuffle.control_guidance_start = 0.0
    # controlnet_map.controlnet_shuffle.control_guidance_end = 1.0
    # controlnet_map.controlnet_depth.enable = True
    # controlnet_map.controlnet_depth.use_preprocessor = True
    # controlnet_map.controlnet_depth.guess_mode = False
    # controlnet_map.controlnet_depth.controlnet_conditioning_scale = 1.0
    # controlnet_map.controlnet_depth.control_guidance_start = 0.0
    # controlnet_map.controlnet_depth.control_guidance_end = 1.0
    # controlnet_map.controlnet_canny.enable = True
    # controlnet_map.controlnet_canny.use_preprocessor = True
    # controlnet_map.controlnet_canny.guess_mode = False
    # controlnet_map.controlnet_canny.controlnet_conditioning_scale = 1.0
    # controlnet_map.controlnet_canny.control_guidance_start = 0.0
    # controlnet_map.controlnet_canny.control_guidance_end = 1.0
    # controlnet_map.controlnet_inpaint.enable = True
    # controlnet_map.controlnet_inpaint.use_preprocessor = True
    # controlnet_map.controlnet_inpaint.guess_mode = False
    # controlnet_map.controlnet_inpaint.controlnet_conditioning_scale = 1.0
    # controlnet_map.controlnet_inpaint.control_guidance_start = 0.0
    # controlnet_map.controlnet_inpaint.control_guidance_end = 1.0
    # controlnet_map.controlnet_lineart.enable = True
    # controlnet_map.controlnet_lineart.use_preprocessor = True
    # controlnet_map.controlnet_lineart.guess_mode = False
    # controlnet_map.controlnet_lineart.controlnet_conditioning_scale = 1.0
    # controlnet_map.controlnet_lineart.control_guidance_start = 0.0
    # controlnet_map.controlnet_lineart.control_guidance_end = 1.0
    # controlnet_map.controlnet_mlsd.enable = True
    # controlnet_map.controlnet_mlsd.use_preprocessor = True
    # controlnet_map.controlnet_mlsd.guess_mode = False
    # controlnet_map.controlnet_mlsd.controlnet_conditioning_scale = 1.0
    # controlnet_map.controlnet_mlsd.control_guidance_start = 0.0
    # controlnet_map.controlnet_mlsd.control_guidance_end = 1.0
    # controlnet_map.controlnet_normalbae.enable = True
    # controlnet_map.controlnet_normalbae.use_preprocessor = True
    # controlnet_map.controlnet_normalbae.guess_mode = False
    # controlnet_map.controlnet_normalbae.controlnet_conditioning_scale = 1.0
    # controlnet_map.controlnet_normalbae.control_guidance_start = 0.0
    # controlnet_map.controlnet_normalbae.control_guidance_end = 1.0
    # controlnet_map.controlnet_scribble.enable = True
    # controlnet_map.controlnet_scribble.use_preprocessor = True
    # controlnet_map.controlnet_scribble.guess_mode = False
    # controlnet_map.controlnet_scribble.controlnet_conditioning_scale = 1.0
    # controlnet_map.controlnet_scribble.control_guidance_start = 0.0
    # controlnet_map.controlnet_scribble.control_guidance_end = 1.0
    # controlnet_map.controlnet_seg.enable = True
    # controlnet_map.controlnet_seg.use_preprocessor = True
    # controlnet_map.controlnet_seg.guess_mode = False
    # controlnet_map.controlnet_seg.controlnet_conditioning_scale = 1.0
    # controlnet_map.controlnet_seg.control_guidance_start = 0.0
    # controlnet_map.controlnet_seg.control_guidance_end = 1.0
    # controlnet_map.controlnet_ref.enable = False
    # controlnet_map.controlnet_ref.ref_image = ref_image/ref_sample.png
    # controlnet_map.controlnet_ref.attention_auto_machine_weight = 0.3
    # controlnet_map.controlnet_ref.gn_auto_machine_weight = 0.3
    # controlnet_map.controlnet_ref.style_fidelity = 0.5
    # controlnet_map.controlnet_ref.reference_attn = True
    # controlnet_map.controlnet_ref.reference_adain = False
    # upscale_config.scheduler = k_dpmpp_sde
    # upscale_config.steps = 20
    # upscale_config.strength = 0.5
    # upscale_config.guidance_scale = 10
    # upscale_config.controlnet_tile.enable = True
    # upscale_config.controlnet_tile.controlnet_conditioning_scale = 1.0
    # upscale_config.controlnet_tile.guess_mode = False
    # upscale_config.controlnet_tile.control_guidance_start = 0.0
    # upscale_config.controlnet_tile.control_guidance_end = 1.0
    # upscale_config.controlnet_line_anime.enable = False
    # upscale_config.controlnet_line_anime.controlnet_conditioning_scale = 1.0
    # upscale_config.controlnet_line_anime.guess_mode = False
    # upscale_config.controlnet_line_anime.control_guidance_start = 0.0
    # upscale_config.controlnet_line_anime.control_guidance_end = 1.0
    # upscale_config.controlnet_ip2p.enable = False
    # upscale_config.controlnet_ip2p.controlnet_conditioning_scale = 0.5
    # upscale_config.controlnet_ip2p.guess_mode = False
    # upscale_config.controlnet_ip2p.control_guidance_start = 0.0
    # upscale_config.controlnet_ip2p.control_guidance_end = 1.0
    # upscale_config.controlnet_ref.enable = False
    # upscale_config.controlnet_ref.use_frame_as_ref_image = False
    # upscale_config.controlnet_ref.use_1st_frame_as_ref_image = False
    # upscale_config.controlnet_ref.ref_image = ref_image/path_to_your_ref_img.jpg
    # upscale_config.controlnet_ref.attention_auto_machine_weight = 1.0
    # upscale_config.controlnet_ref.gn_auto_machine_weight = 1.0
    # upscale_config.controlnet_ref.style_fidelity = 0.25
    # upscale_config.controlnet_ref.reference_attn = True
    # upscale_config.controlnet_ref.reference_adain = False
    # output.format = gif
    # output.fps = 8
    # output.encode_param.crf = 10

    def predict(
        self,
        prompt_travel_json: str = Input(
            description="Hacky way to funnel info into `config/prompts/prompt_travel.json`",
            default="",
        ),
        # base_model: str = Input(
        #     description="Select a base model (DreamBooth checkpoint)",
        #     default="realisticVisionV20_v20",
        #     choices=[
        #         "realisticVisionV20_v20",
        #         "lyriel_v16",
        #         "majicmixRealistic_v5Preview",
        #         "rcnzCartoon3d_v10",
        #         "toonyou_beta3",
        #     ],
        # ),
        frames: int = Input(
            description="Length of the video in frames (playback is at 8 fps e.g. 16 frames @ 8 fps is 2 seconds)",
            default=128,
            ge=1,
            le=1024,
        ),
        width: int = Input(
            description="Width of generated video in pixels",
            default=256,
            ge=64,
            le=2160,
        ),
        height: int = Input(
            description="Height of generated video in pixels",
            default=384,
            ge=64,
            le=2160,
        ),
        context: int = Input(
            description="Number of frames to condition on (default: max of <length> or 32). max for motion module v1 is 24",
            default=16,
            ge=1,
            le=32,
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        # dreambooth_path = f"share/Stable-diffusion/{base_model}.safetensors"
        # config = FAKE_PROMPT_TRAVEL_JSON.format(dreambooth_path=dreambooth_path)

        # Read the content from the file
        file_path = "config/prompts/prompt_travel.json"
        with open(file_path, "r") as file:
            config = file.read()

        # Debugging: Print the command that's about to run
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
            str(frames),
            "-C",
            str(context),
        ]
        print(f"Running command: {' '.join(cmd)}")

        # Execute the command and wait for it to finish
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        (
            stdout_output,
            stderr_output,
        ) = process.communicate()  # NOTE: Done to purposely block, only continuing when cmmand completes

        # Debugging: Print the output and any error messages
        print(stdout_output)
        if stderr_output:
            print(f"Error: {stderr_output}")

        # Check if the process completed successfully
        if process.returncode:
            raise ValueError(f"Command exited with code: {process.returncode}")

        # Debugging: Determine the output path
        print("Identifying the GIF path from the generated outputs...")

        # Identify the most recently created directory in the 'output/' folder
        recent_dir = max(
            (
                os.path.join("output", d)
                for d in os.listdir("output")
                if os.path.isdir(os.path.join("output", d))
            ),
            key=os.path.getmtime,
        )
        print(f"Identified directory: {recent_dir}")

        # Search for a `.gif` or `.mp4` file inside the identified directory
        media_files = [f for f in os.listdir(recent_dir) if f.endswith((".gif", ".mp4"))]

        if not media_files:
            raise ValueError(f"No GIF or MP4 files found in directory: {recent_dir}")

        # Get the full path of the media file
        media_path = os.path.join(recent_dir, media_files[0])
        print(f"Identified Media Path: {media_path}")

        # Return the path to the found media file
        return Path(media_path)
