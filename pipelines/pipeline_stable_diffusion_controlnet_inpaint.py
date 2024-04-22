import inspect
from builtins import tuple
from typing import Union, List, Optional, Callable, Dict, Any, Tuple, Iterable

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from torch.utils import checkpoint
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, \
    CLIPTextModelWithProjection, CLIPVisionConfig

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor
from attenprocessor import CrossAttnProcessor, AttentionStore, GaussianSmoothing
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import is_compiled_module, is_torch_version, randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

from utils.utils import split_multi_net

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> # !pip install opencv-python transformers accelerate
        >>> from diffusers import ControlNetModel, UniPCMultistepScheduler
        >>> from diffusers.utils import load_image
        >>> import numpy as np
        >>> import torch

        >>> import cv2
        >>> from PIL import Image

        >>> # download an image
        >>> image = load_image(
        ...     "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
        ... )
        >>> image = np.array(image)

        >>> # get canny image
        >>> image = cv2.Canny(image, 100, 200)
        >>> image = image[:, :, None]
        >>> image = np.concatenate([image, image, image], axis=2)
        >>> canny_image = Image.fromarray(image)

        >>> # load control net and stable diffusion v1-5
        >>> controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
        >>> pipe = StableDiffusionControlNetImg2ImgInpaintPipeline.from_pretrained(
        ...     "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
        ... )

        >>> # speed up diffusion process with faster scheduler and memory optimization
        >>> pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        >>> # remove following line if xformers is not installed
        >>> pipe.enable_xformers_memory_efficient_attention()

        >>> pipe.enable_model_cpu_offload()

        >>> # generate image
        >>> generator = torch.manual_seed(0)
        >>> image = pipe(
        ...     "futuristic-looking woman", num_inference_steps=20, generator=generator, image=canny_image
        ... ).images[0]
        ```
"""

# (*) The focus attention map required to mask the controlnet is collected from the cross attention modules of the unet:
#     the parts of the latent image which receive attention with respect a `focus prompt`. Ideally, it's true that we
#     could accumulate the attention only in the down block of the unet then mask the controlnet output but this would
#     move the logic inside the unet which is not a light-hearted action. Therefore, we collect the attention in a step
#     and we use it in the subsequent one. About the first step, where `attn_mask` is None we avoid to mess up the unet
#     with unmasked controlnet signals, by passing None in place of controlnet mid and down blocks.

class StableDiffusionControlNetImg2ImgInpaintPipeline(
    DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin, FromSingleFileMixin
):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion with MultiControlNet guidance modified with
    custom controlnet controls.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] for loading IP Adapters

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        controlnet ([`ControlNetModel`] or `List[ControlNetModel]`):
            Provides additional conditioning to the `unet` during the denoising process. If you set multiple
            ControlNets as a list, the outputs from each ControlNet are added together to create one combined
            additional conditioning.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        controlnet_text_encoder: (`CLIPTextModelWithProjection`, *optional*):
            A CLIPText encoder used by the ControlNet(s). This is add-on to standard pipeline since we require to
            encode controlnet prompts either they are text or image during training.
        controlnet_image_encoder: (`CLIPVisionModelWithProjection`, *optional*):
            A CLIPVision encoder used by the ControlNet(s). This is add-on to standard pipeline since we require to
            encode controlnet prompts either they are text or image during training.           
        controlnet_prompt_seq_projection: (`bool`):
            Apply the projection layer to the whole sequence of the output of the CLIP encoder (the last hidden layer).
            The projection layer is usually applied during CLIP training to the last token of the sequence to match
            text and video and train the network. We employ such projection layer to match the hidden size of text
            and image encodings to pass to the unet cross attention blocks.  
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """

    model_cpu_offload_seq = "text_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    _exclude_from_cpu_offload = ["safety_checker"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel],
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        controlnet_text_encoder: CLIPTextModelWithProjection = None,
        controlnet_image_encoder: CLIPVisionModelWithProjection = None,
        controlnet_prompt_seq_projection: bool = False,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        if isinstance(controlnet, (list, tuple)):
            controlnet = MultiControlNetModel(controlnet)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            controlnet_text_encoder=controlnet_text_encoder,
            controlnet_image_encoder=controlnet_image_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        self.mask_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
        self.conditioning_image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False)
        self.register_to_config(requires_safety_checker=requires_safety_checker)
        self.controlnet_prompt_seq_projection = controlnet_prompt_seq_projection
        self.attention_store = None
        self.attn_mask = None

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def _get_image_embeddings(self, image_path, encoder, device=None, dtype=None):
        image = [PIL.Image.open(p).convert("RGB") for p in image_path]
        image = self.image_processor.preprocess(image, 224, 224).to(device=device).to(dtype=dtype)
        image_embeds = encoder(pixel_values=image).last_hidden_state

        return image_embeds.to(device=device, dtype=dtype)

    def _get_text_embeddings(self, prompt, encoder, max_length, check_truncation=True, clip_skip=None, device=None, dtype=None):
        text_inputs = self.tokenizer(prompt, padding="max_length", max_length=max_length, truncation=True, return_length=True, return_tensors="pt")
        text_tokenized_length = text_inputs.length - 2  # <startoftext< and <endoftext>
        text_input_ids = text_inputs.input_ids.to(device=device)

        if check_truncation:
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1: -1])
                logger.warning(f"The following part of your input was truncated because CLIP can only handle sequences up to {self.tokenizer.model_max_length} tokens: {removed_text}")

        if hasattr(encoder.config, "use_attention_mask") and encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        if clip_skip is None:
            prompt_embeds = encoder(text_input_ids.to(device), attention_mask=attention_mask).last_hidden_state
        else:
            prompt_embeds = encoder(text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True)
            # Access the `hidden_states` first, that contains a tuple of all the hidden states from the encoder layers.
            # Then index into the tuple to access the hidden states from the desired layer.
            prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
            # We also need to apply the final LayerNorm here to not mess with the representations.
            # The `last_hidden_states` that we typically use for obtaining the final prompt representations passes through the LayerNorm layer.
            prompt_embeds = encoder.text_model.final_layer_norm(prompt_embeds)

        return prompt_embeds.to(device=device, dtype=dtype), text_input_ids, text_tokenized_length

    def encode_prompt(
        self,
        prompt,
        device,
        do_classifier_free_guidance,
        encoder,
        num_images_per_prompt=1,
        negative_prompt=None,
        return_tokenizer_output=False,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
        return_tuple: Optional[bool] = True,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:[
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            encoder (`CLIPTextModel`,`CLIPVisionModel`,`CLIPVisionModelWithProjection`,`CLIPVisionModelWithProjection`):
                encoder instance to use for the input whether text or image
            num_images_per_prompt (`int`, *optional*):
                number of images that should be generated per prompt
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            return_tokenizer_output (`bool`, *optional*):
                return tokenized prompt ids and lengths
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            return_tuple (`bool`, *optional*):
                whether return tuple or a concatenated output of negative and positive embeddings
        """
        dtype = encoder.dtype if encoder is not None else self.unet.dtype if self.unet is not None else prompt_embeds.dtype
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(encoder, lora_scale)
            else:
                scale_lora_layers(encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # textual inversion: process multi-vector tokens if necessary
        if isinstance(self, TextualInversionLoaderMixin) and prompt is not None:
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

        if encoder.config_class == CLIPVisionConfig:
            tokenized_prompt_ids = None
            tokenized_prompt_length = None
            if prompt_embeds is None:
                prompt_embeds = self._get_image_embeddings(prompt, encoder, device, dtype)
        else:
            if prompt_embeds is None:
                prompt_embeds, tokenized_prompt_ids, tokenized_prompt_length = self._get_text_embeddings(prompt, encoder, self.tokenizer.model_max_length, device=device, dtype=dtype)
            else:
                tokenized_prompt = self.tokenizer(prompt if prompt is not None else "", padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_length=True, return_tensors="pt")
                tokenized_prompt_ids = tokenized_prompt.input_ids.to(device=device)
                tokenized_prompt_length = tokenized_prompt.length - 2

            tokenized_prompt_ids = tokenized_prompt_ids.repeat_interleave(num_images_per_prompt, 0)
            tokenized_prompt_length = tokenized_prompt_length.repeat_interleave(num_images_per_prompt, 0)

        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt,0)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            if negative_prompt_embeds is None:
                uncond_tokens: List[str]
                if negative_prompt is None:
                    uncond_tokens = [""] * batch_size
                elif isinstance(negative_prompt, str):
                    uncond_tokens = [negative_prompt]
                elif batch_size != len(negative_prompt):
                    raise ValueError(
                        f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                        f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                        " the batch size of `prompt`."
                    )
                else: uncond_tokens = negative_prompt
    
                # textual inversion: process multi-vector tokens if necessary
                if isinstance(self, TextualInversionLoaderMixin):
                    uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)
    
                negative_prompt_embeds, _, _ = self._get_text_embeddings(uncond_tokens, encoder, max_length=prompt_embeds.shape[1], check_truncation=False, device=device, dtype=dtype)

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_images_per_prompt, 0)

        if isinstance(self, LoraLoaderMixin) and USE_PEFT_BACKEND:
            # Retrieve the original scale by scaling back the LoRA layers
            unscale_lora_layers(encoder, lora_scale)

        if not return_tuple and do_classifier_free_guidance:
            output = torch.cat([negative_prompt_embeds, prompt_embeds])
        elif not return_tuple and not do_classifier_free_guidance:
            output = prompt_embeds
        else:
            output = (prompt_embeds, negative_prompt_embeds)
        if return_tokenizer_output:
            output = output, tokenized_prompt_ids, tokenized_prompt_length

        return output

    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept

    def _decode_vae_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        # Prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def get_timesteps(self, num_inference_steps, strength):
        # Get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start

    def check_inputs(
        self, prompt, controlnet_prompt, focus_prompt, image, height, width, mask=None, negative_prompt=None, prompt_embeds=None, negative_prompt_embeds=None, focus_prompt_embeds=None, controlnet_conditioning_scale=1.0, control_guidance_start=0.0, control_guidance_end=1.0, callback_on_step_end_tensor_inputs=None, generator=None
    ):
        if height is not None and height % 8 != 0 or width is not None and width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
                k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs},"
                f" but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        # Check `prompt`
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        # Check `controlnet_prompt`
        batched_input = not isinstance(prompt, str)
        # logger.info("INFO: The input provided will be considered as batched.")
        input_batch_size = (len(prompt) if isinstance(prompt, list) else 1) if prompt is not None else len(prompt_embeds)
        if self.controlnet is not None and controlnet_prompt is None:
            raise ValueError("The field `controlnet` is not None but no `controlnet_prompt` has been specified.")
        if batched_input:  # batched input
            if not isinstance(controlnet_prompt, list) or (len(controlnet_prompt) != input_batch_size):
                raise ValueError(f"The batched controlnet prompt do not match with the batch size.")
            elif (isinstance(self.controlnet, MultiControlNetModel) and any([len(controlnet_prompt_) != len(self.controlnet.nets) for controlnet_prompt_ in controlnet_prompt])) \
                    or (isinstance(self.controlnet, ControlNetModel) and any([not isinstance(controlnet_prompt_, str) for controlnet_prompt_ in controlnet_prompt])):
                raise ValueError(f"The number of `controlnet_prompt` passed do not match the number of ControlNet(s) available.")
        else:
            if isinstance(self.controlnet, MultiControlNetModel):
                if not isinstance(controlnet_prompt, list):
                    raise TypeError("For multiple ControlNet(s), `controlnet_prompt` must be type `list`.")
                elif len(controlnet_prompt) != len(self.controlnet.nets):
                    raise ValueError(f"You have {len(self.controlnet.nets)} ControlNets but you have passed "
                                     f"{len(controlnet_prompt)} `controlnet_prompt`, which is an invalid configuration.")
            elif isinstance(self.controlnet, ControlNetModel) and not isinstance(controlnet_prompt, str):
                raise ValueError(f"You have a single ControlNet but {len(controlnet_prompt)} `controlnet_prompt` are passed.")

        # Check `focus_prompt`
        if focus_prompt is not None and focus_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `focus_prompt`: {focus_prompt} and `focus_prompt_embeds`: {focus_prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        if focus_prompt is not None:
            if input_batch_size>1:  # batched input
                if any([not isinstance(focus_prompt_, str) for focus_prompt_ in focus_prompt]) and not all([isinstance(focus_prompt_, list) for focus_prompt_ in focus_prompt]):
                    raise ValueError("Cannot forward mixed types of `focus_prompt`. Only strings or lists of strings are accepted.")
                if isinstance(focus_prompt[0], list) and any([not all([isinstance(focus_prompt__, str) for focus_prompt__ in focus_prompt_]) for focus_prompt_ in focus_prompt]):
                    raise ValueError("Only strings or lists of strings are accepted as `focus_prompt`.")
            elif not isinstance(focus_prompt, str) and not isinstance(focus_prompt, list):
                    raise ValueError("Only strings or lists of strings are accepted as `focus_prompt`.")

        # Check `image`
        is_compiled = hasattr(F, "scaled_dot_product_attention") and isinstance(self.controlnet, torch._dynamo.eval_frame.OptimizedModule)
        if (isinstance(self.controlnet, ControlNetModel) or is_compiled and isinstance(self.controlnet._orig_mod, ControlNetModel)):
            self.check_image(image, prompt, prompt_embeds)
        elif (isinstance(self.controlnet, MultiControlNetModel) or is_compiled and isinstance(self.controlnet._orig_mod, MultiControlNetModel)):
            if batched_input:  # batched input
                if not isinstance(image, Iterable) or len(image) != input_batch_size:
                    raise ValueError(f"The batched controlnet conditioning do not match with the batch size.")
                elif not all([isinstance(image_, Iterable) for image_ in image]):
                    raise TypeError("For multiple controlnets: `conditioning_image` must be type `list`.")
                elif any([len(image_) != len(self.controlnet.nets) for image_ in image]):
                    raise ValueError(f"You have {len(self.controlnet.nets)} ControlNets but you have passed "
                                     f"a sample with invalid number of `conditioning_image`.")
            else:
                if not isinstance(image, Iterable):
                    raise TypeError("For multiple controlnets: `conditioning_image` must be type `list`.")
                elif len(image) != len(self.controlnet.nets):
                    raise ValueError(f"You have {len(self.controlnet.nets)} ControlNets but you have passed "
                                     f"{len(image)} `conditioning_image`, which is an invalid configuration.")
                for image_ in image:
                    self.check_image(image_, prompt, prompt_embeds)

        if mask is not None:
            if not isinstance(mask, PIL.Image.Image) and (not isinstance(mask, Iterable) or (isinstance(mask, Iterable) and not all([isinstance(mask_, PIL.Image.Image) or isinstance(mask_, Iterable) for mask_ in mask]))):
                raise TypeError("`mask` can only be one of PIL.Image.Image, torch.Tensor or numpy.array.")
            if (isinstance(mask, PIL.Image.Image) and input_batch_size>1) or (isinstance(mask, Iterable) and len(mask) != input_batch_size):
                if not input_batch_size % (1 if isinstance(mask, PIL.Image.Image) else len(mask)) == 0:
                    raise ValueError(
                        "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                        f" a total batch size of {input_batch_size}. Make sure the number of masks that you pass"
                        " is divisible by the total requested batch size."
                    )
                logger.warning("WARN: The passed mask and the required batch size don't match, but it will be replicated across the batches.")

        # Check `controlnet_conditioning_scale`
        if (isinstance(self.controlnet, ControlNetModel) or is_compiled and isinstance(self.controlnet._orig_mod, ControlNetModel)):
            if not isinstance(controlnet_conditioning_scale, float):
                raise TypeError("For single controlnet: `controlnet_conditioning_scale` must be type `float`.")
        elif (isinstance(self.controlnet, MultiControlNetModel) or is_compiled and isinstance(self.controlnet._orig_mod, MultiControlNetModel)):
            if isinstance(controlnet_conditioning_scale, Iterable):
                if any(isinstance(i, list) for i in controlnet_conditioning_scale):
                    raise ValueError("A single batch of multiple conditionings are supported at the moment.")
            elif isinstance(controlnet_conditioning_scale, Iterable) and len(controlnet_conditioning_scale) != len(self.controlnet.nets):
                raise ValueError(
                    "For multiple controlnets: When `controlnet_conditioning_scale` is specified as `list`, it must have"
                    " the same length as the number of controlnets"
                )

        if not isinstance(control_guidance_start, (tuple, list)):
            control_guidance_start = [control_guidance_start]

        if not isinstance(control_guidance_end, (tuple, list)):
            control_guidance_end = [control_guidance_end]

        if len(control_guidance_start) != len(control_guidance_end):
            raise ValueError(
                f"`control_guidance_start` has {len(control_guidance_start)} elements, but `control_guidance_end` has "
                f"{len(control_guidance_end)} elements. Make sure to provide the same number of elements to each list."
            )

        if isinstance(self.controlnet, MultiControlNetModel):
            if len(control_guidance_start) != len(self.controlnet.nets):
                raise ValueError(
                    f"`control_guidance_start`: {control_guidance_start} has {len(control_guidance_start)} elements but "
                    f"there are {len(self.controlnet.nets)} controlnets available. Make sure to provide {len(self.controlnet.nets)}."
                )

        for start, end in zip(control_guidance_start, control_guidance_end):
            if start >= end:
                raise ValueError(f"control guidance start: {start} cannot be larger or equal to control guidance end: {end}.")
            if start < 0.0:
                raise ValueError(f"control guidance start: {start} can't be smaller than 0.")
            if end > 1.0:
                raise ValueError(f"control guidance end: {end} can't be larger than 1.0.")

        if generator is not None and isinstance(generator, list) and len(generator) != input_batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {input_batch_size}. Make sure the batch size matches the length of the generators."
            )

    def check_image(self, image, prompt, prompt_embeds):
        image_is_pil = isinstance(image, PIL.Image.Image)
        image_is_tensor = isinstance(image, torch.Tensor)
        image_is_np = isinstance(image, np.ndarray)
        image_is_pil_list = isinstance(image, list) and isinstance(image[0], PIL.Image.Image)
        image_is_tensor_list = isinstance(image, list) and isinstance(image[0], torch.Tensor)
        image_is_np_list = isinstance(image, list) and isinstance(image[0], np.ndarray)

        if (
            not image_is_pil
            and not image_is_tensor
            and not image_is_np
            and not image_is_pil_list
            and not image_is_tensor_list
            and not image_is_np_list
        ):
            raise TypeError(
                f"image must be passed and be one of PIL image, numpy array, torch tensor, list of PIL images, list of numpy arrays or list of torch tensors, but is {type(image)}"
            )

        if image_is_pil:
            image_batch_size = 1
        else:
            image_batch_size = len(image)

        if prompt is not None and isinstance(prompt, str):
            prompt_batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            prompt_batch_size = len(prompt)
        elif prompt_embeds is not None:
            prompt_batch_size = prompt_embeds.shape[0]

        if image_batch_size != 1 and image_batch_size != prompt_batch_size:
            raise ValueError(
                f"If image batch size is not 1, image batch size must be same as prompt batch size. image batch size: {image_batch_size}, prompt batch size: {prompt_batch_size}"
            )

    def mod_unet(self):
        attn_procs = {}
        for name in self.unet.attn_processors.keys():
            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                place_in_unet = "down"
            else:
                continue

            # Replace attention module
            if "attn2" in name:
                attn_procs[name] = CrossAttnProcessor(
                    attention_store=self.attention_store,
                    place_in_unet=place_in_unet
                )
            else:
                attn_procs[name] = AttnProcessor()

        self.unet.set_attn_processor(attn_procs)

    def prepare_conditioning_image(
        self, image: torch.tensor, num_images_per_prompt, device, dtype, do_classifier_free_guidance=False, guess_mode=False,
    ):
        image = image.repeat_interleave(num_images_per_prompt, dim=0)
        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    def prepare_latents(
        self, num_images_per_prompt, num_channels_latents, height, width, dtype, device, generator=None, latents=None, image=None, timestep=None, is_strength_max=True
    ):
        # Image is used mixed with random noise for the given timestep if strength is less than 1 and latents is None
        shape = (image.size(0)*num_images_per_prompt, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)

        image_latents = None
        if image is not None:
            image = image.to(device=device, dtype=dtype)
            if image.shape[1] == 4:
                image_latents = image
            else:
                image_latents = self._encode_vae_image(image=image, generator=generator)
            image_latents = image_latents.repeat_interleave(num_images_per_prompt, 0)

        # get the noisy_latents and scale it by the standard deviation required by the scheduler
        if latents is None:
            # Note in img2img when the strength is forced to 1, image_latents is not used for the initialization
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)  # latents as complete noise 
            noisy_latents = (noise * self.scheduler.init_noise_sigma) if image is None or is_strength_max else self.scheduler.add_noise(image_latents, noise, timestep)
        else:
            noise = latents.to(device)
            noisy_latents = noise * self.scheduler.init_noise_sigma

        return noisy_latents, noise, image_latents


    def prepare_mask_latents(
        self, mask: torch.tensor, masked_image: torch.tensor, num_images_per_prompt, height, width, dtype, device, generator=None, do_classifier_free_guidance=False
    ):
        # Resize the mask to latents shape as we concatenate the mask to the latents
        # We do that before converting to dtype to avoid breaking in case we're using cpu_offload and half precision
        mask = torch.nn.functional.interpolate(mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor))
        mask = mask.to(device=device, dtype=dtype)

        masked_image = masked_image.to(device=device, dtype=dtype)

        if masked_image.shape[1] == 4:
            masked_image_latents = masked_image
        else:
            masked_image_latents = self._encode_vae_image(masked_image, generator=generator)

        # replicate mask and masked_image_latents for each generation across batched prompts
        mask = mask.repeat_interleave(num_images_per_prompt, 0)
        masked_image_latents = masked_image_latents.repeat_interleave(num_images_per_prompt, 0)
        # cfg
        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents)

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        return mask, masked_image_latents

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator=None):
        def retrieve_latents(encoder_output, generator):
            if hasattr(encoder_output, "latent_dist"):
                return encoder_output.latent_dist.sample(generator)
            elif hasattr(encoder_output, "latents"):
                return encoder_output.latents
            else:
                raise AttributeError("Could not access latents of provided encoder_output")

        if isinstance(generator, list):
            image_latents = [retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i]) for i in range(image.shape[0])]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator)

        image_latents = self.vae.config.scaling_factor * image_latents

        return image_latents

    def enable_freeu(self, s1: float, s2: float, b1: float, b2: float):
        r"""Enables the FreeU mechanism as in https://arxiv.org/abs/2309.11497.

        The suffixes after the scaling factors represent the stages where they are being applied.

        Please refer to the [official repository](https://github.com/ChenyangSi/FreeU) for combinations of the values
        that are known to work well for different pipelines such as Stable Diffusion v1, v2, and Stable Diffusion XL.

        Args:
            s1 (`float`):
                Scaling factor for stage 1 to attenuate the contributions of the skip features. This is done to
                mitigate "oversmoothing effect" in the enhanced denoising process.
            s2 (`float`):
                Scaling factor for stage 2 to attenuate the contributions of the skip features. This is done to
                mitigate "oversmoothing effect" in the enhanced denoising process.
            b1 (`float`): Scaling factor for stage 1 to amplify the contributions of backbone features.
            b2 (`float`): Scaling factor for stage 2 to amplify the contributions of backbone features.
        """
        if not hasattr(self, "unet"):
            raise ValueError("The pipeline must have `unet` for using FreeU.")
        self.unet.enable_freeu(s1=s1, s2=s2, b1=b1, b2=b2)

    def disable_freeu(self):
        """Disables the FreeU mechanism if enabled."""
        self.unet.disable_freeu()

    def get_guidance_scale_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            timesteps (`torch.Tensor`):
                generate embedding vectors at these timesteps
            embedding_dim (`int`, *optional*, defaults to 512):
                dimension of the embeddings to generate
            dtype:
                data type of the generated embeddings

        Returns:
            `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    @property
    def clip_skip(self):
        return self._clip_skip

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        controlnet_prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        mask_image: PipelineImageInput = None,
        conditioning_image: PipelineImageInput = None,
        strength: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        noisy_latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 0.8,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        focus_prompt: Optional[List[str]] = None,
        focus_prompt_embeds = None,
        gradient_checkpointing: bool = False,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.
        
        MultiControlNet img2img/inpaint behaviour
            if image is not None and mask is None:
              start the latent as a combination of image_latents (iff strength < 1) and random noise.
            elif image is not None and mask is not None:
              if unet.input_channels == 9:
                the input of the unet is [latents, mask resized, masked_image_latents] 
              elif unet.input_channels == 4:
                the mask is used to sample partially from image_latents and partially from noisy latents

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            controlnet_prompt (`str` or `List[str]`, *optional*):
                The alternative prompt or prompts to pass to controlnet to guide image generation.
            image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.FloatTensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The ControlNet input condition to provide guidance to the `unet` for generation. If the type is
                specified as `torch.FloatTensor`, it is passed to ControlNet as is. `PIL.Image.Image` can also be
                accepted as an image. The dimensions of the output image defaults to `image`'s dimensions. If height
                and/or width are passed, `image` is resized accordingly. If multiple ControlNets are specified in
                `init`, images must be passed as a list such that each element of the list can be correctly batched for
                input to a single ControlNet.
            mask_image (`torch.Tensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, to mask `image`. White pixels in the mask will be
                repainted, while black pixels will be preserved. If `mask_image` is a PIL image, it will be converted
                to a single channel (luminance) before use. If it's a tensor, it should contain one color channel (L)
                instead of 3, so the expected shape would be `(B, H, W, 1)`.
            conditioning_image (`torch.FloatTensor`, `PIL.Image.Image`, `List[torch.FloatTensor]` or `List[PIL.Image.Image]`):
                The ControlNet input condition. ControlNet uses this input condition to generate guidance to Unet. If
                the type is specified as `Torch.FloatTensor`, it is passed to ControlNet as is. PIL.Image.Image` can
                also be accepted as an image. The control image is automatically resized to fit the output image.
            strength (`float`, *optional*):
                Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1. `image`
                will be used as a starting point, adding more noise to it the larger the `strength`. The number of
                denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will
                be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `image`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            noisy_latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original `unet`. If multiple ControlNets are specified in `init`, you can set
                the corresponding scale as a list.
            guess_mode (`bool`, *optional*, defaults to `False`):
                The ControlNet encoder tries to recognize the content of the input image even if you remove all
                prompts. A `guidance_scale` value between 3.0 and 5.0 is recommended.
            control_guidance_start (`float` or `List[float]`, *optional*, defaults to 0.0):
                The percentage of total steps at which the ControlNet starts applying.
            control_guidance_end (`float` or `List[float]`, *optional*, defaults to 1.0):
                The percentage of total steps at which the ControlNet stops applying.
            focus_prompt: (List(str), *optional*): The additional prompt to cllect attention inside the unet.
            focus_prompt_embeds (List(str), *optional*): Pre-generated text embeddings for the focus prompt. If not
                provided, text embeddings are generated from the `focus_prompt` input argument.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeine class.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = (
                mult * [control_guidance_start],
                mult * [control_guidance_end],
            )

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, controlnet_prompt, focus_prompt, conditioning_image, height, width, mask_image, negative_prompt, prompt_embeds, negative_prompt_embeds, focus_prompt_embeds, controlnet_conditioning_scale, control_guidance_start, control_guidance_end, callback_on_step_end_tensor_inputs, generator
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        if negative_prompt is not None and isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        if controlnet_prompt is not None and isinstance(controlnet_prompt, str):
            controlnet_prompt = [controlnet_prompt]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}

        if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        global_pool_conditions = (
            False if controlnet is None
            else controlnet.config.global_pool_conditions if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        ) 
        guess_mode = guess_mode or global_pool_conditions

        if focus_prompt:
            enable_prompt_focus = True
            if batch_size == 1:
                if isinstance(focus_prompt, str):
                    self.num_focus_prompts = 1
                    focus_prompt = [focus_prompt]
                elif isinstance(focus_prompt, list):
                    self.num_focus_prompts = len(focus_prompt)
            else:
                self.num_focus_prompts = 1 if isinstance(focus_prompt[0], str) else len(focus_prompt[0])
        elif focus_prompt_embeds is not None:
            enable_prompt_focus = True
            self.num_focus_prompts = focus_prompt_embeds.shape[0]
            if len(focus_prompt) == 0: focus_prompt = ["<unspecified>"]*self.num_focus_prompts  # patch
        else:
            enable_prompt_focus = False
            self.num_focus_prompts = 0

        if enable_prompt_focus:
            # initialize mask smoother, attention store and mod the unet
            self.attention_store = AttentionStore(store_averaged_over_steps=True, batch_size=batch_size*num_images_per_prompt, classifier_free_guidance=do_classifier_free_guidance, device=device)
            self.attn_cum_map = torch.tensor([], device=device)
            self.attn_cum_filter = torch.tensor([], device=device)
            self.attn_filter = None
            self.attn_mask = None
            self.mod_unet()

        # 3. Encode input prompt
        text_encoder_lora_scale = self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        prompt_embeds, prompt_ids, _ = self.encode_prompt(
            prompt,
            device,
            do_classifier_free_guidance,
            self.text_encoder,
            num_images_per_prompt=num_images_per_prompt,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
            return_tokenizer_output=True,
            return_tuple=False
        )  # (cfg*b*1*n,seq,hid)
        
        if controlnet is not None:
            encoder = self.controlnet_text_encoder if self.controlnet_text_encoder is not None else \
                      self.controlnet_image_encoder if self.controlnet_image_encoder is not None else self.text_encoder
            # flatten ((B)atches, (M)ulti controlnet (P)rompts)
            nets = len(self.controlnet.nets) if isinstance(self.controlnet, MultiControlNetModel) else 1
            controlnet_prompt = [[batch] * nets for batch in prompt] if controlnet_prompt is None else \
                                [[batch] if isinstance(batch, str) else batch for batch in controlnet_prompt]
            controlnet_prompt = [p for batch in controlnet_prompt for p in batch]
            controlnet_negative_prompt = [p for batch in [[batch] * nets for batch in negative_prompt] for p in batch] if negative_prompt is not None else None
            
            controlnet_prompt_embeds = self.encode_prompt(
                controlnet_prompt,
                device,
                do_classifier_free_guidance and not guess_mode,  # (CFG)
                encoder,
                num_images_per_prompt=num_images_per_prompt,  # (N)
                negative_prompt=controlnet_negative_prompt,
                lora_scale=text_encoder_lora_scale,
                return_tuple=False
            )  # (cfg*b*mp*n,seq,hid)
            if self.controlnet_prompt_seq_projection:
                indices = torch.arange(controlnet_prompt_embeds.size(0)).view((1+int(do_classifier_free_guidance and not guess_mode))*batch_size,nets,num_images_per_prompt)[:,-1].flatten()
                if hasattr(encoder, "visual_projection"):
                    controlnet_prompt_embeds[indices] = self.controlnet_image_encoder.visual_projection(controlnet_prompt_embeds[indices]).to(controlnet_prompt_embeds.dtype)
                elif hasattr(encoder, "text_projection"):
                    controlnet_prompt_embeds[indices] = self.controlnet_text_encoder.text_projection(controlnet_prompt_embeds[indices]).to(controlnet_prompt_embeds.dtype)
                else:
                    logger.warning("`controlnet_prompt_seq_projection` is True but no text_encoder with projection is given.")
    
        if enable_prompt_focus:
            # flatten batched lists of focus prompts
            if batch_size > 1 and self.num_focus_prompts > 1:
                focus_prompt = [p for batch in focus_prompt for p in batch]
            focus_prompt_embeds, focus_prompt_ids, focus_prompt_lengths = self.encode_prompt(
                focus_prompt,
                device,
                False,
                self.text_encoder,
                num_images_per_prompt=num_images_per_prompt,
                prompt_embeds=focus_prompt_embeds,
                return_tokenizer_output=True,
                return_tuple=False
            )  # (1*b*fp*n,seq,hid)
            # list of prompts in one batch for which the cross attention is stored by the AttentionStore (debug helper)
            prompts_per_batch = ([prompt] if isinstance(prompt, str) else [prompt[0]] if isinstance(prompt, list) else [None]) + ([focus_prompt] if isinstance(focus_prompt, str) else focus_prompt[:self.num_focus_prompts])
            # embeds to feed the unet
            prompt_embeds = torch.cat([prompt_embeds, focus_prompt_embeds], 0)  # ([(cfg*b*1*n)+(1*b*fp*n)],seq,hid) 

        # 4. Prepare image
        if controlnet is not None:
            # flatten (batches, multi controlnet conditioning)
            nets = len(self.controlnet.nets) if isinstance(self.controlnet, MultiControlNetModel) else 1
            if batch_size > 1 or nets > 1:
                if isinstance(conditioning_image, torch.Tensor):
                    conditioning_image = [el for el in conditioning_image]  # either you end up with chw, bchw or nchw is fine for the preprocessor  
                elif isinstance(conditioning_image, list) and isinstance(conditioning_image[0], list):
                    conditioning_image = [image for batch in conditioning_image for image in batch]
            conditioning_image = self.conditioning_image_processor.preprocess(conditioning_image, height=height, width=width)
            conditioning_image = self.prepare_conditioning_image(
                image=conditioning_image,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guess_mode=guess_mode,
            )  # (cfg*b*n,c,h,w)

        # Preprocess mask and image - resizes image and mask w.r.t height and width
        if image is None:
            init_image = None
            mask_image = None
        else:
            init_image = self.image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32).to(device)
            # repeat for the batch if needed
            init_image = init_image.repeat(batch_size // init_image.shape[0], 1, 1, 1)  # (b,c,h,w)
            if mask_image is not None:
                # Prepare mask latent
                init_mask = self.mask_processor.preprocess(mask_image, height=height, width=width).to(device)
                # repeat for the batch if needed 
                init_mask = init_mask.repeat(batch_size // init_mask.shape[0], 1, 1, 1)  # (b,c,h,w)
                masked_image = init_image.clone()
                masked_image[init_mask.expand(init_image.shape) > 0.5] = 0  # 0 in [-1,1] image because this is what sd wants
                height, width = init_image.shape[-2:]

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        # When strength is less than 1.0 in img2img we start from a latent partially mixed with the given image (i.e. no full noise)
        # therefore we can reduce the number of timestep of denoising (e.g. 50% if strength is 0.5)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps=num_inference_steps, strength=strength)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        self._num_timesteps = len(timesteps)

        # 6. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        num_channels_unet = self.unet.config.in_channels
        return_image_latents = num_channels_unet == 4
        is_strength_max = strength == 1.0

        # The noisy_latents_unscaled version and the image_latents are used only for the light inpainting
        noisy_latents, init_noise, image_latents = self.prepare_latents(
            num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            noisy_latents,
            image=init_image,                               # img2img
            timestep=latent_timestep,                       # img2img
            is_strength_max=is_strength_max,                # img2img
        )  # (b*n,c_vae,h_vae,w_vae) Note: cfg is done inside the loop for the noise

        if mask_image is not None:
            # Prepare mask latent variables
            mask, masked_image_latents = self.prepare_mask_latents(
                init_mask,
                masked_image,
                num_images_per_prompt,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                do_classifier_free_guidance,
            )  # (cfg*b*n,c,h_vae,w_vae) (cfg*b*n,c_vae,h_vae,w_vae)
            # append at the end a replica for focus prompts (b*n*fp,c/c_vae,h_vae,w_vae)
            mask = torch.cat([mask] + [mask[:batch_size*num_images_per_prompt]]*self.num_focus_prompts)
            masked_image_latents = torch.cat([masked_image_latents] + [masked_image_latents[:batch_size*num_images_per_prompt]]*self.num_focus_prompts)

        # 6.5 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim).to(device=device, dtype=noisy_latents.dtype)

        # 7. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e) for s, e in zip(control_guidance_start, control_guidance_end)]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        is_unet_compiled = is_compiled_module(self.unet)
        is_controlnet_compiled = is_compiled_module(self.controlnet)
        is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")

        attn_th, res, block_positions, filter_th = 0.92, 16, ["up"], 0.00
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Relevant thread: https://dev-discuss.pytorch.org/t/cudagraphs-in-pytorch-2-0/1428
                if (is_unet_compiled and is_controlnet_compiled) and is_torch_higher_equal_2_1:
                    torch._inductor.cudagraph_mark_step_begin()

                # expand the latents if we are doing classifier free guidance or if there are focus prompts [neg][pos][focus]
                latent_model_input = torch.cat([noisy_latents] * (1+int(do_classifier_free_guidance)+self.num_focus_prompts))  # ((fp+cfg)*b*n,c_vae,h_vae,w_vae)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # NOTE: for inpainting, in the masked area of the rgb image use the latent filling option you prefer:
                #   fill: default
                #   original: latent_model_input[mask.expand(latent_model_input.shape) > 0.5] = torch.mean(latent_model_input[mask.expand(latent_model_input.shape) > 0.5])
                #   latent nothing: latent_model_input[mask.expand(latent_model_input.shape) > 0.5] = torch.randn_like(latent_model_input)[mask.expand(latent_model_input.shape) > 0.5]
                #   latent noise: latent_model_input[mask.expand(latent_model_input.shape) > 0.5] = 0

                # controlnet(s) call
                if controlnet is not None:
                    control_model_input = torch.cat([noisy_latents] * (1+int(do_classifier_free_guidance and not guess_mode)))
                    control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                    control_attention_mask = torch.cat([self.attn_mask] * (1+int(do_classifier_free_guidance and not guess_mode))) if self.attn_mask is not None else None  # (cfg*b*n,1,h_block,w_block)
                    
                    if isinstance(controlnet_keep[i], list):
                        cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                    else:
                        controlnet_cond_scale = controlnet_conditioning_scale
                        if isinstance(controlnet_cond_scale, list):
                            controlnet_cond_scale = controlnet_cond_scale[0]
                        cond_scale = controlnet_cond_scale * controlnet_keep[i]

                    down_block_res_samples, mid_block_res_sample = split_multi_net(
                        self.controlnet, control_attention_mask,
                        controlnet_prompt_embeds, conditioning_image, cond_scale, guess_mode, batch_size*num_images_per_prompt,
                        **dict(sample=control_model_input, timestep=t, return_dict=False)
                    )

                    if guess_mode and do_classifier_free_guidance:
                        # Infered ControlNet only for the conditional batch.
                        # To apply the output of ControlNet to both the unconditional and conditional batches,
                        # add 0 to the unconditional batch to keep it unchanged.
                        down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                        mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

                    # append at the end (b*n*fp,c_block,h_block,w_block)
                    down_block_res_samples = [torch.cat([d]+self.num_focus_prompts*[torch.zeros_like(d[:batch_size*num_images_per_prompt])]) for d in down_block_res_samples]
                    mid_block_res_sample = torch.cat([mid_block_res_sample]+self.num_focus_prompts*[torch.zeros_like(mid_block_res_sample[:batch_size*num_images_per_prompt])])
                else:
                    down_block_res_samples = None
                    mid_block_res_sample = None

                # predict the noise residual
                if mask_image is not None and num_channels_unet == 9:
                    # Full inpaint: feed SD with [image, mask and masked image] to let the model predict noise conscious of the inpainting process
                    latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)  # ((fp+cfg)*b*n,(c_vae+c+c_vae),h_vae,w_vae)

                if gradient_checkpointing:
                    noise_pred = checkpoint.checkpoint(self.unet,
                    latent_model_input, t, prompt_embeds, None, timestep_cond, None, self.cross_attention_kwargs, None, 
                           None if controlnet is None else [d.detach() for d in down_block_res_samples] if not enable_prompt_focus or (i>0 and enable_prompt_focus) else None,
                           None if controlnet is None else mid_block_res_sample.detach() if not enable_prompt_focus or (i>0 and enable_prompt_focus) else None, None, None, False,
                    use_reentrant=False)[0]
                else:
                    noise_pred = self.unet(
                        sample=latent_model_input,
                        timestep=t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        down_block_additional_residuals=None if controlnet is None else down_block_res_samples if not enable_prompt_focus or (i>0 and enable_prompt_focus) else None,  # workaround (*) 
                        mid_block_additional_residual=None if controlnet is None else mid_block_res_sample if not enable_prompt_focus or (i>0 and enable_prompt_focus) else None,
                        return_dict=False,
                    )[0]

                # split guidance and focus prompts
                noise_pred = noise_pred.chunk(1 + int(do_classifier_free_guidance) + self.num_focus_prompts)
                if do_classifier_free_guidance:
                    noise_pred_neg, noise_pred_pos, _ = noise_pred[0], noise_pred[1], noise_pred[2:]  # neg, pos, focus
                    noise_pred = noise_pred_neg + guidance_scale * (noise_pred_pos - noise_pred_neg)
                else:
                    noise_pred, _ = noise_pred[0], noise_pred[1:]

                # compute the previous noisy sample x_t -> x_t-1
                denoised_latents = self.scheduler.step(noise_pred, t, noisy_latents, **extra_step_kwargs, return_dict=False)[0]

                if mask_image is not None and num_channels_unet == 4:
                    # Light inpaint: overwrite the latent sampling pixel values from mask
                    init_noisy_latents = image_latents
                    latents_inpaint_mask = mask.chunk(1+int(do_classifier_free_guidance)+self.num_focus_prompts)[0]

                    if i < len(timesteps) - 1:
                        noise_timestep = timesteps[i + 1]
                        init_noisy_latents = self.scheduler.add_noise(image_latents, init_noise, torch.tensor([noise_timestep]))

                    denoised_latents = (1 - latents_inpaint_mask) * init_noisy_latents + latents_inpaint_mask * denoised_latents

                if enable_prompt_focus:
                    tokens_position = torch.cat([torch.stack([p[0] == fp[torch.logical_and(fp > 0, fp < 49406)][0]] * num_images_per_prompt) for p, fp in zip(prompt_ids.chunk(batch_size), focus_prompt_ids.chunk(batch_size))])
                    attn_mask, attn_map = self.attention_store.get_cross_attention_mask(block_positions, res, 0, tokens_position, attn_th)
                    self.attn_cum_filter =  attn_map if self.attn_cum_filter.size(0) == 0 else torch.stack([self.attn_cum_filter, attn_map]).sum(0)  # (filter helper)
                    self.attn_filter = attn_map.flatten(start_dim=1).max(-1)[0]>filter_th

                    self.attn_mask = attn_mask.unsqueeze(1)  # store to be used in next step
                    self.attn_cum_map = attn_mask if self.attn_cum_map.size(0) == 0 else torch.stack([self.attn_cum_map, attn_mask]).sum(0)  # (viz helper)
                    # reset accumulated attention maps in the store
                    self.attention_store.step(False)

                # Cycle and callbacks
                noisy_latents = denoised_latents

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        latents = noisy_latents

        # If we do sequential model offloading, let's offload unet and controlnet manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.to("cpu")
            torch.cuda.empty_cache()

        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        elif output_type == "pt":
            image = self._decode_vae_latents(latents)
            image = self.image_processor.postprocess(image, output_type="pt")
            has_nsfw_concept = None
        elif output_type == "pil":
            # 8. Post-processing
            image = self._decode_vae_latents(latents)
            # (viz helper) uncomment to show the masked controlnet area dictated by the focus prompt 
            # cross_mask = F.interpolate(self.atten_cum_map, image.shape[-2:]).expand_as(image)
            # cross_mask = (cross_mask/cross_mask.max())*2
            # image[:, 0] = image[:, 0] + cross_mask[:, 0]
            image = self.image_processor.postprocess(image, output_type="np")

            # 9. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

            # 10. Convert to PIL
            image = self.numpy_to_pil(image)
        else:
            # 8. Post-processing
            image = self._decode_vae_latents(latents)
            image = self.image_processor.postprocess(image, output_type="np")

            # 9. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)        

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return image, has_nsfw_concept

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
