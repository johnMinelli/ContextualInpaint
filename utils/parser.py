import argparse

class Train_args:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="ControlNet training script.")
        self.parser.add_argument("--pretrained_model_name_or_path", type=str, default=None, required=True, help="Path to pretrained model or model identifier from huggingface.co/models.",)
        self.parser.add_argument("--controlnet_model_name_or_path", type=str, default=None, help="Path to pretrained controlnet model or model identifier from huggingface.co/models. If not specified controlnet weights are initialized from unet.",)
        self.parser.add_argument("--revision", type=str, default=None, required=False, help=("Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be float32 precision."),)
        self.parser.add_argument("--tokenizer_name", type=str, default=None, help="Pretrained tokenizer name or path if not the same as model_name",)
        self.parser.add_argument("--output_dir", type=str, default="controlnet-model", help="The output directory where the model predictions and checkpoints will be written.",)
        self.parser.add_argument("--cache_dir", type=str, default=None, help="The directory where the downloaded models and datasets will be stored.",)
        self.parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
        self.parser.add_argument("--sd_unlock", type=int, default=-1, help="Number of epochs after which we unlock the training for unet weights (only the second part of sd unet architecture). Set to -1 to disable it.")
        self.parser.add_argument("--resolution", type=int, default=512, help=("The resolution for input images, all the images in the train/validation dataset will be resized to this resolution"),)
        self.parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
        self.parser.add_argument("--num_train_epochs", type=int, default=10)
        self.parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",)
        self.parser.add_argument("--checkpointing_steps", type=int, default=5000, help=(
                "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume`. "
                "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
                "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
                "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step instructions."),)
        self.parser.add_argument("--checkpoints_total_limit", type=int, default=None, help=("Max number of checkpoints to store."),)
        self.parser.add_argument("--resume", type=str, default=None, help=("Whether training should be resumed from a previous checkpoint. Use a path saved by `--checkpointing_steps`, or `latest` to automatically select the last available checkpoint."),)
        self.parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
        self.parser.add_argument("--gradient_checkpointing", action="store_true", help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",)
        self.parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598). `w` of equation 2. of [Imagen Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.")
        self.parser.add_argument("--learning_rate", type=float, default=5e-6, help="Initial learning rate (after the potential warmup period) to use.",)
        self.parser.add_argument("--scale_lr", action="store_true", default=False, help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",)
        self.parser.add_argument("--lr_scheduler", type=str, default="constant", help=('The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]'),)
        self.parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
        self.parser.add_argument("--lr_num_cycles", type=int, default=1, help="Number of hard resets of the lr in cosine_with_restarts scheduler.", )
        self.parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
        self.parser.add_argument("--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes.")
        self.parser.add_argument("--use_classemb", action="store_true", help="Whether or not to use Class embedding conditioning for the training of the ControlNet.")
        self.parser.add_argument("--dataloader_num_workers", type=int,default=0, help=("Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."),)
        self.parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
        self.parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
        self.parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
        self.parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
        self.parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
        self.parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
        self.parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
        self.parser.add_argument("--hub_model_id", type=str, default=None, help="The name of the repository to keep in sync with the local `output_dir`.",)
        self.parser.add_argument("--logging_dir", type=str, default="logs", help=("[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."),)
        self.parser.add_argument("--allow_tf32", action="store_true", help=("Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"),)
        self.parser.add_argument("--log", type=str, default=None, help=('The integration to report the results and logs to. Supported platforms are `"tensorboard"` (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'),)
        self.parser.add_argument("--lora", type=str, default=None, help=('LoRA model checkpoint to use for training'),)
        self.parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"], help=(
                "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
                " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
                " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."),)
        self.parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
        self.parser.add_argument("--enable_cpu_offload", action="store_true", help="Whether or not to use accelerator's model offload functionalities.")
        self.parser.add_argument("--set_grads_to_none", action="store_true", help=(
                "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
                " behaviors, so disable this argument if it causes any problems. More info:"
                " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"),)
        self.parser.add_argument("--train_data_dir", nargs='+', type=str, default=None, help="A folder containing preprocessed training data.")
        self.parser.add_argument("--train_data_file", type=str, default="prompt.json", help="Name of the file to be searched in the `train_data_dir` containing training samples details.")
        self.parser.add_argument("--proportion_empty_prompts", type=float, default=0, help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).")
        self.parser.add_argument("--validation_file", type=str, default="data/validation/validation.json", help="A json file detailing the files for validation which happens every `--validation_steps` and logged to `--log`.")
        self.parser.add_argument("--num_validation_images", type=int, default=4, help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair")
        self.parser.add_argument("--validation_steps", type=int, default=200000000, help="Run validation every X steps. Validation consists of running the prompt `args.validation_prompt` multiple times: `args.num_validation_images` and logging the images.")
        self.parser.add_argument("--tracker_project_name", type=str, default="train_controlnet", help="The `project_name` argument passed to Accelerator.init_trackers for more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator")
        self.parser.add_argument("--rank", type=int, default=4, help="The dimension of the LoRA update matrices. Check LyCORICE library for more info.")
        self.parser.add_argument("--alpha_rank", type=int, default=4, help="The weight scaling LoRA update matrices. Check LyCORICE library for more info.")
        self.parser.add_argument("--lycorice_algo", type=str, default="full", choices=["full", "loha", "lora", "loka"], help="The algorithm used for LyCORICE library.")

    def parse_args(self, input_args=None):
        if input_args is not None:
            args = self.parser.parse_args(input_args)
        else:
            args = self.parser.parse_args()

        if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
            raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")
    
        if args.resolution % 8 != 0:
            raise ValueError("`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder.")
    
        return args

class Eval_args:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="ControlNet eval script.")
        self.parser.add_argument("--pretrained_model_name_or_path", type=str, default=None, required=True, help="Path to pretrained model or model identifier from huggingface.co/models.",)
        self.parser.add_argument("--controlnet_model_name_or_path", nargs='+', type=str, default=None, help="Path to pretrained controlnet model or model identifier from huggingface.co/models. If not specified controlnet weights are initialized from unet.",)
        self.parser.add_argument("--lora_path", type=str, default=None, help="Path to Lycoris network (fine-tuned unet) pretrained model.",)
        self.parser.add_argument("--revision", type=str, default=None, required=False, help=("Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be float32 precision."),)
        self.parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
        self.parser.add_argument("--batch_size", type=int, default=1, help="Number of images per batch.")
        self.parser.add_argument("--resolution", type=int, default=512, help=("The resolution for input images, all the images in the train/validation dataset will be resized to this resolution"),)
        self.parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598). `w` of equation 2. of [Imagen Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.")
        self.parser.add_argument("--log", type=str, default=None, help=('The integration to report the results and logs to. Supported platforms are `"tensorboard"` (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'),)
        self.parser.add_argument("--log_run_id", type=str, default=None, help=('Id of the run you want to resume.'),)
        self.parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"], help=(
                "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
                " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
                " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."),)
        self.parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
        self.parser.add_argument("--enable_cpu_offload", action="store_true", help="Whether or not to use accelerator's model offload functionalities.")
        self.parser.add_argument("--evaluation_file", type=str, default="data/validation/evaluation.json", help="A json file detailing the files for validation which happens every `--validation_steps` and logged to `--log`.")
        self.parser.add_argument("--train_data_dir", nargs='+', type=str, default=None, help="A folder containing preprocessed training data.")
        self.parser.add_argument("--train_data_file", type=str, default="prompt.json", help="Name of the file to be searched in the `train_data_dir` containing training samples details.")
        self.parser.add_argument("--steps", type=int, default=50, help="Denoising steps.")
        self.parser.add_argument("--num_validation_images", type=int, default=4, help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",)
        self.parser.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",)

    def parse_args(self, input_args=None):
        if input_args is not None:
            args = self.parser.parse_args(input_args)
        else:
            args = self.parser.parse_args()
    
        if args.resolution % 8 != 0:
            raise ValueError("`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder.")
    
        return args

class Optim_args:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="ControlNet eval script.")
        self.parser.add_argument("--pretrained_model_name_or_path", type=str, default=None, required=True, help="Path to pretrained model or model identifier from huggingface.co/models.",)
        self.parser.add_argument("--controlnet_model_name_or_path", nargs='+', type=str, default=None, help="Path to pretrained controlnet model or model identifier from huggingface.co/models. If not specified controlnet weights are initialized from unet.",)
        self.parser.add_argument("--revision", type=str, default=None, required=False, help=("Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be float32 precision."),)
        self.parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
        self.parser.add_argument("--resolution", type=int, default=512, help=("The resolution for input images, all the images in the train/validation dataset will be resized to this resolution"),)
        self.parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598). `w` of equation 2. of [Imagen Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.")
        self.parser.add_argument("--log", type=str, default=None, help=('The integration to report the results and logs to. Supported platforms are `"tensorboard"` (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'),)
        self.parser.add_argument("--log_run_id", type=str, default=None, help=('Id of the run you want to resume.'),)
        self.parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"], help=(
                "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
                " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
                " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."),)
        self.parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
        self.parser.add_argument("--enable_cpu_offload", action="store_true", help="Whether or not to use accelerator's model offload functionalities.")
        self.parser.add_argument("--evaluation_file", type=str, default="data/validation/evaluation.json", help="A json file detailing the files for validation which happens every `--validation_steps` and logged to `--log`.")
        self.parser.add_argument("--steps", type=int, default=50, help="Denoising steps.")
        self.parser.add_argument("--num_validation_images", type=int, default=4, help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",)
        self.parser.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",)
        self.parser.add_argument("--dataloader_num_workers", type=int,default=0, help=("Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."),)
        self.parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
        self.parser.add_argument("--logging_dir", type=str, default="logs", help=("[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."),)
        self.parser.add_argument("--allow_tf32", action="store_true", help=("Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"),)
        self.parser.add_argument("--set_grads_to_none", action="store_true", help=(
                "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
                " behaviors, so disable this argument if it causes any problems. More info:"
                " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"),)
        self.parser.add_argument("--proportion_empty_prompts", type=float, default=0, help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).")
        self.parser.add_argument("--validation_file", type=str, default="data/validation/validation.json", help="A json file detailing the files for validation which happens every `--validation_steps` and logged to `--log`.")
        self.parser.add_argument("--validation_steps", type=int, default=250, help="Run validation every X steps. Validation consists of running the prompt `args.validation_prompt` multiple times: `args.num_validation_images` and logging the images.")
        self.parser.add_argument("--tracker_project_name", type=str, default="train_controlnet", help="The `project_name` argument passed to Accelerator.init_trackers for more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator")
        self.parser.add_argument("--batch_size", type=int, default=1, help="Batch size (per device) for the generator dataloader.")
        self.parser.add_argument("--learning_rate", type=float, default=1e-5, help="Initial learning rate (after the potential warmup period) to use.",)
        self.parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
        self.parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
        self.parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
        self.parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")

    def parse_args(self, input_args=None):
        if input_args is not None:
            args = self.parser.parse_args(input_args)
        else:
            args = self.parser.parse_args()
    
        if args.resolution % 8 != 0:
            raise ValueError("`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder.")
    
        return args
