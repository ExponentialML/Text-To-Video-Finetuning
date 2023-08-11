import torch

class DenoiseCallback(object):
    def __init__(
        self,
        pipeline,
        latents=None,
        rand_noise=None,
        cond_frames=None,
        slice_input_func=None
    ):
        self.pipeline = pipeline
        self.latents = latents
        self.rand_noise = rand_noise
        self.cond_frames = cond_frames
        self.slice_input_func = slice_input_func
    
    def update_callback_conditions(self, latents, rand_noise):
        self.latents = latents
        self.rand_noise = rand_noise

    def callback(self, i, t, forward_latents):
        if None in [self.latents, self.rand_noise, self.cond_frames, self.slice_input_func]:
            return

        original_latents = self.latents.clone()
        
        # Get the current timesteps
        timesteps = self.pipeline.scheduler.timesteps 
        timesteps[-1] = 1

        non_cond_frames = -original_latents.shape[2]
        new_forward = self.slice_input_func(forward_latents.clone(), non_cond_frames)

        noisy_original = self.pipeline.scheduler.add_noise(
            self.slice_input_func(original_latents, self.cond_frames, is_stop=True), 
            self.slice_input_func(self.rand_noise, self.cond_frames, is_stop=True), 
            timesteps[i]
        )

        masked_latents = torch.cat((noisy_original, new_forward), dim=2)

        forward_latents.copy_(masked_latents)