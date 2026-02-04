"""
Sampler Nodes with Progress Tracking for ComfyUI
Redis-based batch and step progress reporting
"""

import json
import logging

import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview

logger = logging.getLogger(__name__)


class KSamplerProgressNode:
    """KSampler with Redis progress tracking for batch processing."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "redis_url": ("STRING", {"default": "redis://localhost:6379"}),
                "job_id": ("STRING", {"default": ""}),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("LATENT",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "sample"
    CATEGORY = "IXIWORKS/Sampler"

    def sample(self, model, positive, negative, latent_image,
               seed, steps, cfg, sampler_name, scheduler, denoise,
               redis_url, job_id):

        # Extract single values from lists
        redis_url_str = redis_url[0] if isinstance(redis_url, list) else redis_url
        job_id_str = job_id[0] if isinstance(job_id, list) else job_id

        # Get batch size from list inputs
        batch_total = len(positive)

        # Extract single values for sampler params (same for all batches)
        _model = model[0] if isinstance(model, list) else model
        _steps = steps[0] if isinstance(steps, list) else steps
        _cfg = cfg[0] if isinstance(cfg, list) else cfg
        _sampler_name = sampler_name[0] if isinstance(sampler_name, list) else sampler_name
        _scheduler = scheduler[0] if isinstance(scheduler, list) else scheduler
        _denoise = denoise[0] if isinstance(denoise, list) else denoise

        # Setup Redis connection
        redis_client = None
        redis_key = None
        if redis_url_str and job_id_str:
            try:
                import redis
                redis_client = redis.from_url(redis_url_str)
                redis_key = f"comfyui:progress:{job_id_str}"
                logger.info(f"[IXIWORKS] KSamplerProgress: Connected to Redis, key={redis_key}")
            except ImportError:
                logger.error("[IXIWORKS] KSamplerProgress: redis package not installed")
            except Exception as e:
                logger.error(f"[IXIWORKS] KSamplerProgress: Redis connection failed - {e}")

        def update_progress(batch_current, step_current, step_total):
            """Update progress in Redis"""
            if redis_client and redis_key:
                try:
                    total_steps_all = batch_total * step_total
                    completed_steps = (batch_current - 1) * step_total + step_current
                    overall_progress = completed_steps / total_steps_all if total_steps_all > 0 else 0

                    progress_data = {
                        "batch_current": batch_current,
                        "batch_total": batch_total,
                        "step_current": step_current,
                        "step_total": step_total,
                        "progress": round(overall_progress, 4),
                    }
                    redis_client.set(redis_key, json.dumps(progress_data), ex=3600)  # 1 hour TTL
                except Exception as e:
                    logger.warning(f"[IXIWORKS] KSamplerProgress: Redis update failed - {e}")

        results = []

        for batch_idx in range(batch_total):
            # Get batch-specific inputs
            _positive = positive[batch_idx] if batch_idx < len(positive) else positive[-1]
            _negative = negative[batch_idx] if batch_idx < len(negative) else negative[-1]
            _latent = latent_image[batch_idx] if batch_idx < len(latent_image) else latent_image[-1]
            _seed = seed[batch_idx] if batch_idx < len(seed) else seed[-1]

            # Update progress at batch start
            update_progress(batch_idx + 1, 0, _steps)

            logger.info(f"[IXIWORKS] KSamplerProgress: Batch {batch_idx + 1}/{batch_total}, seed={_seed}")

            # Create step callback
            current_batch = batch_idx + 1

            def make_callback(batch_num):
                def callback(step, x0, x, total_steps):
                    update_progress(batch_num, step + 1, total_steps)
                return callback

            step_callback = make_callback(current_batch)

            # Prepare latent
            latent = _latent.copy()
            latent_samples = latent["samples"]

            # Get noise
            batch_inds = latent.get("batch_index", None)
            noise = comfy.sample.prepare_noise(latent_samples, _seed, batch_inds)

            # Create sampler and sigmas
            sampler = comfy.samplers.KSampler(
                _model, steps=_steps, device=_model.load_device,
                sampler=_sampler_name, scheduler=_scheduler,
                denoise=_denoise, model_options=_model.model_options,
            )

            # Sample with callback
            samples = sampler.sample(
                noise, _positive, _negative, cfg=_cfg,
                latent_image=latent_samples, start_step=0,
                last_step=_steps, force_full_denoise=True,
                denoise_mask=None, sigmas=None, callback=step_callback,
                disable_pbar=False, seed=_seed,
            )

            # Store result
            result_latent = latent.copy()
            result_latent["samples"] = samples
            results.append(result_latent)

            # Update progress at batch end
            update_progress(batch_idx + 1, _steps, _steps)

        # Final progress update
        if redis_client and redis_key:
            try:
                progress_data = {
                    "batch_current": batch_total,
                    "batch_total": batch_total,
                    "step_current": _steps,
                    "step_total": _steps,
                    "progress": 1.0,
                    "status": "completed",
                }
                redis_client.set(redis_key, json.dumps(progress_data), ex=3600)
            except Exception:
                pass

        logger.info(f"[IXIWORKS] KSamplerProgress: Completed {batch_total} batches")
        return (results,)


NODE_CLASS_MAPPINGS = {
    "KSamplerProgress": KSamplerProgressNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KSamplerProgress": "KSampler Progress (Redis)",
}
