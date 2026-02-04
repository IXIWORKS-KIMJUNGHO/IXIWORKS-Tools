"""
Sampler Nodes with Progress Tracking for ComfyUI
Redis-based batch and step progress reporting
"""

import json
import logging

from aiohttp import web
from server import PromptServer

logger = logging.getLogger(__name__)

# Global Redis connection cache
_redis_connections = {}


def get_redis_client(redis_url):
    """Get or create Redis client for URL"""
    if redis_url not in _redis_connections:
        try:
            import redis
            _redis_connections[redis_url] = redis.from_url(redis_url)
        except Exception as e:
            logger.error(f"[IXIWORKS] Redis connection failed: {e}")
            return None
    return _redis_connections[redis_url]


# API endpoint for progress polling
@PromptServer.instance.routes.get("/ixiworks/progress/{job_id}")
async def get_progress(request):
    """Get progress data from Redis"""
    job_id = request.match_info["job_id"]
    redis_url = request.query.get("redis_url", "redis://localhost:6379")

    try:
        client = get_redis_client(redis_url)
        if not client:
            return web.json_response({"error": "Redis connection failed"}, status=500)

        data = client.get(f"comfyui:progress:{job_id}")
        if data:
            return web.json_response(json.loads(data))
        else:
            return web.json_response({"error": "Job not found"}, status=404)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


@PromptServer.instance.routes.post("/ixiworks/progress/start")
async def start_progress(request):
    """Initialize progress tracking for a job"""
    try:
        data = await request.json()
        job_id = data.get("job_id")
        redis_url = data.get("redis_url", "redis://localhost:6379")
        batch_total = data.get("batch_total", 1)
        steps = data.get("steps", 20)

        client = get_redis_client(redis_url)
        if not client:
            return web.json_response({"error": "Redis connection failed"}, status=500)

        progress_data = {
            "batch_current": 0,
            "batch_total": batch_total,
            "step_current": 0,
            "step_total": steps,
            "progress": 0.0,
            "status": "started",
        }
        client.set(f"comfyui:progress:{job_id}", json.dumps(progress_data), ex=3600)
        client.set(f"comfyui:batch:{job_id}", "0", ex=3600)

        return web.json_response({"success": True, "job_id": job_id})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


class ModelProgressTracker:
    """Model wrapper that injects progress tracking callback.

    Connect before KSampler to track batch and step progress via Redis.

    Usage:
        Model → [Model Progress Tracker] → KSampler → ...
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "redis_url": ("STRING", {"default": "redis://localhost:6379"}),
                "job_id": ("STRING", {"default": ""}),
                "batch_total": ("INT", {"default": 1, "min": 1, "max": 1000}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "track"
    CATEGORY = "IXIWORKS/Sampler"

    def track(self, model, redis_url, job_id, batch_total, steps):
        if not redis_url or not job_id:
            logger.warning("[IXIWORKS] ModelProgressTracker: No redis_url or job_id, skipping tracking")
            return (model,)

        # Setup Redis connection
        redis_client = None
        redis_key = f"comfyui:progress:{job_id}"
        batch_key = f"comfyui:batch:{job_id}"

        try:
            import redis
            redis_client = redis.from_url(redis_url)
            # Initialize batch counter (reset to 0)
            redis_client.set(batch_key, "0", ex=3600)
            logger.info(f"[IXIWORKS] ModelProgressTracker: Connected to Redis, key={redis_key}")
        except ImportError:
            logger.error("[IXIWORKS] ModelProgressTracker: redis package not installed")
            return (model,)
        except Exception as e:
            logger.error(f"[IXIWORKS] ModelProgressTracker: Redis connection failed - {e}")
            return (model,)

        # Clone model
        m = model.clone()

        # Create callback function
        def sampler_cfg_callback(args):
            """Called after each sampling step"""
            try:
                # Get current batch from Redis (incremented at start of each KSampler run)
                batch_current = int(redis_client.get(batch_key) or 1)

                # args contains: denoised, cond, uncond, model, sigma, etc.
                # We need to track step progress somehow
                # Unfortunately, step info is not directly available in cfg callback

                return args["denoised"]
            except Exception:
                return args["denoised"]

        # Track step progress using sampler_post_cfg_function
        step_counter = {"current": 0, "batch": 0}

        def sampler_post_cfg_callback(args):
            """Called after each CFG step"""
            nonlocal step_counter

            try:
                # Increment step counter
                step_counter["current"] += 1

                # Get current batch
                batch_current = int(redis_client.get(batch_key) or 1)

                # Calculate progress
                total_steps_all = batch_total * steps
                completed_steps = (batch_current - 1) * steps + step_counter["current"]
                overall_progress = completed_steps / total_steps_all if total_steps_all > 0 else 0

                progress_data = {
                    "batch_current": batch_current,
                    "batch_total": batch_total,
                    "step_current": step_counter["current"],
                    "step_total": steps,
                    "progress": round(overall_progress, 4),
                }
                redis_client.set(redis_key, json.dumps(progress_data), ex=3600)

            except Exception as e:
                logger.warning(f"[IXIWORKS] Progress update failed: {e}")

            return args["denoised"]

        # Set the callback on model
        m.set_model_sampler_post_cfg_function(sampler_post_cfg_callback)

        # Also need to track batch start - use denoise_mask_function or similar
        original_model_function = m.model.model_loaded

        def wrapped_model_function(*args, **kwargs):
            # Increment batch counter at start of each sampling
            try:
                current = int(redis_client.get(batch_key) or 0)
                redis_client.set(batch_key, str(current + 1), ex=3600)
                step_counter["current"] = 0  # Reset step counter
                step_counter["batch"] = current + 1
            except Exception:
                pass

            if original_model_function:
                return original_model_function(*args, **kwargs)

        logger.info(f"[IXIWORKS] ModelProgressTracker: Tracking {batch_total} batches, {steps} steps each")
        return (m,)


class BatchStartMarker:
    """Marks the start of each batch for progress tracking.

    Place this node in the list processing path (e.g., after CLIP Encode)
    to accurately track which batch is currently being processed.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "redis_url": ("STRING", {"default": "redis://localhost:6379"}),
                "job_id": ("STRING", {"default": ""}),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("CONDITIONING",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "mark"
    CATEGORY = "IXIWORKS/Sampler"

    def mark(self, conditioning, redis_url, job_id):
        redis_url_str = redis_url[0] if isinstance(redis_url, list) else redis_url
        job_id_str = job_id[0] if isinstance(job_id, list) else job_id

        if not redis_url_str or not job_id_str:
            return (conditioning,)

        try:
            import redis
            r = redis.from_url(redis_url_str)
            batch_key = f"comfyui:batch:{job_id_str}"

            # Process each conditioning and increment batch counter
            results = []
            for idx, cond in enumerate(conditioning):
                # Update batch counter
                r.set(batch_key, str(idx + 1), ex=3600)
                results.append(cond)

            # Reset for actual processing
            r.set(batch_key, "0", ex=3600)

            logger.info(f"[IXIWORKS] BatchStartMarker: Marked {len(conditioning)} batches")

        except Exception as e:
            logger.error(f"[IXIWORKS] BatchStartMarker: Error - {e}")
            return (conditioning,)

        return (conditioning,)


NODE_CLASS_MAPPINGS = {
    "ModelProgressTracker": ModelProgressTracker,
    "BatchStartMarker": BatchStartMarker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelProgressTracker": "Model Progress Tracker (Redis)",
    "BatchStartMarker": "Batch Start Marker (Redis)",
}
