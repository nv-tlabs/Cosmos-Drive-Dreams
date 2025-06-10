import ray
from tqdm import tqdm

def wait_for_futures(futures, desc='Processing all tasks'):
    # Monitor progress using tqdm
    progress_bar = tqdm(total=len(futures), desc=desc)
    while len(futures):
        done, futures = ray.wait(futures)
        progress_bar.update(len(done))
        for obj_ref in done:
            try:
                ray.get(obj_ref)
            except Exception as e:
                print(f"Exception in processing video: {e}")

    progress_bar.close()


def ray_remote(use_ray=True, **ray_kwargs):
    """
    Custom decorator to allow switching between single-process debugging mode and Ray distributed mode.

    :param use_ray: Whether to enable Ray distributed functionality. If False, return the function itself.
    :param ray_kwargs: If Ray is enabled, pass keyword arguments to @ray.remote.
    """
    def decorator(func):
        if use_ray:
            # If Ray is enabled, apply @ray.remote and return the remote object
            return ray.remote(**ray_kwargs)(func)
        else:
            # If Ray is not enabled, return the original function (single-process execution)
            return func
        
    return decorator
