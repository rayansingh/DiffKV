import os
import copy
from functools import partial
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Tuple, Union

from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig)
from vllm.engine.ray_utils import RayWorkerVllm, initialize_cluster, ray


if ray:
    # Ray 2.x compatibility: ray.air.util.torch_dist was removed
    try:
        from ray.air.util.torch_dist import init_torch_dist_process_group
    except (ImportError, ModuleNotFoundError):
        # For Ray 2.x, implement a replacement for init_torch_dist_process_group
        import socket
        
        def init_torch_dist_process_group(workers, backend="nccl"):
            """Initialize torch distributed process group for Ray 2.x
            
            This function sets up the necessary environment variables on each worker
            so that torch.distributed can initialize properly.
            """
            # Get world size
            world_size = len(workers)
            
            # Use Ray's head node as the master
            master_addr = ray.util.get_node_ip_address()
            
            # Get a free port for distributed initialization
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))
                s.listen(1)
                master_port = s.getsockname()[1]
            
            # Create the distributed init method URL
            distributed_init_method = f"tcp://{master_addr}:{master_port}"
            
            # Set environment variables on each worker
            refs = []
            for rank, worker in enumerate(workers):
                # Call the set_environment_variables method we added to RayWorkerVllm
                ref = worker.set_environment_variables.remote(
                    rank, world_size, master_addr, master_port, distributed_init_method
                )
                refs.append(ref)
            
            # Wait for all workers to complete
            ray.get(refs)
            
            return distributed_init_method
    
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup
    
    
class Orchestrator:
    ''' The Orchestrator class is responsible for managing the distributed cluster and workers.
        In the case of multiple workers, the cluster is initialized with Ray 
    '''
    
    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
    ) -> None:
        self.model_config = model_config
        self.cache_config = cache_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        # initialize the cluster
        self.distributed_init_method: str = ''
        self.placement_group: "PlacementGroup" = None
        self.workers = []
    
    def initialize_cluster(self) -> None:
        self.distributed_init_method, self.placement_group = initialize_cluster(
            self.parallel_config)
        # Create the parallel GPU workers.
        if self.parallel_config.worker_use_ray:
            # Disable Ray usage stats collection.
            ray_usage = os.environ.get("RAY_USAGE_STATS_ENABLED", "0")
            if ray_usage != "1":
                os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
            self._init_workers_ray(self.placement_group)
        else:
            self._init_workers(self.distributed_init_method)
    
    def _init_workers(self, distributed_init_method: str) -> None:
        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        from vllm.worker.worker import Worker

        assert self.parallel_config.world_size == 1, (
            "Ray is required if parallel_config.world_size > 1.")

        self.workers: List[Worker] = []
        worker = Worker(
            self.model_config,
            self.parallel_config,
            self.scheduler_config,
            0,
            distributed_init_method,
        )
        self.workers.append(worker)
        self._set_worker_ids()
        self.run_workers(
            "init_model",
            get_all_outputs=True,
        )
        self.run_workers(
            "load_model",
            get_all_outputs=True,
            max_concurrent_workers=self.parallel_config.
            max_parallel_loading_workers,
            kv_buffer_size=self.cache_config.kv_buffer_size,
            max_kv_slots=self.cache_config.max_kv_slots,
        )
    
    def _init_workers_ray(self, placement_group: "PlacementGroup",
                          **ray_remote_kwargs):
        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        from vllm.worker.worker import Worker
        
        self.workers: List[Worker] = []
        for bundle in placement_group.bundle_specs:
            if not bundle.get("GPU", 0):
                continue
            if self.parallel_config.tensor_parallel_size == 1:
                num_gpus = self.cache_config.gpu_memory_utilization
            else:
                num_gpus = 1
            worker = ray.remote(
                num_cpus=0,
                num_gpus=num_gpus,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=placement_group,
                    placement_group_capture_child_tasks=True),
                **ray_remote_kwargs,
            )(RayWorkerVllm).remote(self.model_config.trust_remote_code)
            self.workers.append(worker)

        # Initialize torch distributed process group for the workers.
        distributed_init_method = init_torch_dist_process_group(self.workers, backend="nccl")
        model_config = copy.deepcopy(self.model_config)
        parallel_config = copy.deepcopy(self.parallel_config)
        scheduler_config = copy.deepcopy(self.scheduler_config)
        
        # Create a closure that uses the distributed_init_method
        def worker_init_fn():
            # Get the distributed_init_method from the RayWorkerVllm wrapper
            import os
            distributed_init = os.getenv("MASTER_ADDR")
            if distributed_init:
                distributed_init = f"tcp://{os.getenv('MASTER_ADDR')}:{os.getenv('MASTER_PORT')}"
            return Worker(
                model_config,
                parallel_config,
                scheduler_config,
                rank=None,  # Will be read from RANK env var
                distributed_init_method=distributed_init,
            )
        
        self.run_workers("init_worker",
                          get_all_outputs=True,
                          worker_init_fn=worker_init_fn)
        self._set_worker_ids()
        self.run_workers(
            "init_model",
            get_all_outputs=True,
        )
        self.run_workers(
            "load_model",
            get_all_outputs=True,
            max_concurrent_workers=self.parallel_config.
            max_parallel_loading_workers,
            kv_buffer_size=self.cache_config.kv_buffer_size,
            max_kv_slots=self.cache_config.max_kv_slots,
        )
    
    def _set_worker_ids(self) -> None:
        all_outputs = []
        for i, worker in enumerate(self.workers):
            if self.parallel_config.worker_use_ray:
                executor = partial(worker.execute_method.remote, 'set_worker_id')
                output = executor(i)
                all_outputs.append(output)
            else:
                executor = getattr(worker, 'set_worker_id')
                output = executor(i)
                all_outputs.append(output)
        if self.parallel_config.worker_use_ray:
            all_outputs = ray.get(all_outputs)
    
    def _run_workers_in_batch(
        self,
        workers,
        method: str,
        *args,
        **kwargs,
    ):
        all_outputs = []
        for worker in workers:
            if self.parallel_config.worker_use_ray:
                executor = partial(worker.execute_method.remote, method)
            else:
                executor = getattr(worker, method)

            output = executor(*args, **kwargs)
            all_outputs.append(output)
        if self.parallel_config.worker_use_ray:
            all_outputs = ray.get(all_outputs)
        return all_outputs
    
    def run_workers(
        self,
        method: str,
        *args,
        get_all_outputs: bool = False,
        max_concurrent_workers: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""
        all_outputs = []
        if max_concurrent_workers:
            work_groups = [
                self.workers[i:i + max_concurrent_workers]
                for i in range(0, len(self.workers), max_concurrent_workers)
            ]
        else:
            work_groups = [self.workers]

        for workers in work_groups:
            all_outputs.extend(
                self._run_workers_in_batch(workers, method, *args, **kwargs))

        if get_all_outputs:
            return all_outputs

        # Make sure all workers have the same results.
        output = all_outputs[0]
        for other_output in all_outputs[1:]:
            assert output == other_output
        return output
    
    def run_workers_bool_all(
        self,
        method: str,
        *args,
        max_concurrent_workers: Optional[int] = None,
        **kwargs,
    ) -> bool:
        all_outputs = self.run_workers(
            method,
            *args,
            get_all_outputs=True,
            max_concurrent_workers=max_concurrent_workers,
            **kwargs,
        )
        assert all(isinstance(x, bool) for x in all_outputs)
        return all(all_outputs)

    def run_workers_bool_any(
        self,
        method: str,
        *args,
        max_concurrent_workers: Optional[int] = None,
        **kwargs,
    ) -> bool:
        all_outputs = self.run_workers(
            method,
            *args,
            get_all_outputs=True,
            max_concurrent_workers=max_concurrent_workers,
            **kwargs,
        )
        assert all(isinstance(x, bool) for x in all_outputs)
        return any(all_outputs)