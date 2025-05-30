import pprint
from functools import partial

from google.cloud import storage

from tqdm import tqdm, trange
import numpy as np
import mlxu
import subprocess as sp
import neural_tangents as nt

import timeit
import os

import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
from flax.training.train_state import TrainState
from transformers import AutoTokenizer

import optax

from EasyLM.data import DatasetFactory
from EasyLM.checkpoint import StreamingCheckpointer
from EasyLM.optimizers import OptimizerFactory
from EasyLM.jax_utils import (
    JaxRNG, JaxDistributedConfig, next_rng, match_partition_rules,
    cross_entropy_loss_and_accuracy, global_norm, get_float_dtype_by_name,
    set_random_seed, average_metrics, make_shard_and_gather_fns,
    with_sharding_constraint, cross_entropy_loss_and_accuracy_with_weight_decay
)
from EasyLM.models.llama.llama_model import (
    LLaMAConfigurator, FlaxLLaMAForCausalLMModule
)


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=42,
    mesh_dim='1,-1,1',
    dtype='fp32',
    param_dtype='fp32',
    total_steps=10000,
    load_llama_config='',
    update_llama_config='',
    load_checkpoint='',
    load_dataset_state='',
    log_freq=50,
    save_model_freq=0,
    save_milestone_freq=0,
    eval_steps=0,
    eval_freq=0,
    weight_average=False,
    weight_average_decay=0.99,
    outer_loop_method='None',
    tokenizer='openlm-research/open_llama_3b_v2',
    train_dataset_batch_size=8,
    train_dataset=DatasetFactory.get_default_config(),
    eval_dataset=DatasetFactory.get_default_config(),
    optimizer=OptimizerFactory.get_default_config(),
    checkpointer=StreamingCheckpointer.get_default_config(),
    llama=LLaMAConfigurator.get_default_config(),
    logger=mlxu.WandBLogger.get_default_config(),
    log_all_worker=False,
    jax_distributed=JaxDistributedConfig.get_default_config(),
    start_step=0,
    experiment_id=0,
    gc_bucket=''
)

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

def is_embedding_param(param_name, param_value):
    if 'embedding' in param_name:
        return True
    return False

def count_params(params):
    non_embedding_count = 0
    total_count = 0

    for param_name, param_value in jax.tree_util.tree_leaves_with_path(params):
        # print(param_name[-1].key, is_embedding_param(param_name[-1].key, param_value), jnp.prod(jnp.array(param_value.size)))
        total_count += jnp.prod(jnp.array(param_value.size))
        if not is_embedding_param(param_name[-1].key, param_value):
            non_embedding_count += jnp.prod(jnp.array(param_value.size))
            print(param_name[-5:], is_embedding_param(param_name[-1].key, param_value), jnp.prod(jnp.array(param_value.size)))
        else:
            print(param_name, is_embedding_param(param_name[-1].key, param_value), jnp.prod(jnp.array(param_value.size)))
    # print(non_embedding_count)
    return total_count, non_embedding_count

def load_from_gcp(bucket_name, gc_path, local_path):
    """
    Downloads a file or all files in a directory from a GCP bucket to a local path.

    Args:
        bucket_name (str): Name of the GCP bucket.
        gc_path (str): Path to a file or directory in the GCP bucket.
        local_path (str): Path to the local file or directory where data will be saved.
    """
    if not bucket_name:
        raise ValueError("GCP bucket not specified.")

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Check if the given GCP path is a file or directory
    blobs = list(bucket.list_blobs(prefix=gc_path))

    if not blobs:
        raise ValueError(f"No files found at {gc_path} in bucket {bucket_name}")

    if len(blobs) == 1 and blobs[0].name == gc_path:  # Single file case
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blobs[0].download_to_filename(local_path)
        print(f"Downloaded {gc_path} to {local_path}")
    else:  # Directory case
        if not local_path.endswith('/'):
            local_path += '/'  # Ensure local directory structure
        os.makedirs(local_path, exist_ok=True)

        for blob in blobs:
            if not blob.name.endswith('/'):  # Ignore "directory" markers
                relative_path = blob.name[len(gc_path):].lstrip('/')  # Remove the prefix
                local_file_path = os.path.join(local_path, relative_path)
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                blob.download_to_filename(local_file_path)
                print(f"Downloaded {blob.name} to {local_file_path}")

    print("Download complete.")

def load_ckpt_from_gcp(bucket, checkpoint_path, local_path='/tmp/model.ckpt'):
    if bucket == '':
        raise ValueError("GCP bucket not specified.")
    ckpt_type, ckpt_path = checkpoint_path.split('::')
    local_path = load_from_gcp(bucket, ckpt_path, local_path)
    print(f"Checkpoint downloaded to {local_path}")
    return f'{ckpt_type}::/{local_path}'

def main(argv):
    JaxDistributedConfig.initialize(FLAGS.jax_distributed)
    variant = mlxu.get_user_flags(FLAGS, FLAGS_DEF)
    flags_config_dict = mlxu.user_flags_to_config_dict(FLAGS, FLAGS_DEF)
    logger = mlxu.WandBLogger(
        config=FLAGS.logger,
        variant=variant,
        enable=FLAGS.log_all_worker or (jax.process_index() == 0),
    )
    set_random_seed(FLAGS.seed)

    print(FLAGS.train_dataset)

    if FLAGS.gc_bucket != '':
        FLAGS.load_checkpoint = load_ckpt_from_gcp(FLAGS.gc_bucket, FLAGS.load_checkpoint)
        if FLAGS.eval_dataset.huggingface_dataset.pretokenized_dataset_dir != '':
            FLAGS.eval_dataset.huggingface_dataset.pretokenized_dataset_dir = load_from_gcp(FLAGS.gc_bucket, FLAGS.eval_dataset.huggingface_dataset.pretokenized_dataset_dir, '/tmp/eval_dataset')
        if FLAGS.train_dataset.huggingface_dataset.pretokenized_dataset_dir != '':
            FLAGS.train_dataset.huggingface_dataset.pretokenized_dataset_dir = load_from_gcp(FLAGS.gc_bucket, FLAGS.train_dataset.huggingface_dataset.pretokenized_dataset_dir, '/tmp/train_dataset')


    tokenizer = AutoTokenizer.from_pretrained(FLAGS.tokenizer)
    dataset = DatasetFactory.load_dataset(FLAGS.train_dataset, tokenizer)
    if FLAGS.load_dataset_state != '':
        dataset.load_state_dict(mlxu.load_pickle(FLAGS.load_dataset_state))

    if FLAGS.eval_steps > 0:
        eval_dataset = DatasetFactory.load_dataset(
            FLAGS.eval_dataset, dataset.tokenizer
        )
        eval_iterator = iter(eval_dataset)

    seq_length = dataset.seq_length
    llama_config = LLaMAConfigurator.finalize_config(FLAGS.llama)

    model = FlaxLLaMAForCausalLMModule(
        llama_config,
        dtype=get_float_dtype_by_name(FLAGS.dtype),
        param_dtype=get_float_dtype_by_name(FLAGS.param_dtype),
    )

    optimizer, optimizer_info = OptimizerFactory.get_optimizer(FLAGS.optimizer)

    # print(hasattr(FLAGS.optimizer, 'soap_optimizer'))
    # if hasattr(FLAGS.optimizer, 'soap_optimizer') and FLAGS.optimizer.soap_optimizer.lr_decay_steps != FLAGS.total_steps // (FLAGS.train_dataset_batch_size * FLAGS.optimizer.accumulate_gradient_steps):
    #     raise ValueError("Remember to set decay_steps to total_steps // (batch_size * accumulate_gradient_steps)")
    # elif hasattr(FLAGS.optimizer, 'adamw_optimizer') and FLAGS.optimizer.adamw_optimizer.lr_decay_steps != FLAGS.total_steps // (FLAGS.train_dataset_batch_size * FLAGS.optimizer.accumulate_gradient_steps):
    #     raise ValueError("Remember to set decay_steps to total_steps // (batch_size * accumulate_gradient_steps)")
    
    def create_trainstate_from_params(params):
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def init_fn(rng):
        rng_generator = JaxRNG(rng)
        params = model.init(
            input_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            position_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            attention_mask=jnp.ones((4, seq_length), dtype=jnp.int32),
            rngs=rng_generator(LLaMAConfigurator.rng_keys()),
        )
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def train_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)

        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))

        def loss_and_accuracy(params):
            logits = model.apply(
                params, batch['input_tokens'], deterministic=False,
                rngs=rng_generator(LLaMAConfigurator.rng_keys()),
            ).logits
            return cross_entropy_loss_and_accuracy(
                logits, batch['target_tokens'], batch['loss_masks']
            )
        grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
        (loss, accuracy), grads = grad_fn(train_state.params)
        try:
            perplexity = jnp.exp(loss)
        except OverflowError:
            perplexity = jnp.float32("inf")
        train_state = train_state.apply_gradients(grads=grads)
        metrics = dict(
            loss=loss,
            perplexity=perplexity,
            accuracy=accuracy,
            learning_rate=optimizer_info['learning_rate_schedule'](train_state.step),
            gradient_norm=global_norm(grads),
            param_norm=global_norm(train_state.params),
            gpu_memory=get_gpu_memory()[0],
        )
        return train_state, rng_generator(), metrics

    # def train_step(train_state, rng, batch):
    #     """
    #     Perform a training step using microbatches.

    #     Args:
    #         train_state: Current state of the model, including parameters and optimizer state.
    #         rng: Random number generator state.
    #         batch: A dictionary containing input tokens, target tokens, and loss masks.
    #         num_micro_batches: Number of microbatches to split the batch into.

    #     Returns:
    #         Updated train_state, new RNG state, and training metrics.
    #     """
    #     num_micro_batches = 4
    #     rng_generator = JaxRNG(rng)

    #     # Split the batch into microbatches
    #     microbatches = {
    #         key: jnp.split(value, num_micro_batches, axis=0)
    #         for key, value in batch.items()
    #     }
        
    #     def loss_and_accuracy(params, microbatch, rngs):
    #         logits = model.apply(
    #             params, 
    #             microbatch['input_tokens'], 
    #             deterministic=False,
    #             rngs=rngs,
    #         ).logits
    #         return cross_entropy_loss_and_accuracy(
    #             logits, microbatch['target_tokens'], microbatch['loss_masks']
    #         )
        
    #     def microbatch_step(carry, microbatch_index):
    #         rngs = rng_generator(LLaMAConfigurator.rng_keys())
    #         cumulative_grads, total_loss, total_accuracy = carry
            
    #         microbatch = {
    #             key: jax.lax.dynamic_index_in_dim(jax.numpy.array(value), microbatch_index, keepdims=False)
    #             for key, value in microbatches.items()
    #         }

    #         grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
    #         (loss, accuracy), grads = grad_fn(train_state.params, microbatch, rngs)

    #         # Initialize cumulative_grads with the correct structure
    #         cumulative_grads = jax.tree_map(lambda x: jnp.zeros_like(x), train_state.params)

    #         # Update cumulative_grads in the loop
    #         cumulative_grads = jax.tree_util.tree_map(
    #             jnp.add, cumulative_grads, grads
    #         )
    #         return (
    #             cumulative_grads,
    #             total_loss + loss,
    #             total_accuracy + accuracy,
    #         ), None

    #     # Accumulate gradients and metrics over all microbatches
    #     init_carry = ({}, 0.0, 0.0)  # Use an empty dict instead of None

    #     (cumulative_grads, total_loss, total_accuracy), _ = jax.lax.scan(
    #         microbatch_step,
    #         init=init_carry,
    #         xs=jnp.arange(num_micro_batches)
    #     )
        
    #     # Average the accumulated gradients and metrics
    #     cumulative_grads = jax.tree_util.tree_map(
    #         lambda x: x / num_micro_batches, cumulative_grads
    #     )
    #     avg_loss = total_loss / num_micro_batches
    #     avg_accuracy = total_accuracy / num_micro_batches

    #     # Apply gradients to update the model
    #     train_state = train_state.apply_gradients(grads=cumulative_grads)
    #     try:
    #         perplexity = jnp.exp(avg_loss)
    #     except OverflowError:
    #         perplexity = jnp.float32("inf")
        
    #     metrics = dict(
    #         loss=avg_loss,
    #         perplexity=perplexity,
    #         accuracy=avg_accuracy,
    #         learning_rate=optimizer_info['learning_rate_schedule'](train_state.step),
    #         gradient_norm=global_norm(cumulative_grads),
    #         param_norm=global_norm(train_state.params),
    #         gpu_memory=get_gpu_memory()[0],
    #     )
    #     return train_state, rng_generator(), metrics



    
    def train_step_tayl(train_state, rng, batch, **kwargs):
        rng_generator = JaxRNG(rng)
        f_tayl = nt.taylor_expand(model.apply, train_state.params, 2)
        # f_tayl = jax.block_until_ready(f_tayl)
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))

        # lr = kwargs['learning_rate']
        # wd = kwargs['weight_decay']
        # step = kwargs['step']
        lr = 0.00001
        wd = 0.5

        # @partial(jax.jit, static_argnums=0)
        def apply_taylor_fn(f_tayl, batch, rng, new_params, old_params, weight_decay):
            rng_generator = JaxRNG(rng)
            def loss_and_accuracy(params, old_params):
                logits = f_tayl(
                    params, batch['input_tokens'], deterministic=False,
                    rngs=rng_generator(LLaMAConfigurator.rng_keys()),
                ).logits

                return cross_entropy_loss_and_accuracy_with_weight_decay(logits, batch['target_tokens'], params, old_params, batch['loss_masks'], weight_decay=weight_decay)

            grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
            (loss, accuracy), grads = grad_fn(new_params, old_params)
            return grads, loss, accuracy

        new_params = train_state.params
        tayl_solver = optax.adam(learning_rate=lr) # TODO: check this?
        opt_state = tayl_solver.init(new_params)
        apply_fn_jit = jax.jit(apply_taylor_fn, static_argnums=0)

        for i in range(100):
            grads, loss, accuracy = apply_fn_jit(f_tayl, batch, rng, new_params, train_state.params, wd) # TODO: fix weight decay param
            updates, opt_state = tayl_solver.update(grads, opt_state, new_params)
            new_params = optax.apply_updates(new_params, updates)

        train_state = train_state.replace(
            step=train_state.step+1,
            params=new_params
        ) 

        metrics = dict(
            loss=loss,
            accuracy=accuracy,
            learning_rate=optimizer_info['learning_rate_schedule'](train_state.step), # doesn't mean anything for taylor
            gradient_norm=global_norm(grads), # not sure if this is accurate?
            param_norm=global_norm(train_state.params),
            gpu_memory=get_gpu_memory()[0],
        )
        # jax.clear_caches()
        # del apply_taylor_fn, apply_fn_jit
        return train_state, rng_generator(), metrics
    
    def train_step_tayl_scan(train_state, rng, batch, **kwargs):
        rng_generator = JaxRNG(rng)

        f_tayl = nt.taylor_expand(model.apply, train_state.params, 2)
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))

        # lr = kwargs['learning_rate']
        # wd = kwargs['weight_decay']
        # step = kwargs['step']
        lr = 0.00001
        wd = 0.5

        # @partial(jax.jit, static_argnums=0)
        def apply_taylor_fn(f_tayl, batch, rng, new_params, old_params, weight_decay):
            rng_generator = JaxRNG(rng)
            def loss_and_accuracy(params, old_params):
                logits = f_tayl(
                    params, batch['input_tokens'], deterministic=False,
                    rngs=rng_generator(LLaMAConfigurator.rng_keys()),
                ).logits

                return cross_entropy_loss_and_accuracy_with_weight_decay(logits, batch['target_tokens'], params, old_params, batch['loss_masks'], weight_decay=weight_decay)

            grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
            (loss, accuracy), grads = grad_fn(new_params, old_params)
            return grads, loss, accuracy

        new_params = train_state.params
        tayl_solver = optax.adam(learning_rate=lr)
        opt_state = tayl_solver.init(new_params)
        apply_fn_jit = jax.jit(apply_taylor_fn, static_argnums=0)

        def step_fn(carry, _):
            new_params, opt_state = carry
            grads, loss, accuracy = apply_fn_jit(f_tayl, batch, rng, new_params, train_state.params, wd)
            updates, opt_state = tayl_solver.update(grads, opt_state, new_params)
            new_params = optax.apply_updates(new_params, updates)
            return (new_params, opt_state), (loss, accuracy, grads)

        # def step_fn(carry, _):
        #     new_params, opt_state = carry
            
        #     # Use jax.checkpoint to reduce memory usage by recomputing intermediate values
        #     def compute_grads(params):
        #         grads, loss, accuracy = apply_taylor_fn(f_tayl, batch, rng, params, train_state.params, wd)
        #         return grads, loss, accuracy

        #     grads, loss, accuracy = jax.checkpoint(compute_grads)(new_params)
        #     updates, opt_state = tayl_solver.update(grads, opt_state, new_params)
        #     new_params = optax.apply_updates(new_params, updates)
            
        #     return (new_params, opt_state), (loss, accuracy, grads)

        init_carry = (new_params, opt_state)
        init_steps = None

        # (carry, outputs) = jax.lax.scan(step_fn, init_carry, init_steps, length=10)
        (carry, outputs) = jax.lax.scan(step_fn, init_carry, init_steps, length=12)

        init_carry = carry
        init_steps = None

        new_params, opt_state = carry
        f_tayl = nt.taylor_expand(model.apply, new_params, 2)
        (carry, outputs) = jax.lax.scan(step_fn, init_carry, init_steps, length=50)
        new_params, opt_state = carry
        loss, accuracy, grads = outputs


        train_state = train_state.replace(
            step=train_state.step + 1,
            params=new_params
        )

        metrics = dict(
            loss=loss[-1],
            accuracy=accuracy[-1],
            learning_rate=optimizer_info['learning_rate_schedule'](train_state.step), # doesn't mean anything for taylor
            gradient_norm=global_norm(grads), # not sure if this is accurate?
            param_norm=global_norm(train_state.params),
            gpu_memory=get_gpu_memory()[0],
        )
        # jax.clear_caches()
        # del apply_taylor_fn, apply_fn_jit
        return train_state, rng_generator(), metrics


    # def eval_step(train_state, rng, batch):
    #     rng_generator = JaxRNG(rng)
    #     batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
    #     logits = model.apply(
    #         train_state.params, batch['input_tokens'], deterministic=True,
    #         rngs=rng_generator(LLaMAConfigurator.rng_keys()),
    #     ).logits
    #     loss, accuracy = cross_entropy_loss_and_accuracy(
    #         logits, batch['target_tokens'], batch['loss_masks']
    #     )
    #     try:
    #         perplexity = jnp.exp(loss)
    #     except OverflowError:
    #         perplexity = jnp.float32("inf")
    #     metrics = dict(
    #         eval_loss=loss,
    #         eval_accuracy=accuracy,
    #         eval_perplexity=perplexity,
    #     )
    #     return rng_generator(), metrics

    def eval_step(params, rng, batch):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
        logits = model.apply(
            params, batch['input_tokens'], deterministic=True,
            rngs=rng_generator(LLaMAConfigurator.rng_keys()),
        ).logits
        loss, accuracy = cross_entropy_loss_and_accuracy(
            logits, batch['target_tokens'], batch['loss_masks']
        )
        try:
            perplexity = jnp.exp(loss)
        except OverflowError:
            perplexity = jnp.float32("inf")
        metrics = dict(
            eval_loss=loss,
            eval_accuracy=accuracy,
            eval_perplexity=perplexity,
        )
        return rng_generator(), metrics

    train_state_shapes = jax.eval_shape(init_fn, next_rng())
    train_state_partition = match_partition_rules(
        LLaMAConfigurator.get_partition_rules(), train_state_shapes
    )

    batch_partition = {
        'input_tokens': PS(('dp', 'fsdp')), 
        'loss_masks': PS(('dp', 'fsdp')),
        'target_tokens': PS(('dp', 'fsdp')),
    }

    shard_fns, gather_fns = make_shard_and_gather_fns(
        train_state_partition, train_state_shapes
    )
    checkpointer = StreamingCheckpointer(
        FLAGS.checkpointer, logger.output_dir,
        enable=jax.process_index() == 0,
    )

    sharded_init_fn = pjit(
        init_fn,
        in_shardings=PS(),
        out_shardings=train_state_partition
    )

    sharded_create_trainstate_from_params = pjit(
        create_trainstate_from_params,
        in_shardings=(train_state_partition.params, ),
        out_shardings=train_state_partition,
        donate_argnums=(0, ),
    )

    sharded_train_step = pjit(
        train_step,
        in_shardings=(train_state_partition, PS(), batch_partition),
        out_shardings=(train_state_partition, PS(), PS()),
        donate_argnums=(0, 1),
    )
    # sharded_train_step = train_step_tayl_scan

    sharded_eval_step = pjit(
        eval_step,
        in_shardings=(train_state_partition.params, PS(), PS()),
        out_shardings=(PS(), PS()),
        donate_argnums=(1,),
    )

    def save_checkpoint(train_state, milestone=False):
        step = int(jax.device_get(train_state.step))
        metadata = dict(
            step=step,
            variant=variant,
            flags=flags_config_dict,
            llama_config=llama_config.to_dict(),
        )
        checkpointer.save_all(
            train_state=train_state,
            gather_fns=gather_fns,
            metadata=metadata,
            dataset=dataset.get_state_dict(),
            milestone=milestone,
        )

    def shard_batch(batch, num_devices):
        # Shard each tensor along the first axis
        sharded = {k: np.array_split(v, num_devices) for k, v in batch.items()}
        # Group the shards for each device into a list of dictionaries
        return [{k: sharded[k][i] for k in batch} for i in range(num_devices)]

    # FLAGS.load_checkpoint = 'trainstate_params::/n/holyscratch01/barak_lab/Users/nabreu/SOO-LM/checkpoint/47578767/streaming_train_state_1000' #1000step warmup adam bs=32k lr=0.01
    
    # jax.profiler.start_trace("tensorboard-base")
    mesh = LLaMAConfigurator.get_jax_mesh(FLAGS.mesh_dim)
    print(f"Mesh axes names: {mesh.axis_names}")
    print(f"Mesh shape: {mesh.shape}")

    with mesh:
        print(mesh)
        train_state, restored_params = None, None
        if FLAGS.load_checkpoint != '':
            train_state, restored_params = checkpointer.load_trainstate_checkpoint(
                FLAGS.load_checkpoint, train_state_shapes, shard_fns
            )
            print('loaded checkpoint')

        if train_state is None and restored_params is None:
            # Initialize from scratch
            train_state = sharded_init_fn(next_rng())
        elif train_state is None and restored_params is not None:
            # Restore from params but initialize train_state
            train_state = sharded_create_trainstate_from_params(restored_params)
            del restored_params

        # param_count = sum(x.size for x in jax.tree_leaves(train_state.params))
        param_count, param_count_nonembed = count_params(train_state.params)
        param_count = jax.device_get(param_count)
        param_count_nonembed = jax.device_get(param_count_nonembed)
        logger.log({"param_count_nonembed": param_count_nonembed})
        logger.log({"param_count": param_count})

        start_step = int(jax.device_get(train_state.step))

        if FLAGS.save_model_freq > 0:
            save_checkpoint(train_state)

        sharded_rng = next_rng()

        step_counter = trange(start_step, FLAGS.total_steps, ncols=0)

        assert FLAGS.train_dataset_batch_size % mesh.shape['dp'] == 0, \
            "Batch size must be divisible by the number of devices in 'dp'."
        
        if FLAGS.weight_average:
            print('Using weight average')
            ema = train_state.params

        for step, (batch, dataset_metrics) in zip(step_counter, dataset):

            batch = jax.tree_map(
                lambda x: jax.lax.with_sharding_constraint(x, PS(('dp', 'fsdp'))),
                batch
            )

            train_state, sharded_rng, metrics = sharded_train_step(
                train_state, sharded_rng, batch
            )

            if FLAGS.weight_average:
                alpha = FLAGS.weight_average_decay
                ema = jax.tree_util.tree_map(lambda x, y: alpha*x + (1-alpha)*y, ema, train_state.params)

            

            if step % FLAGS.log_freq == 0:
                log_metrics = {"step": step}
                log_metrics.update(metrics)
                log_metrics.update(dataset_metrics)
                

                if FLAGS.eval_freq != 0 and FLAGS.eval_steps > 0: # eval_freq must be | by log_freq
                    if step % FLAGS.eval_freq == 0:
                        eval_iterator = iter(eval_dataset)
                        eval_metric_list = []
                        for _ in range(FLAGS.eval_steps):
                            eval_batch, _ = next(eval_iterator)

                            if FLAGS.weight_average:
                                eval_params=ema
                            else:
                                eval_params = train_state.params
                            sharded_rng, eval_metrics = sharded_eval_step(
                                eval_params, sharded_rng, eval_batch
                            )
                            eval_metric_list.append(eval_metrics)
                        log_metrics.update(average_metrics(eval_metric_list))
                        # metrics.update({"step": step})
                        # metrics = jax.device_get(metrics)
                        # logger.log(metrics)
                log_metrics = jax.device_get(log_metrics)
                logger.log(log_metrics)
                tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")
            
            

            if FLAGS.save_milestone_freq > 0 and (step + 1) % FLAGS.save_milestone_freq == 0:
                save_checkpoint(train_state, milestone=True)
            elif FLAGS.save_model_freq > 0 and (step + 1) % FLAGS.save_model_freq == 0:
                save_checkpoint(train_state)

        if FLAGS.eval_freq != 0 and FLAGS.eval_steps > 0: # eval_freq must be | by log_freq
            eval_iterator = iter(eval_dataset)
            eval_metric_list = []
            for _ in range(FLAGS.eval_steps):
                eval_batch, _ = next(eval_iterator)

                if FLAGS.weight_average:
                    eval_params=ema
                else:
                    eval_params = train_state.params
                sharded_rng, eval_metrics = sharded_eval_step(
                    eval_params, sharded_rng, eval_batch
                )
                eval_metric_list.append(eval_metrics)
            log_metrics.update(average_metrics(eval_metric_list))
            log_metrics = jax.device_get(log_metrics)
            logger.log(log_metrics)
            tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")
        if FLAGS.save_model_freq > 0:
            save_checkpoint(train_state)

    # jax.profiler.stop_trace()


if __name__ == "__main__":
    print(jax.local_devices())
    print(jax.devices())
    mlxu.run(main)