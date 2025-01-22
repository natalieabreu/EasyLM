import pprint
from functools import partial

import tracemalloc
from tqdm import tqdm
import numpy as np
import mlxu
import subprocess as sp
import os
import neural_tangents as nt

import psutil
import wandb

import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.experimental import checkify

from jax.sharding import PartitionSpec as PS
from flax.training.train_state import TrainState
from transformers import AutoTokenizer

import optax

import jax.profiler

from EasyLM.data import DatasetFactory
from EasyLM.checkpoint import StreamingCheckpointer
from EasyLM.optimizers import OptimizerFactory
from EasyLM.jax_utils import (
    JaxRNG, JaxDistributedConfig, next_rng, match_partition_rules,
    cross_entropy_loss_and_accuracy, global_norm, get_float_dtype_by_name,
    set_random_seed, average_metrics, make_shard_and_gather_fns,
    with_sharding_constraint, cross_entropy_loss_and_accuracy_with_weight_decay,
    get_gpu_memory, get_ram, count_params, print_shape, display_top
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
    eval_freq=0,
    eval_steps=0,
    gradient_accumulation_steps=1,
    inner_loop_iter=100,
    tokenizer='openlm-research/open_llama_3b_v2',
    train_dataset_batch_size=8,
    train_dataset=DatasetFactory.get_default_config(),
    eval_dataset=DatasetFactory.get_default_config(),
    # optimizer=OptimizerFactory.get_default_config(),
    checkpointer=StreamingCheckpointer.get_default_config(),
    llama=LLaMAConfigurator.get_default_config(),
    # logger=mlxu.WandBLogger.get_default_config(),
    log_all_worker=False,
    jax_distributed=JaxDistributedConfig.get_default_config(),
    taylor_order=2,
    outer_loop_method='replace',
    lr_sched='cosine',
    inner_loop_lr=0.001,
    inner_loop_wd=0.0,
    end_lr=0.0,
    global_warmup=0.2,
    inner_loop_warmup=0.0,

    inner_b1=0.9,
    inner_b2=0.999,
    inner_clip_gradient=0.0,
    optimizer_wd=0.0,

    wandb_run_id='',
    start_tokens=0,

    wandb_project='',
    wandb_dir='/n/netscratch/kempner_barak_lab/Lab/nabreu/SOO-LM/experiment_output/open_llama_7b',
    output_dir='',
    notes='',
    experiment_id='',

    weight_average=False,
    weight_average_decay=0.99,
    load_ema_checkpoint='',
)

def main(argv):
    print('Init',  get_gpu_memory())
    output_dir = os.path.join(FLAGS.output_dir, FLAGS.experiment_id)

    JaxDistributedConfig.initialize(FLAGS.jax_distributed)
    variant = mlxu.get_user_flags(FLAGS, FLAGS_DEF)
    flags_config_dict = mlxu.user_flags_to_config_dict(FLAGS, FLAGS_DEF)
    log_config = mlxu.flatten_config_dict(flags_config_dict)
    # logger = mlxu.WandBLogger(
    #     config=FLAGS.logger,
    #     variant=variant,
    #     enable=FLAGS.log_all_worker or (jax.process_index() == 0),
    # )
    set_random_seed(FLAGS.seed)

    assert FLAGS.gradient_accumulation_steps == 1, 'Gradient accumulation not supported'

    dataset = DatasetFactory.load_dataset(FLAGS.train_dataset, None)
    if FLAGS.load_dataset_state != '':
        dataset.load_state_dict(mlxu.load_pickle(FLAGS.load_dataset_state))

    if FLAGS.eval_steps > 0:
        eval_dataset = DatasetFactory.load_dataset(
            FLAGS.eval_dataset, dataset.tokenizer
        )

    seq_length = dataset.seq_length
    llama_config = LLaMAConfigurator.finalize_config(FLAGS.llama)

    print('Dataset', get_gpu_memory())

    model = FlaxLLaMAForCausalLMModule(
        llama_config,
        dtype=get_float_dtype_by_name(FLAGS.dtype),
        param_dtype=get_float_dtype_by_name(FLAGS.param_dtype),
    )

    def get_global_lr_sched(method, lr, taylor_steps, inner_loop_iter, warmup, inner_warmup, end_lr):
        if method == 'global_cosine':
            decay_steps = taylor_steps*inner_loop_iter
            decay_steps = int(decay_steps)
            if warmup <= 1.0:
                warmup = int(warmup*decay_steps)

            if isinstance(warmup, tuple):
                warmup = int(warmup[0])
            
            schedule = optax.warmup_cosine_decay_schedule(
                init_value=lr*0.1,
                peak_value=lr,
                warmup_steps=warmup,
                decay_steps=decay_steps,
                end_value=end_lr,
            )
        elif method == 'cosine_with_global_schedule':
            decay_steps = taylor_steps
            decay_steps = int(decay_steps)
            if warmup <= 1.0:
                warmup = int(warmup*decay_steps)
            if isinstance(warmup, tuple):
                warmup = int(warmup[0])

            if inner_warmup <= 1.0:
                inner_warmup = int(inner_warmup*inner_loop_iter)
            if isinstance(inner_warmup, tuple):
                inner_warmup = int(inner_warmup[0])
            
            global_sched = optax.warmup_cosine_decay_schedule(
                init_value=lr*0.1,
                peak_value=lr,
                warmup_steps=warmup,
                decay_steps=decay_steps,
                end_value=end_lr,
            )
            schedules = []
            boundaries = []
            for step in range(taylor_steps):
                peak_lr = global_sched(step)
                inner_sched = optax.warmup_cosine_decay_schedule(
                    init_value=peak_lr*0.1,
                    peak_value=peak_lr,
                    warmup_steps=inner_warmup,
                    decay_steps=inner_loop_iter,
                    end_value=end_lr,
                )
                schedules.append(inner_sched)
                boundaries.append(step*inner_loop_iter)

            schedule = optax.join_schedules(schedules, boundaries)

        elif method == 'constant':
            schedule = optax.constant_schedule(lr)
        else:
            raise ValueError(f"Unknown global schedule method: {method}")

        return schedule

    def build_optimizer(lr_sched, b1, b2, grad_clip=None, wd=0.0):
        if grad_clip:
            optimizer = optax.chain(
                optax.clip_by_global_norm(grad_clip),
                optax.adamw(
                    learning_rate=lr_sched,
                    b1=b1,
                    b2=b2,
                    mu_dtype=jnp.float32,
                    weight_decay=wd
                )
            )
        else:
            optimizer = optax.adamw(
                learning_rate=lr_sched,
                b1=b1,
                b2=b2,
                mu_dtype=jnp.float32,
                weight_decay=wd
            )
        return optimizer

    # optimizer, optimizer_info = OptimizerFactory.get_optimizer(FLAGS.optimizer)
    lr_sched = get_global_lr_sched(FLAGS.lr_sched, FLAGS.inner_loop_lr, FLAGS.total_steps, FLAGS.inner_loop_iter, FLAGS.global_warmup, FLAGS.inner_loop_warmup, FLAGS.end_lr)
    tayl_solver = build_optimizer(lr_sched, FLAGS.inner_b1, FLAGS.inner_b2, FLAGS.inner_clip_gradient, FLAGS.optimizer_wd)

    def create_trainstate_from_params(params):
        return TrainState.create(params=params, tx=tayl_solver, apply_fn=None)

    def init_fn(rng):
        rng_generator = JaxRNG(rng)
        params = model.init(
            input_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            position_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            attention_mask=jnp.ones((4, seq_length), dtype=jnp.int32),
            rngs=rng_generator(LLaMAConfigurator.rng_keys()),
        )
        return TrainState.create(params=params, tx=tayl_solver, apply_fn=None)

    def eval_step(params, batch, rng):
        # rng: per-device RNG key of shape (2,)
        # batch: per-device batch data
        # params: replicated parameters (same on all devices)

        rng_generator = JaxRNG(rng)

        # Apply the model in evaluation mode (deterministic=True)
        logits = model.apply(
            params, batch['input_tokens'], deterministic=True,
            rngs=rng_generator(LLaMAConfigurator.rng_keys()),
        ).logits

        # Compute loss and accuracy
        loss, accuracy = cross_entropy_loss_and_accuracy(
            logits, batch['target_tokens'], batch['loss_masks']
        )
        # Compute perplexity
        perplexity = jnp.exp(loss)

        metrics = {
            'eval_loss': loss,
            'eval_accuracy': accuracy,
            'eval_perplexity': perplexity,
        }
        return metrics

    
    train_state_shapes = jax.eval_shape(init_fn, next_rng())

    # print(train_state_shapes)
    train_state_partition = match_partition_rules(
        LLaMAConfigurator.get_partition_rules(), train_state_shapes
    )

    shard_fns, gather_fns = make_shard_and_gather_fns(
        train_state_partition, train_state_shapes
    )
    checkpointer = StreamingCheckpointer(
        FLAGS.checkpointer, output_dir,
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

    # sharded_eval_step = jax.jit(
    #     eval_step,
    #     in_shardings=(train_state_partition.params, PS(), PS()),
    #     out_shardings=(PS(), PS()),
    #     donate_argnums=(1,),
    # )
    sharded_eval_step = jax.jit(eval_step)

    # p_eval_step = jax.pmap(eval_step, axis_name='batch')
    # p_eval_step = jax.jit(eval_step) # try this?

    def shard_batch(batch, num_devices):
        # Ensure batch size is divisible by num_devices
        batch_size = batch['input_tokens'].shape[0]
        per_device_batch_size = batch_size // num_devices
        batch_size = per_device_batch_size * num_devices  # Trim to divisible size

        sharded_batch = {}
        for key in batch:
            data = batch[key][:batch_size]  # Trim the data
            # Reshape to (num_devices, per_device_batch_size, ...)
            sharded_batch[key] = data.reshape(num_devices, per_device_batch_size, *data.shape[1:])
        return sharded_batch
    
    def get_sharded_rng(rng_key):
        num_devices = jax.local_device_count()
        rngs = jax.random.split(rng_key, num=num_devices)
        return rngs  # Shape: (num_devices, 2)


    def run_evaluation(params, eval_dataset, global_step, total_tokens):
        num_devices = jax.local_device_count()
        eval_losses = []
        eval_accuracies = []
        eval_perplexities = []

        eval_iterator = iter(eval_dataset)

        # Initialize a base RNG key
        rng_key = jax.random.PRNGKey(FLAGS.seed+1)

        for _ in range(FLAGS.eval_steps):
            eval_batch, _ = next(eval_iterator)
            # Shard the evaluation batch
            # sharded_eval_batch = shard_batch(eval_batch, num_devices)

            # Generate per-device RNGs
            rng_key, step_rng_key = jax.random.split(rng_key)
            # sharded_rng = get_sharded_rng(step_rng_key)

            # Replicate params
            # sharded_params = jax.device_put_replicated(params, jax.local_devices())

            # Run the sharded evaluation step
            metrics = p_eval_step(params, eval_batch, step_rng_key)

            # Collect metrics from devices
            eval_losses.append(jax.device_get(metrics['eval_loss']))
            eval_accuracies.append(jax.device_get(metrics['eval_accuracy']))
            eval_perplexities.append(jax.device_get(metrics['eval_perplexity']))

        # # Concatenate and compute mean metrics
        # eval_loss = np.mean(np.concatenate(eval_losses))
        # eval_accuracy = np.mean(np.concatenate(eval_accuracies))
        # eval_perplexity = np.mean(np.concatenate(eval_perplexities))

        eval_loss = np.mean(eval_losses)
        eval_accuracy = np.mean(eval_accuracies)
        eval_perplexity = np.mean(eval_perplexities)

        # Log metrics to wandb
        wandb.log({
            'eval_loss': eval_loss,
            'eval_accuracy': eval_accuracy,
            'eval_perplexity': eval_perplexity,
            'global_step': global_step,
            'dataset_total_tokens': total_tokens,
        })
        
    def save_checkpoint(train_state, ema=None, milestone=False):
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
            ema=ema,
            # dataset=dataset.get_state_dict(),
            milestone=milestone,
        )
    
    def loss_fn(params, batch, rng):
        rng_generator = JaxRNG(rng)

        logits = model.apply(
            params, batch['input_tokens'], deterministic=False,
            rngs=rng_generator(LLaMAConfigurator.rng_keys()),
        ).logits
        return cross_entropy_loss_and_accuracy(
            logits, batch['target_tokens'], batch['loss_masks']
        )
    
    def apply_taylor_fn(f_tayl, batch, rng, new_params, old_params, weight_decay):
            
            rng_generator = JaxRNG(rng)
            batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))

            def loss_and_accuracy_with_wd(params, old_params):
                logits = f_tayl(
                    params, batch['input_tokens'], deterministic=True,
                    rngs=rng_generator(LLaMAConfigurator.rng_keys()),
                ).logits

                return cross_entropy_loss_and_accuracy_with_weight_decay(logits, batch['target_tokens'], params, old_params, batch['loss_masks'], weight_decay=weight_decay)
            
            grad_fn = jax.value_and_grad(loss_and_accuracy_with_wd, has_aux=True)
            (wd_loss, aux), grads = grad_fn(new_params, old_params)
            accuracy, base_loss = aux
            return grads, base_loss, accuracy, wd_loss
    

    batch_size = FLAGS.train_dataset_batch_size
    degree = FLAGS.taylor_order
    wd = FLAGS.inner_loop_wd
    num_devices = jax.device_count()
    batch_size_per_device = batch_size // num_devices

    num_devices = jax.local_device_count()
    print(f"Number of devices: {num_devices}")

    mesh = LLaMAConfigurator.get_jax_mesh(FLAGS.mesh_dim)
    with mesh:
    
        start_step = 0
        ema = None
        opt_state = None

        train_state, restored_params = None, None
        if FLAGS.load_checkpoint != '':
            train_state, restored_params = checkpointer.load_trainstate_checkpoint(
                FLAGS.load_checkpoint, train_state_shapes, shard_fns
            )
            # distinguish between loading from train_state and loading from params
            if train_state is not None and output_dir in FLAGS.load_checkpoint: # need to distinguish between loading adam initial ckpt and taylor mid-run ckpt
                start_step = int(jax.device_get(train_state.step))
                start_tokens = int(jax.device_get(train_state.step)) * FLAGS.inner_loop_iter * batch_size * seq_length + FLAGS.train_dataset.huggingface_dataset.tokens_count_at_start
                dataset.set_start_tokens(start_tokens)
                print('loaded checkpoint, starting at step', start_step)
                print('\tstart tokens:', start_tokens)

                _, ema = checkpointer.load_trainstate_checkpoint(
                    FLAGS.load_ema_checkpoint, train_state_shapes, shard_fns
                )

            if train_state is not None: # do this in both cases
                opt_state = train_state.opt_state

        if train_state is None and restored_params is None:
            # Initialize from scratch
            train_state = sharded_init_fn(next_rng())
        elif train_state is None and restored_params is not None:
            # Restore from params but initialize train_state
            train_state = sharded_create_trainstate_from_params(restored_params)
            del restored_params

        param_count, param_count_nonembed = count_params(train_state.params)
        param_count = jax.device_get(param_count)
        param_count_nonembed = jax.device_get(param_count_nonembed)

        flags_config_dict['param_count'] = param_count
        flags_config_dict['param_count_nonembed'] = param_count_nonembed

        # Initialize wandb
        if FLAGS.wandb_run_id:
            wandb.init(entity='harvardml', project=FLAGS.wandb_project, resume="must", id=FLAGS.wandb_run_id, dir=FLAGS.wandb_dir)
        else:
            wandb.init(entity='harvardml', project=FLAGS.wandb_project, config=log_config, dir=FLAGS.wandb_dir)  # Replace with your project name

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(os.path.join(output_dir, 'wandb_id.txt'), 'w+') as f:
                f.write(wandb.run.id) # hacky but easier than handling in train state loader

        # start_step = int(jax.device_get(train_state.step)) # TODO: Start from checkpoint

        # if FLAGS.save_model_freq > 0:
        #     save_checkpoint(train_state)

        sharded_rng = next_rng()
        params = train_state.params
        old_params = params  # Initialize old_params

        dataset = iter(dataset)

        # global_step = 0  # To keep track of the global step for logging

        if FLAGS.weight_average and ema is None:
            print('Using weight average')
            ema = params

        # parallel_loss_fn = jax.pmap(loss_fn, axis_name='batch')
        parallel_loss_fn = jax.jit(loss_fn)

        if opt_state is None:
            opt_state = tayl_solver.init(params) 

        wd = jax.device_put_replicated(wd, jax.local_devices())
        new_params = train_state.params

        # tracemalloc.start(25)

        for global_step in tqdm(range(start_step, FLAGS.total_steps)):
            rng = next_rng()

            f_tayl = nt.taylor_expand(model.apply, train_state.params, FLAGS.taylor_order)

            parallel_apply_taylor_fn = partial(apply_taylor_fn, f_tayl)
            parallel_apply_taylor_fn = jax.pmap(parallel_apply_taylor_fn, axis_name='batch')
            
            inner_losses = []
            for i in range(FLAGS.inner_loop_iter):
                batch, dataset_metrics = next(dataset)

                sharded_batch = shard_batch(batch, num_devices)
                # print_shape(sharded_batch, prefix='sharded batch:')

                sharded_rng_rep = jax.device_put_replicated(rng, jax.local_devices())
                sharded_new_params = jax.device_put_replicated(new_params, jax.local_devices())
                sharded_old_params = jax.device_put_replicated(train_state.params, jax.local_devices())

                grads, base_loss, accuracy, wd_loss = parallel_apply_taylor_fn(sharded_batch, sharded_rng_rep, sharded_new_params, sharded_old_params, wd)
                # print_shape(grads, prefix='grads:')
                del sharded_old_params

                grads = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), grads)
                # print_shape(grads, prefix='grads mean:')
                wd_loss = jnp.mean(wd_loss)
                loss = jnp.mean(base_loss)
                accuracy = jnp.mean(accuracy)
                inner_losses.append(loss)


                if i == FLAGS.inner_loop_iter - 1:
                    # model_loss, model_accuracy = parallel_loss_fn(new_params, batch, rng)
                    model_loss, model_accuracy = parallel_loss_fn(new_params, batch, rng)
                    model_loss = jnp.mean(model_loss)
                    model_accuracy = jnp.mean(model_accuracy)

                del sharded_new_params, sharded_batch, sharded_rng_rep
                # old_params = new_params  # Should this be per inner step or per param update?
                updates, opt_state = tayl_solver.update(grads, opt_state, new_params)
                new_params = optax.apply_updates(new_params, updates)

                del grads, updates

                if FLAGS.weight_average:
                    alpha = FLAGS.weight_average_decay
                    ema = jax.tree_util.tree_map(lambda x, y: alpha*x + (1-alpha)*y, ema, new_params)


                rng = next_rng()

            del parallel_apply_taylor_fn, f_tayl
            train_state = train_state.replace(
                step=train_state.step+1,
                opt_state=opt_state,
                params=new_params
            )

            if global_step % 50 == 0:
                jax.clear_caches()
            
            if global_step % FLAGS.log_freq == 0:  
                inner_loss = jax.device_get(inner_losses)
                for inner_step, inner_loss in enumerate(inner_losses):
                    overall_step = (global_step * FLAGS.inner_loop_iter) + inner_step
                    wandb.log({
                        'inner_loss': inner_loss,
                        'learning_rate': lr_sched(overall_step),
                        'inner_step': overall_step,
                    })

                # logs the metrics from the final iteration of the inner loop
                loss, wd_loss, accuracy = jax.device_get((loss, wd_loss, accuracy))
                model_loss, model_accuracy = jax.device_get((model_loss, model_accuracy))

                wandb.log({
                    'taylor_loss': loss,
                    'wd_loss': wd_loss,
                    'accuracy': accuracy,
                    'model_loss': model_loss,
                    'model_accuracy': model_accuracy,
                    'global_step': global_step,
                    'dataset_total_tokens': dataset_metrics['dataset_total_tokens'],
                    'gpu_memory': get_gpu_memory()[0],
                    'avail_ram': get_ram(),
                })

            # jax.profiler.save_device_memory_profile(f"memory{global_step}.prof")

                # snapshot = tracemalloc.take_snapshot()
                # filter = tracemalloc.Filter(True, "/n/holystore01/LABS/barak_lab/Users/nabreu/.mamba/envs/EasyLM/lib/python3.10/site-packages/jax/_src/array.py")

                # top_stats = snapshot.filter_traces(filters=[filter]).statistics('lineno')

                # for stat in top_stats[:10]:
                #     print(stat)
                #     for line in stat.traceback.format():
                #         print(line, flush=True)
                # display_top(snapshot)
            
            if FLAGS.eval_freq and global_step % FLAGS.eval_freq == 0 and FLAGS.eval_steps > 0:
                eval_iterator = iter(eval_dataset)
                eval_metric_list = []
                if FLAGS.weight_average:
                    eval_params=ema
                else:
                    eval_params = train_state.params
                for _ in range(FLAGS.eval_steps):
                    eval_batch, _ = next(eval_iterator)

                    eval_metrics = sharded_eval_step(
                        eval_params, eval_batch, rng
                    )
                    eval_metric_list.append(eval_metrics)
                    rng = next_rng()

                avg_metrics = average_metrics(eval_metric_list)
                avg_metrics.update({'global_step': global_step})
                avg_metrics = jax.device_get(avg_metrics)
                wandb.log(avg_metrics)
                # if FLAGS.weight_average:
                #     run_evaluation(ema, eval_dataset, global_step, dataset_metrics['dataset_total_tokens'])
                # else:
                #     run_evaluation(new_params, eval_dataset, global_step, dataset_metrics['dataset_total_tokens'])

            if FLAGS.save_model_freq > 0 and global_step % FLAGS.save_model_freq == 0:
                if FLAGS.weight_average:
                    ema = jax.device_get(ema)
                    save_checkpoint(train_state, ema=ema)
                else:
                    save_checkpoint(train_state)

        if FLAGS.eval_freq and FLAGS.eval_steps > 0:
            eval_iterator = iter(eval_dataset)
            eval_metric_list = []
            if FLAGS.weight_average:
                eval_params=ema
            else:
                eval_params = train_state.params
            for _ in range(FLAGS.eval_steps):
                eval_batch, _ = next(eval_iterator)

                eval_metrics = sharded_eval_step(
                    eval_params, eval_batch, rng
                )
                eval_metric_list.append(eval_metrics)
                rng = next_rng()

            avg_metrics = average_metrics(eval_metric_list)
            avg_metrics.update({'global_step': global_step})
            avg_metrics = jax.device_get(avg_metrics)
            wandb.log(avg_metrics)


        if FLAGS.save_model_freq > 0:
            if FLAGS.weight_average:
                save_checkpoint(train_state, ema=ema)
            else:
                save_checkpoint(train_state)

        wandb.finish()

if __name__ == "__main__":
    print(jax.devices())
    mlxu.run(main)
