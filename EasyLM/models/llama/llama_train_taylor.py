import pprint
from functools import partial

from tqdm import tqdm, trange
import numpy as np
import mlxu
import subprocess as sp
import os
import neural_tangents as nt

import psutil
import tracemalloc
import linecache
import gc


import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
from flax.training.train_state import TrainState
from transformers import AutoTokenizer

from jaxopt import BacktrackingLineSearch

import optax

import jax.profiler
import pdb
import ipdb
import copy

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
    eval_freq=0,
    eval_steps=0,
    gradient_accumulation_steps=1,
    inner_loop_iter=100,
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
    taylor_order=2,
    inner_loop_passes=1,
    outer_loop_method='replace',
    inner_loop_sched='cosine',
    inner_loop_lr=0.0001,
    inner_loop_wd=0.5,
    inner_loop_end_lr=0.0,
    inner_b1=0.9,
    inner_b2=0.999,
    inner_clip_gradient=0.0,
    inner_loop_warmup=0.2,
    tayl_outer_loop_warmup=0.1, # for cosine_with_global_schedule
    tayl_outer_loop_lr=0.0001,
    adam_steps=0,
    start_step=0,
    start_tokens=0,
    weight_average=False,
    weight_average_decay=0.99,
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

def get_ram():
    return psutil.virtual_memory().available * 100 / psutil.virtual_memory().total

def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print("#%s: %s:%s: %.1f KiB"
              % (index, frame.filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))

def main(argv):
    print('Init',  get_gpu_memory())

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

    print('Dataset', get_gpu_memory())

    model = FlaxLLaMAForCausalLMModule(
        llama_config,
        dtype=get_float_dtype_by_name(FLAGS.dtype),
        param_dtype=get_float_dtype_by_name(FLAGS.param_dtype),
    )

    optimizer, optimizer_info = OptimizerFactory.get_optimizer(FLAGS.optimizer)

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

    def train_step(train_state, rng, batch, **kwargs): # Original train step fn
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
        train_state = train_state.apply_gradients(grads=grads)
        metrics = dict(
            loss=loss,
            accuracy=accuracy,
            learning_rate=optimizer_info['learning_rate_schedule'](train_state.step),
            gradient_norm=global_norm(grads),
            param_norm=global_norm(train_state.params),
            gpu_memory=get_gpu_memory()[0],
        )
        return train_state, rng_generator(), metrics
        

    def eval_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
        logits = model.apply(
            train_state.params, batch['input_tokens'], deterministic=True,
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

    # print(train_state_shapes)
    train_state_partition = match_partition_rules(
        LLaMAConfigurator.get_partition_rules(), train_state_shapes
    )

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
        in_shardings=(train_state_partition, PS(), PS()),
        out_shardings=(train_state_partition, PS(), PS()),
        donate_argnums=(0, 1),
    )

    sharded_eval_step = pjit(
        eval_step,
        in_shardings=(train_state_partition, PS(), PS()),
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

    # FLAGS.load_checkpoint = 'trainstate_params::/n/holyscratch01/barak_lab/Users/nabreu/SOO-LM/checkpoint/46400671/streaming_train_state_145920'
    # FLAGS.load_checkpoint = 'trainstate_params::/n/holyscratch01/barak_lab/Users/nabreu/SOO-LM/checkpoint/47006257/streaming_train_state_28800' # 20% warmup adam bs=32k
    # FLAGS.load_checkpoint = 'trainstate_params::/n/holyscratch01/barak_lab/Users/nabreu/SOO-LM/checkpoint/47252627/streaming_train_state_14400' #10% warmup adam bs=32k lr=0.01
    FLAGS.load_checkpoint = 'trainstate_params::/n/holyscratch01/barak_lab/Users/nabreu/SOO-LM/checkpoint/47578767/streaming_train_state_1000' #1000step warmup adam bs=32k lr=0.01
    mesh = LLaMAConfigurator.get_jax_mesh(FLAGS.mesh_dim)
    with mesh:
        # print('checkpoint:', FLAGS.load_checkpoint)
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

        param_count, param_count_nonembed = count_params(train_state.params)
        param_count = jax.device_get(param_count)
        param_count_nonembed = jax.device_get(param_count_nonembed)
        logger.log({"param_count_nonembed": param_count_nonembed})
        logger.log({"param_count": param_count})

        start_step = int(jax.device_get(train_state.step))

        if FLAGS.save_model_freq > 0:
            save_checkpoint(train_state)

        sharded_rng = next_rng()

        # step_counter = trange(start_step, FLAGS.total_steps, ncols=0)

        taylor_order = FLAGS.taylor_order
        lr = FLAGS.inner_loop_lr
        wd = FLAGS.inner_loop_wd
        gradient_accumulation_steps = FLAGS.gradient_accumulation_steps


        if FLAGS.outer_loop_method == 'update':
            outer_lr_sched = optax.warmup_cosine_decay_schedule(
                init_value=FLAGS.optimizer.adamw_optimizer.init_lr,
                peak_value=FLAGS.optimizer.adamw_optimizer.lr,
                warmup_steps=FLAGS.optimizer.adamw_optimizer.lr_warmup_steps,
                decay_steps=FLAGS.optimizer.adamw_optimizer.lr_decay_steps,
                end_value=FLAGS.optimizer.adamw_optimizer.end_lr,
            )

        

        dataset = iter(dataset)
        exit = False
        step = 0
        inner_step = 0
        if FLAGS.adam_steps:
            assert FLAGS.optimizer.accumulate_gradient_steps == gradient_accumulation_steps * FLAGS.inner_loop_iter
            taylor_steps = FLAGS.total_steps - FLAGS.adam_steps
        else:
            taylor_steps = FLAGS.total_steps

        if FLAGS.weight_average:
            ema = train_state.params

        if FLAGS.inner_loop_sched == 'global_cosine':
            decay_steps = taylor_steps*FLAGS.inner_loop_iter*gradient_accumulation_steps
            decay_steps = int(decay_steps)
            if FLAGS.inner_loop_warmup <= 1.0:
                warmup = int(FLAGS.inner_loop_warmup*decay_steps)
            else:
                warmup = FLAGS.inner_loop_warmup

            if isinstance(warmup, tuple):
                warmup = int(warmup[0])
            

            inner_lr_sched = optax.warmup_cosine_decay_schedule(
                init_value=lr*0.1,
                peak_value=lr,
                warmup_steps=warmup,
                decay_steps=decay_steps,
                end_value=FLAGS.inner_loop_end_lr,
            )
            inner_optimizer_info = dict(
                    learning_rate_schedule=inner_lr_sched,
                )
            
            tayl_solver = build_optimizer(inner_lr_sched, FLAGS.inner_b1, FLAGS.inner_b2, FLAGS.inner_clip_gradient, wd)

            opt_state = tayl_solver.init(train_state.params)
        elif FLAGS.inner_loop_sched == 'cosine_with_global_schedule':
            global_decay_steps = taylor_steps
            global_warmup = FLAGS.tayl_outer_loop_warmup # TODO: Add flag
            lr = FLAGS.tayl_outer_loop_lr
            global_lr_sched = optax.warmup_cosine_decay_schedule(
                init_value=lr*0.1,
                peak_value=lr,
                warmup_steps=global_warmup,
                decay_steps=global_decay_steps,
                end_value=0.00001
            )
            global_lr_sched_info = dict(
                learning_rate_schedule=global_lr_sched
            )

        prev_opt_state = None
        with tqdm(initial=start_step, total=FLAGS.total_steps) as progress_bar:
            while not exit:       
                rng_generator = JaxRNG(sharded_rng)
                f_tayl = nt.taylor_expand(model.apply, train_state.params, taylor_order)
        
                def apply_taylor_fn(f_tayl, batch, rng, new_params, old_params, weight_decay):
                    rng_generator = JaxRNG(rng)
                    def loss_and_accuracy_with_wd(params, old_params):
                        logits = f_tayl(
                            params, batch['input_tokens'], deterministic=False,
                            rngs=rng_generator(LLaMAConfigurator.rng_keys()),
                        ).logits

                        return cross_entropy_loss_and_accuracy_with_weight_decay(logits, batch['target_tokens'], params, old_params, batch['loss_masks'], weight_decay=weight_decay)

                    def loss_and_accuracy(params):
                        logits = f_tayl(
                            params, batch['input_tokens'], deterministic=False,
                            rngs=rng_generator(LLaMAConfigurator.rng_keys()),
                        ).logits

                        return cross_entropy_loss_and_accuracy(logits, batch['target_tokens'], batch['loss_masks'])
                    grad_fn = jax.value_and_grad(loss_and_accuracy_with_wd, has_aux=True)
                    (wd_loss, wd_accuracy), grads = grad_fn(new_params, old_params)
                    loss, accuracy = loss_and_accuracy(new_params)
                    return grads, loss, accuracy, wd_loss

                # actual loss fn - take average over batch size 4m
                @jax.jit
                def loss_fn(params, batch, rng):
                    rng_generator = JaxRNG(rng)
                    logits = model.apply(
                        params, batch['input_tokens'], deterministic=False,
                        rngs=rng_generator(LLaMAConfigurator.rng_keys()),
                    ).logits
                    return cross_entropy_loss_and_accuracy(
                        logits, batch['target_tokens'], batch['loss_masks']
                    )


                new_params = train_state.params
                if FLAGS.inner_loop_sched == 'cosine':
                    if FLAGS.inner_loop_warmup <= 1.0:
                        warmup = FLAGS.inner_loop_warmup*FLAGS.inner_loop_iter*gradient_accumulation_steps,
                        if isinstance(warmup, tuple):
                            warmup = warmup[0]
                        warmup = int(warmup)
                    else:
                        warmup = FLAGS.inner_loop_warmup
                    lr_sched = optax.warmup_cosine_decay_schedule(
                        init_value=0.1*lr, 
                        peak_value=lr,
                        warmup_steps=warmup,
                        decay_steps=FLAGS.inner_loop_iter*gradient_accumulation_steps, 
                        end_value=FLAGS.inner_loop_end_lr
                    )
                    inner_optimizer_info = dict(
                        learning_rate_schedule=lr_sched,
                    )
                elif FLAGS.inner_loop_sched == 'constant':
                    lr_sched = lr
                elif FLAGS.inner_loop_sched == 'cosine_with_global_schedule':
                    if FLAGS.inner_loop_warmup <= 1.0:
                        warmup = FLAGS.inner_loop_warmup*FLAGS.inner_loop_iter*gradient_accumulation_steps,
                        if isinstance(warmup, tuple):
                            warmup = warmup[0]
                        warmup = int(warmup)
                    else:
                        warmup = FLAGS.inner_loop_warmup
                    max_lr = global_lr_sched(step) # use lr from global schedule
                    lr_sched = optax.warmup_cosine_decay_schedule(
                        init_value=0.1*max_lr, 
                        peak_value=max_lr,
                        warmup_steps=warmup,
                        decay_steps=FLAGS.inner_loop_iter*gradient_accumulation_steps, 
                        end_value=FLAGS.inner_loop_end_lr
                    )
                  
                if FLAGS.inner_loop_sched != 'global_cosine':
                    tayl_solver = build_optimizer(lr_sched, FLAGS.inner_b1, FLAGS.inner_b2, FLAGS.inner_clip_gradient)
                    
                    opt_state = tayl_solver.init(new_params)
                    # print(opt_state)

                    # replace mu, nu, count to retain adam state
                    if prev_opt_state is not None:
                        if FLAGS.inner_clip_gradient: # 1 if clip gradient is enabled, 0 otherwise
                            adamw_opt_state = opt_state[1]
                            prev_adamw_opt_state = prev_opt_state[1]
                            adamw_opt_state = (adamw_opt_state[0]._replace(
                                mu=prev_adamw_opt_state[0].mu,
                                nu=prev_adamw_opt_state[0].nu,
                                count=prev_adamw_opt_state[0].count,
                                ),
                            ) + adamw_opt_state[1:]


                            opt_state = (opt_state[0],) + (adamw_opt_state,)
                            # print('updated opt state')
                            # print(opt_state)
                        else:
                            opt_state = (opt_state[0]._replace(
                                mu=prev_opt_state[0].mu,
                                nu=prev_opt_state[0].nu,
                                count=prev_opt_state[0].count,
                                ),
                            )  + opt_state[1:]

                apply_fn_jit = jax.jit(apply_taylor_fn, static_argnums=0)

                grads = jax.tree_util.tree_map(jnp.zeros_like, new_params)
                
                loss, accuracy, loss_wd = 0, 0, 0
                model_loss, model_accuracy = 0, 0
                inner_losses = []
                for i in range(FLAGS.inner_loop_iter*gradient_accumulation_steps):
                    if i % gradient_accumulation_steps == 0:
                        loss, accuracy, loss_wd = 0, 0, 0
                        model_loss, model_accuracy = 0, 0

                    # try:
                    batch, dataset_metrics = next(dataset)
                    batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
               
                    # except:
                    #     print('Dataset exhausted')
                    #     exit = True
                    #     break

                    grads_, loss_, accuracy_, loss_wd_ = apply_fn_jit(f_tayl, batch, sharded_rng, new_params, train_state.params, wd)
                    grads = jax.tree_util.tree_map(lambda x, y: x + y, grads, grads_)
                    loss += loss_
                    accuracy += accuracy_
                    loss_wd += loss_wd_
                    inner_losses.append(loss_)

                    model_loss_, model_accuracy_ = loss_fn(new_params, batch, sharded_rng)
                    model_loss += model_loss_
                    model_accuracy += model_accuracy_

                    if (i+1) % gradient_accumulation_steps == 0:
                        grads = jax.tree_util.tree_map(lambda x: x / gradient_accumulation_steps, grads)

                        updates, opt_state = tayl_solver.update(grads, opt_state, new_params)
                        new_params = optax.apply_updates(new_params, updates)

                        grads = jax.tree_util.tree_map(jnp.zeros_like, grads)

                        # Will log the average loss of the _last_ inner loop step
                        loss = loss / gradient_accumulation_steps
                        accuracy = accuracy / gradient_accumulation_steps
                        loss_wd = loss_wd / gradient_accumulation_steps

                        model_loss = model_loss / gradient_accumulation_steps
                        model_accuracy = model_accuracy / gradient_accumulation_steps
        
                    sharded_rng = rng_generator()

                if exit: 
                    break

                if FLAGS.weight_average:
                    # ema
                    alpha = FLAGS.weight_average_decay
                    ema = jax.tree_util.tree_map(lambda x, y: alpha*x + (1-alpha)*y, ema, new_params)

                if FLAGS.outer_loop_method == 'replace':
                    train_state = train_state.replace(
                        step=train_state.step+1,
                        params=new_params
                    )
                elif FLAGS.outer_loop_method == 'update':
                    # use same inner loop lr, large outer loop lr
                    outer_lr = outer_lr_sched(train_state.step)
               
                    print('lr: ' + str(outer_lr))
                    dir = jax.tree_util.tree_map(lambda x, y: x - y, new_params, train_state.params)
                    updated_params = jax.tree_util.tree_map(lambda x, y: x + outer_lr*y, train_state.params, dir)
                    train_state = train_state.replace(
                        step=train_state.step+1,
                        params=updated_params
                    )
                    # train_state = train_state.apply_gradients(grads=diff) # subtract old-new (opposite direction) since gradient will be subtracted in the optimizer
                    
                elif FLAGS.outer_loop_method == 'linesearch':
                    pre_fetched_batches = []
                    for _ in range(FLAGS.inner_loop_iter * gradient_accumulation_steps):
                        try:
                            batch, _ = next(dataset)
                            pre_fetched_batches.append(batch)
                        except StopIteration:
                            print('Dataset exhausted')
                            exit = True
                            break
                    
                    if exit:
                        break
                    # loss_partial = partial(compute_average_loss, dataset=dataset, rng=sharded_rng, loss_fn=loss_fn, batch_accumulation_steps=FLAGS.inner_loop_iter*gradient_accumulation_steps)
                    dir = jax.tree_util.tree_map(lambda x, y: x - y, new_params, train_state.params)
                    losses = []
                    for step_size in [1/jnp.sqrt(2)**i for i in range(5)]:
                        # Compute loss using pre-fetched batches
                        updated_params = jax.tree_util.tree_map(lambda x, y: x + step_size*y, train_state.params, dir)
                        accumulated_loss = 0.0
                        for batch in pre_fetched_batches:
                            sharded_rng, subrng = jax.random.split(sharded_rng)
                            loss, _ = loss_fn(updated_params, batch, subrng)
                            accumulated_loss += loss
                        
                        average_loss = accumulated_loss / len(pre_fetched_batches)
                        losses.append((step_size, average_loss))
                    step_size, loss = min(losses, key=lambda x: x[1])
                    print('Step size:', step_size)
                    updated_params = jax.tree_util.tree_map(lambda x, y: x + step_size*y, train_state.params, dir)
                    train_state = train_state.replace(
                        step=train_state.step+1,
                        params=updated_params
                    )
                elif FLAGS.outer_loop_method == 'fixed_1/4_lr':
                    # use fixed lr
                    dir = jax.tree_util.tree_map(lambda x, y: x - y, new_params, train_state.params)
                    updated_params = jax.tree_util.tree_map(lambda x, y: x + (1/4)*y, train_state.params, dir)
                    train_state = train_state.replace(
                        step=train_state.step+1,
                        params=updated_params
                    )
                elif FLAGS.outer_loop_method == 'fixed_1_lr':
                    # use fixed lr
                    dir = jax.tree_util.tree_map(lambda x, y: x - y, new_params, train_state.params)
                    updated_params = jax.tree_util.tree_map(lambda x, y: x + y, train_state.params, dir)
                    train_state = train_state.replace(
                        step=train_state.step+1,
                        params=updated_params
                    )
                elif FLAGS.outer_loop_method == 'fixed_1/2_lr':
                    # use fixed lr
                    dir = jax.tree_util.tree_map(lambda x, y: x - y, new_params, train_state.params)
                    updated_params = jax.tree_util.tree_map(lambda x, y: x + (1/2)*y, train_state.params, dir)
                    train_state = train_state.replace(
                        step=train_state.step+1,
                        params=updated_params
                    )
                else:
                    raise ValueError('Invalid outer loop method')

                try:
                    model_perplexity = jnp.exp(model_loss)
                except OverflowError:
                    model_perplexity = jnp.float32("inf")

                metrics = dict(
                    loss=model_loss,
                    taylor_loss = loss,
                    taylor_loss_wd = loss_wd,
                    perplexity=model_perplexity,
                    accuracy=model_accuracy,
                    # learning_rate=optimizer_info['learning_rate_schedule'](train_state.step),
                    gradient_norm=global_norm(grads), # not sure if this is accurate?
                    param_norm=global_norm(train_state.params),
                    gpu_memory=get_gpu_memory()[0],
                )
                # jax.clear_caches()
                del apply_taylor_fn, apply_fn_jit
                sharded_rng = rng_generator()

                for i, inner_loss in enumerate(inner_losses):
                    if FLAGS.inner_loop_sched == 'cosine' or FLAGS.inner_loop_sched == 'cosine_with_global_schedule':
                        inner_lr = lr_sched(i)
                    elif FLAGS.inner_loop_sched == 'constant':
                        inner_lr = lr
                    elif FLAGS.inner_loop_sched == 'global_cosine':
                        inner_lr = inner_lr_sched(step*FLAGS.inner_loop_iter*gradient_accumulation_steps + i)
                    inner_loop_metrics = {"inner_loss": inner_loss, "inner_step": inner_step, "inner_lr": inner_lr}
                    inner_loop_metrics = jax.device_get(inner_loop_metrics)
                    logger.log(inner_loop_metrics)
                    inner_step += 1


                if step % FLAGS.log_freq == 0:
                    log_metrics = {"step": step}
                    log_metrics.update(metrics)
                    log_metrics.update(dataset_metrics)
                    

                    if FLAGS.eval_freq != 0 and FLAGS.eval_steps > 0: # eval_freq must be | by log_freq
                        if step % FLAGS.eval_freq == 0:
                            eval_metric_list = []
                            for _ in range(FLAGS.eval_steps):
                                eval_batch, _ = next(eval_iterator)
                                sharded_rng, eval_metrics = sharded_eval_step(
                                    train_state, sharded_rng, eval_batch
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

                step += 1
                prev_opt_state = opt_state
                if step == taylor_steps:
                    exit = True
                progress_bar.update(1)

            

            start = step * FLAGS.optimizer.accumulate_gradient_steps

            
            if FLAGS.adam_steps:
                print('Running adam for' + str(FLAGS.adam_steps))
                train_state = train_state.replace(step=start)
                for step in tqdm(range(start, FLAGS.total_steps * FLAGS.optimizer.accumulate_gradient_steps)):
                    batch, dataset_metrics = next(dataset)
                    train_state, sharded_rng, metrics = sharded_train_step(
                        train_state, sharded_rng, batch
                    )

                    if step % (FLAGS.log_freq * FLAGS.optimizer.accumulate_gradient_steps) == 0:
                        log_metrics = {"step": step / FLAGS.optimizer.accumulate_gradient_steps}
                        log_metrics.update(metrics)
                        log_metrics.update(dataset_metrics)
                        log_metrics = jax.device_get(log_metrics)
                        logger.log(log_metrics)
                        tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")

            if FLAGS.save_model_freq > 0:
                save_checkpoint(train_state)

            if FLAGS.eval_freq != 0:
                log_metrics = {"step": step}
                eval_metric_list = []
                for _ in range(FLAGS.eval_steps):
                    eval_batch, _ = next(eval_iterator)
                    sharded_rng, eval_metrics = sharded_eval_step(
                        train_state, sharded_rng, eval_batch
                    )
                    eval_metric_list.append(eval_metrics)
                log_metrics.update(average_metrics(eval_metric_list))
    
                log_metrics = jax.device_get(log_metrics)
                logger.log(log_metrics)
                tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")


            # if FLAGS.weight_average:
            #     train_state = train_state.replace(params=ema)

            #     eval_metric_list = []
            #     for _ in range(FLAGS.eval_steps):
            #         eval_batch, _ = next(eval_iterator)
            #         sharded_rng, eval_metrics = sharded_eval_step(
            #             train_state, sharded_rng, eval_batch
            #         )
            #         eval_metric_list.append(eval_metrics)
            #     metrics.update(average_metrics(eval_metric_list))
            #     metrics = jax.device_get(metrics)
            #     logger.log(metrics)




if __name__ == "__main__":
    mlxu.run(main)
