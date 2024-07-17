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

import optax

import jax.profiler
import pdb
import ipdb

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
    learning_rate=0.0001,
    weight_decay=0.5,
    tokenizer='openlm-research/open_llama_3b_v2',
    train_dataset=DatasetFactory.get_default_config(),
    eval_dataset=DatasetFactory.get_default_config(),
    optimizer=OptimizerFactory.get_default_config(),
    checkpointer=StreamingCheckpointer.get_default_config(),
    llama=LLaMAConfigurator.get_default_config(),
    logger=mlxu.WandBLogger.get_default_config(),
    log_all_worker=False,
    jax_distributed=JaxDistributedConfig.get_default_config(),
)
def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

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
        
    
    # Taylor method that changes batch within inner loop by passing in a larger global batch and splitting it
    # Doesn't really work with memory constraints
    def train_step_tayl_minibatch(train_state, rng, batch, **kwargs):
        rng_generator = JaxRNG(rng)
        f_tayl = nt.taylor_expand(model.apply, train_state.params, 2)
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))

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

        lr = kwargs['learning_rate']
        wd = kwargs['weight_decay']

        new_params = train_state.params
        tayl_solver = optax.adam(learning_rate=lr) # TODO: check this?
        jax.debug.print(str(optimizer_info['learning_rate_schedule'](train_state.step)))
        opt_state = tayl_solver.init(new_params)

        inner_loop_iter = 100
        batch_size = batch['input_tokens'].shape[0]
        minibatch_size = batch_size // inner_loop_iter
        for j in range(0, batch_size, minibatch_size):

            minibatch = {
                'input_tokens': batch['input_tokens'][j:j+minibatch_size],
                'target_tokens': batch['target_tokens'][j:j+minibatch_size],
                'loss_masks': batch['loss_masks'][j:j+minibatch_size]
            }
            grads, loss, accuracy = apply_taylor_fn(f_tayl, minibatch, rng, new_params, train_state.params, 0.5) # TODO: fix weight decay param
            updates, opt_state = tayl_solver.update(grads, opt_state, new_params)
            new_params = optax.apply_updates(new_params, updates)

        train_state = train_state.replace(
            step=train_state.step+1,
            params=new_params
        ) 
        # train_state = train_state.apply_gradients(grads=grads)
        metrics = dict(
            loss=loss,
            accuracy=accuracy,
            learning_rate=optimizer_info['learning_rate_schedule'](train_state.step), # doesn't mean anything for taylor
            gradient_norm=global_norm(grads), # not sure if this is accurate?
            param_norm=global_norm(train_state.params),
            gpu_memory=get_gpu_memory()[0],
        )
        jax.clear_caches()
        return train_state, rng_generator(), metrics


    # Old taylor method with fixed batch within inner loop
    def train_step_tayl(train_state, rng, batch, **kwargs):
        rng_generator = JaxRNG(rng)
        f_tayl = nt.taylor_expand(model.apply, train_state.params, 2)
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))

        # lr = kwargs['learning_rate']
        # wd = kwargs['weight_decay']
        lr = 0.0001
        wd = 0.5

        # Important to define this function within the train_step_tayl function!
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

        for i in range(100):
            grads, loss, accuracy = apply_fn_jit(f_tayl, batch, rng, new_params, train_state.params, wd)
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
        del apply_taylor_fn, apply_fn_jit
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


    # sharded_train_step = pjit(
    #     train_step_tayl,
    #     in_shardings=(train_state_partition, PS(), PS()),
    #     out_shardings=(train_state_partition, PS(), PS()),
    #     donate_argnums=(0, 1),
    # )

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

    mesh = LLaMAConfigurator.get_jax_mesh(FLAGS.mesh_dim)
    with mesh:
        train_state, restored_params = None, None
        if FLAGS.load_checkpoint != '':
            train_state, restored_params = checkpointer.load_trainstate_checkpoint(
                FLAGS.load_checkpoint, train_state_shapes, shard_fns
            )

        if train_state is None and restored_params is None:
            # Initialize from scratch
            train_state = sharded_init_fn(next_rng())
        elif train_state is None and restored_params is not None:
            # Restore from params but initialize train_state
            train_state = sharded_create_trainstate_from_params(restored_params)
            del restored_params

        param_count = sum(x.size for x in jax.tree_leaves(train_state.params))
        param_count = jax.device_get(param_count)
        logger.log({"param_count": param_count})

        start_step = int(jax.device_get(train_state.step))

        if FLAGS.save_model_freq > 0:
            save_checkpoint(train_state)

        sharded_rng = next_rng()

        # step_counter = trange(start_step, FLAGS.total_steps, ncols=0)


        lr = FLAGS.learning_rate
        wd = FLAGS.weight_decay
        gradient_accumulation_steps = FLAGS.gradient_accumulation_steps

        dataset = iter(dataset)
        exit = False
        step = 0
        with tqdm(initial=start_step, total=FLAGS.total_steps) as progress_bar:
            while not exit:       
                rng_generator = JaxRNG(sharded_rng)
                f_tayl = nt.taylor_expand(model.apply, train_state.params, 2)
        
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

                grads = jax.tree_util.tree_map(jnp.zeros_like, new_params)
                
                loss, accuracy = 0, 0
                for i in range(100*gradient_accumulation_steps):
                    if i % gradient_accumulation_steps == 0:
                        loss, accuracy = 0, 0

                    try:
                        batch, dataset_metrics = next(dataset)
                        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
               
                    except:
                        print('Dataset exhausted')
                        exit = True
                        break

                    grads_, loss_, accuracy_ = apply_fn_jit(f_tayl, batch, sharded_rng, new_params, train_state.params, wd)
                    grads = jax.tree_util.tree_map(lambda x, y: x + y, grads, grads_)
                    loss += loss_
                    accuracy += accuracy_

                    if (i+1) % gradient_accumulation_steps == 0:
                        grads = jax.tree_util.tree_map(lambda x: x / gradient_accumulation_steps, grads)

                        updates, opt_state = tayl_solver.update(grads, opt_state, new_params)
                        new_params = optax.apply_updates(new_params, updates)

                        grads = jax.tree_util.tree_map(jnp.zeros_like, grads)

                        # Will log the average loss of the _last_ inner loop step
                        loss = loss / gradient_accumulation_steps
                        accuracy = accuracy / gradient_accumulation_steps
        
                        
                if exit: 
                    break

                train_state = train_state.replace(
                    step=train_state.step+1,
                    params=new_params
                )

                try:
                    perplexity = jnp.exp(loss)
                except OverflowError:
                    perplexity = jnp.float32("inf")

                metrics = dict(
                    loss=loss,
                    perplexity=perplexity,
                    accuracy=accuracy,
                    learning_rate=optimizer_info['learning_rate_schedule'](train_state.step), # doesn't mean anything for taylor
                    gradient_norm=global_norm(grads), # not sure if this is accurate?
                    param_norm=global_norm(train_state.params),
                    gpu_memory=get_gpu_memory()[0],
                )
                # jax.clear_caches()
                del apply_taylor_fn, apply_fn_jit
                sharded_rng = rng_generator()


                if step % FLAGS.log_freq == 0:
                    log_metrics = {"step": step}
                    log_metrics.update(metrics)
                    log_metrics.update(dataset_metrics)
                    log_metrics = jax.device_get(log_metrics)
                    logger.log(log_metrics)
                    tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")
                
                if FLAGS.eval_freq != 0 and FLAGS.eval_steps > 0:
                    if step % FLAGS.eval_freq == 0:
                        eval_metric_list = []
                        for _ in range(FLAGS.eval_steps):
                            eval_batch, _ = next(eval_iterator)
                            sharded_rng, eval_metrics = sharded_eval_step(
                                train_state, sharded_rng, eval_batch
                            )
                            eval_metric_list.append(eval_metrics)
                        metrics.update(average_metrics(eval_metric_list))

                if FLAGS.save_milestone_freq > 0 and (step + 1) % FLAGS.save_milestone_freq == 0:
                    save_checkpoint(train_state, milestone=True)
                elif FLAGS.save_model_freq > 0 and (step + 1) % FLAGS.save_model_freq == 0:
                    save_checkpoint(train_state)

                step += 1
                if step == FLAGS.total_steps:
                    exit = True
                progress_bar.update(1)

            if FLAGS.save_model_freq > 0:
                save_checkpoint(train_state)




if __name__ == "__main__":
    mlxu.run(main)
