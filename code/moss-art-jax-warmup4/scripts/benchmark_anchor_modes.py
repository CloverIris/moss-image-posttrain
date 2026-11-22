#!/usr/bin/env python3
"""Matched four-GPU ART speed benchmark. All options use --config JSON.

Accepts the exact probe_fp8_anchor.py wrapper: base_config, overrides, probe.
The probe options output_dir, initial_checkpoint, sample_id, native_conv_build,
and dry_run apply. Runs BF16, MXFP8, and calibrated MXFP8 from identical states.
No production checkpoint or trainer is mutated. GPU kernels are explicitly warmed.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_fp8_anchor import load_settings, cached_record, prepare_environment, anchor_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    c, probe, base_path = load_settings(args.config)
    if c['sampling']['sampler'] != 'euler' or c['sampling']['scheduler'] != 'moss':
        raise ValueError('Matched benchmark requires Euler / MOSS scheduler')
    if c['sampling']['rollout_batch_size'] != 1 or c['training']['microbatch_size'] != 1:
        raise ValueError('Matched benchmark requires per-GPU rollout/train batch 1')
    if not probe['initial_checkpoint']:
        raise ValueError('An explicit initial checkpoint with optimizer is required')
    record, info = cached_record(c, probe['sample_id'])
    if probe['dry_run']:
        print(json.dumps(dict(dry_run=True, config=c, probe=probe, record=info), indent=2))
        return
    run(c, probe, base_path, record, info)


def run(c, probe, base_path, record, info):
    from moss_art.runtime import preflight, configure_process
    from moss_art.locking import output_lock
    inventory = preflight(c)
    output = Path(probe['output_dir'])
    with output_lock(output):
        if (output / 'report.json').exists():
            raise FileExistsError('Completed benchmark exists; choose a fresh output_dir')
        prepare_environment(output, probe)
        configure_process(c, c['runtime']['devices'])
        import jax
        import numpy as np
        from moss_art.art import covariance, prompt_windows
        from moss_art.mesh_runtime import MeshPool, _loss_vjp
        from moss_art.trainer import (write_json, make_payloads, paired_replay_payloads,
                                     aggregate_gradients, apply_window_update, _optimizer)
        from moss_art.checkpoint import load_checkpoint
        from moss_art.memory import memory_snapshot
        from moss_art.replay import host_snapshot
        from PIL import Image
        report = dict(schema_version=1, status='running', base_config=base_path,
            dataset_record=info, gpu_inventory=inventory, config=c, probe=probe,
            budget=dict(prompts=1, trajectories=8, points_per_trajectory=8,
                        points_per_window=32, resolution=[768,768], sampling_steps=48),
            timings={}, branches={}, memory={}, stage_order=[], quality_gain_measured=False,
            timing_policy='Warm synchronized wall time, same shape/seed/checkpoint. Headline model compute excludes reload, VAE, reward, Qwen and warmup. Common decode/reward warmed once before all measured branches.')
        write_json(output/'resolved_config.json', c)
        started = time.monotonic()
        def stage(name, fn):
            jax.effects_barrier()
            begin = time.monotonic()
            result = fn()
            for leaf in jax.tree.leaves(result):
                if isinstance(leaf,jax.Array): leaf.block_until_ready()
            jax.effects_barrier()
            seconds = time.monotonic()-begin
            report['timings'][name] = seconds
            report['memory'][name] = memory_snapshot(jax.local_devices())
            report['stage_order'].append(name)
            write_json(output/'progress.json', report)
            print(json.dumps(dict(event='benchmark_stage', stage=name, seconds=seconds)), flush=True)
            return result
        def digest(tree):
            leaves, structure = jax.tree.flatten(tree)
            h = hashlib.sha256(str(structure).encode())
            for leaf in leaves:
                arr=np.asarray(leaf)
                h.update(str((arr.shape,str(arr.dtype))).encode())
                h.update(arr.tobytes())
            return h.hexdigest()
        try:
            theta_host, opt_host, metadata = load_checkpoint(probe['initial_checkpoint'],expected_metadata=None)
            theta_host=jax.tree.map(lambda v: np.asarray(v).copy() if hasattr(v,'dtype') else v,theta_host)
            opt_host=jax.tree.map(lambda v: np.asarray(v).copy() if hasattr(v,'dtype') else v,opt_host)
            theta_digest=digest(theta_host)
            optimizer_digest=digest(opt_host)
            report['checkpoint_origin']=dict(path=probe['initial_checkpoint'], update=metadata.get('update'),
                rollout=metadata.get('rollout'), lora_sha256=theta_digest, optimizer_sha256=optimizer_digest)
            pool = stage('initial_bf16_model_load',lambda: MeshPool(c,initial_lora=theta_host))
            with pool:
                pool.info[0]['lora']=None
                if metadata.get('lora_config') != c['lora'] or metadata.get('model_config') != pool.info[0]['model_config']:
                    raise ValueError('Checkpoint LoRA/model configuration differs')
                tx=_optimizer(c)
                expected=jax.eval_shape(tx.init,pool.worker.lora)
                actual_leaves,actual_tree=jax.tree.flatten(opt_host)
                expected_leaves,expected_tree=jax.tree.flatten(expected)
                if actual_tree != expected_tree or any(a.shape!=b.shape for a,b in zip(actual_leaves,expected_leaves)):
                    raise ValueError('Checkpoint optimizer structure differs')
                del expected, actual_leaves,expected_leaves
                full_payload=make_payloads(c,[record],0)
                warm_payload=[dict(p,seeds=p['seeds'][:1]) for p in full_payload]
                report['seeds']=[seed for p in full_payload for seed in p['seeds']]
                common_warmed=False
                def save_images(label,outputs,images,rewards,details):
                    folder=output/'images'/label
                    folder.mkdir(parents=True,exist_ok=True)
                    samples=[]
                    for item in outputs:
                        for seed,sample_id in zip(item['seeds'],item['sample_ids']):
                            samples.append(dict(index=len(samples),seed=seed,sample_id=sample_id))
                    for i,image in enumerate(images):Image.fromarray(image).save(folder/f'{i:04d}.png')
                    write_json(folder/'samples.json',samples)
                    write_json(folder/'rewards.json',dict(rewards=rewards.tolist(),details=details))
                def common(label,outputs,save):
                    stage(label+'_dit_release',pool.pause)
                    decoded=stage(label+'_vae',lambda:pool.decode([x['endpoints'] for x in outputs]))
                    images=np.concatenate([x['images'] for x in decoded]);del decoded
                    details=stage(label+'_reward',lambda:pool.score(images))
                    rewards=np.asarray([v['reward'] for v in details],np.float32)
                    if not np.isfinite(rewards).all():raise FloatingPointError('Nonfinite reward')
                    if save:stage(label+'_save_images',lambda:save_images(label,outputs,images,rewards,details))
                    del images
                    stage(label+'_bf16_reload',pool.resume)
                    return rewards
                def reset_theta(label,restamp):
                    stage(label+'_reset_theta0',lambda:pool.call('set_lora',[dict(lora=theta_host) for _ in pool.workers]))
                    if digest(pool.worker.lora)!=theta_digest:raise AssertionError('Checkpoint fork did not restore theta0')
                    if digest(opt_host)!=optimizer_digest:raise AssertionError('Host optimizer origin was mutated')
                    if restamp:
                        for shard in pool.shards:shard['parameter_version']=pool.parameter_version
                    return pool.parameter_version
                def gradient(window):
                    return aggregate_gradients(pool.call('gradient',[dict(window=window) for _ in pool.workers]))
                def train(label,payloads,source_precision):
                    source_version=pool.parameter_version
                    if any(s['parameter_version']!=source_version for s in pool.shards):
                        raise AssertionError('Replay stamp must match verified theta0')
                    if digest(pool.worker.lora)!=theta_digest:raise AssertionError('Training must start at theta0')
                    pool.config['sampling']['rollout_mode']=label
                    if label=='mxfp8_calibrated':
                        # Bind without executing a cold prepass, then warm one BF16 reference batch.
                        pool.config['sampling']['rollout_mode']='mxfp8'
                        stage(label+'_bind_replay',lambda:pool.call('prepare_update',payloads))
                        pool.config['sampling']['rollout_mode']=label
                        def warm_reference():
                            host,_=next(pool._window_batches(1,include_inactive=True))
                            vals={k:pool.parallel.put(v) for k,v in host.items()}
                            prediction=pool.worker.model.predict_reference(pool.worker.lora,vals['state'],vals['sigma'],vals['positive'],
                                **pool.worker._kwargs(vals['negative'],(vals['pl'],vals['nl'])))
                            prediction.block_until_ready()
                            # Warm the B-only scalar loss executable without taking gradients through DiT.
                            settings=c['art']
                            value=_loss_vjp(prediction,vals['old'],vals['sigma'],vals['direction'],vals['magnitude'],vals['coef'],vals['anchor'],vals['valid'],
                                noise=settings['noise_level'],kl=settings['kl_threshold'],eps=settings['eps'],anchor_mode='calibrated')
                            jax.block_until_ready(value)
                        stage(label+'_reference_warmup',warm_reference)
                        stage(label+'_reference_prepass',lambda:pool.call('prepare_update',payloads))
                        anchor_before=anchor_digest(pool.anchor_cache)
                        report['anchor']=dict(pool.anchor_metrics,sha256=anchor_before)
                    else:
                        stage(label+'_bind_replay',lambda:pool.call('prepare_update',payloads))
                        anchor_before=None
                    # Warm DiT VJP, accumulation, loss and optimizer without mutating theta0 or opt0.
                    grad,stats=stage(label+'_training_warmup',lambda:gradient(0))
                    state=pool.place_state(opt_host)
                    warm=stage(label+'_optimizer_warmup',lambda:apply_window_update(tx,grad,state,pool.worker.lora,stats['kept_count']))
                    del warm,grad,stats
                    gc.collect()
                    if pool.parameter_version!=source_version or digest(pool.worker.lora)!=theta_digest:
                        raise AssertionError('Warmup mutated theta0')
                    branch=dict(source_version=source_version,rollout_precision=source_precision,
                        initial_lora_sha256=theta_digest,initial_optimizer_sha256=optimizer_digest,windows=[])
                    for window,name in ((0,'A'),(1,'B')):
                        def update_window():
                            nonlocal state
                            gradients,stats=gradient(window)
                            lora,next_state,delta,did=apply_window_update(tx,gradients,state,pool.worker.lora,stats['kept_count'])
                            jax.block_until_ready((lora,next_state,delta))
                            if did:pool.call('set_lora',[dict(lora=lora) for _ in pool.workers])
                            state=next_state
                            return dict(did_update=bool(did),parameter_version=pool.parameter_version,**stats)
                        branch['windows'].append(stage(label+'_'+name+'_update',update_window))
                    branch['parameter_updates']=sum(x['did_update'] for x in branch['windows'])
                    if branch['parameter_updates']!=2:raise AssertionError('Expected two real optimizer transitions')
                    if anchor_before is not None:
                        if anchor_digest(pool.anchor_cache)!=anchor_before:raise AssertionError('Frozen B anchors changed')
                        branch['anchor_unchanged']=True
                    branch['final_lora_sha256']=digest(pool.worker.lora)
                    branch['final_optimizer_sha256']=digest(state)
                    branch['warm_AB_seconds']=sum(report['timings'][label+'_'+x+'_update'] for x in ('A','B'))
                    branch['reference_seconds']=report['timings'].get(label+'_reference_prepass',0.)
                    report['branches'][label]=branch
                    del state
                    gc.collect()
                for precision in ('bf16','mxfp8'):
                    if precision!='bf16':
                        pool.call('clear_replay')
                        reset_theta(precision,False)
                    pool.config['sampling']['rollout_mode']=precision
                    stage(precision+'_sampler_reload',lambda:pool._ensure_precision(precision))
                    warm_outputs=stage(precision+'_rollout_warmup_K4',lambda:pool.call('rollout',warm_payload))
                    if not common_warmed:
                        common('common_warmup',warm_outputs,False)
                        common_warmed=True
                    del warm_outputs
                    pool.call('clear_replay')
                    outputs=stage(precision+'_rollout_K8',lambda:pool.call('rollout',full_payload))
                    report[precision+'_rollout_contract']=outputs[0]['precision_contract']
                    rewards=common(precision,outputs,True)
                    endpoints=np.concatenate([x['endpoints'] for x in outputs])
                    groups=np.zeros(len(endpoints),np.int32)
                    stats=covariance(endpoints,rewards,groups,1,mode=c['art']['feedback'],group_size=8,seed=c['training']['seed'],eps=c['art']['eps'])
                    windows=prompt_windows(groups,c['sampling']['train_timesteps'],seed=c['training']['seed'])
                    payloads=paired_replay_payloads(pool,groups,stats,windows)
                    report[precision+'_reward']=dict(mean=float(rewards.mean()),std=float(rewards.std()),values=rewards.tolist())
                    train(precision,payloads,precision)
                    if precision=='mxfp8':
                        reset_theta('mxfp8_calibrated',True)
                        train('mxfp8_calibrated',payloads,precision)
                    del outputs,endpoints,groups,stats,payloads,rewards
                for label,branch in report['branches'].items():
                    origin='bf16' if label=='bf16' else 'mxfp8'
                    branch['warm_rollout_seconds']=report['timings'][origin+'_rollout_K8']
                    branch['warm_vae_reward_seconds']=report['timings'][origin+'_vae']+report['timings'][origin+'_reward']
                    branch['model_compute_seconds']=branch['warm_rollout_seconds']+branch['warm_AB_seconds']+branch['reference_seconds']
                    branch['reload_seconds']=sum(report['timings'][origin+suffix] for suffix in ('_sampler_reload','_dit_release','_bf16_reload'))
                    branch['model_plus_common_seconds']=branch['model_compute_seconds']+branch['warm_vae_reward_seconds']
                    branch['pipeline_seconds']=branch['model_plus_common_seconds']+branch['reload_seconds']
                baseline=report['branches']['bf16']
                for branch in report['branches'].values():
                    branch['rollout_speedup_vs_bf16']=baseline['warm_rollout_seconds']/branch['warm_rollout_seconds']
                    branch['model_compute_speedup_vs_bf16']=baseline['model_compute_seconds']/branch['model_compute_seconds']
                    branch['pipeline_speedup_vs_bf16']=baseline['pipeline_seconds']/branch['pipeline_seconds']
                report.update(status='completed',total_seconds=time.monotonic()-started)
                write_json(output/'report.json',report)
                print(json.dumps(dict(event='benchmark_complete',report=str(output/'report.json'),branches=report['branches'])),flush=True)
        except BaseException as exc:
            report.update(status='failed',error=f'{type(exc).__name__}: {exc}',total_seconds=time.monotonic()-started)
            write_json(output/'failed.json',report)
            raise

if __name__=='__main__':main()
