#!/usr/bin/env python3
"""Strict CPU-only canonical checkpoint to atomic, audited LoRA export."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import uuid

from checkpoint_eval_service import (DirectoryLock, atomic_json, canonical,
    committed_manifest, digest_bytes, digest_file, immutable_json, read_json)


def canonical_api(repo):
    """Use the repository's own checkpoint identity, validator and LoRA export."""
    sys.path.insert(0, str(repo))
    from moss_art.config import load_config
    from moss_art.trainer import identity_for
    from moss_art.checkpoint import load_checkpoint
    from moss_art.lora import export_safetensors
    return SimpleNamespace(load_config=load_config, identity_for=identity_for,
        load_checkpoint=load_checkpoint, export_safetensors=export_safetensors)


def resolve_job(raw):
    job = dict(raw)
    for key in ('repo', 'source_config', 'checkpoint', 'adapter', 'checkpoint_manifest_sha256'):
        if not isinstance(job.get(key), str) or not job[key]:
            raise ValueError('Missing exporter job field: '+key)
    for key in ('repo', 'source_config', 'checkpoint', 'adapter'):
        value = Path(job[key]).expanduser()
        if not value.is_absolute(): raise ValueError(key+' must be absolute')
        if key in ('checkpoint','adapter') and value.is_symlink():
            raise ValueError(key+' must not be symlinked')
        job[key] = str(value.resolve())
    checkpoint = Path(job['checkpoint'])
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ValueError('Checkpoint must be a real directory')
    for name in ('manifest.json', 'COMMITTED', 'arrays.npz'):
        if not (checkpoint/name).is_file() or (checkpoint/name).is_symlink():
            raise ValueError('Missing or symlinked checkpoint file: '+name)
    if not Path(job['repo']).is_dir() or not Path(job['source_config']).is_file():
        raise ValueError('Canonical repository/source config is unavailable')
    if Path(job['adapter']).is_symlink(): raise ValueError('Adapter must not be symlinked')
    return job


def checked_sources(job):
    checksum, manifest = committed_manifest(job['checkpoint'])
    if checksum != job['checkpoint_manifest_sha256']:
        raise ValueError('JOB checkpoint_manifest_sha256 does not match COMMITTED')
    source_hash = digest_file(job['source_config'])
    expected_source = job.get('source_config_sha256')
    if expected_source is not None and source_hash != expected_source:
        raise ValueError('JOB source config SHA256 mismatch')
    payload_hash = digest_file(Path(job['checkpoint'])/'arrays.npz')
    if payload_hash != manifest.get('payload_sha256'):
        raise ValueError('Checkpoint payload checksum mismatch')
    return checksum, source_hash, payload_hash


def verify_export(adapter, audit, contract):
    if audit.get('contract') != contract or audit.get('contract_sha256') != digest_bytes(canonical(contract)):
        raise ValueError('Existing adapter export belongs to a different checkpoint/config identity')
    if not adapter.is_file() or adapter.is_symlink() or adapter.stat().st_size != audit['adapter_size'] \
            or digest_file(adapter) != audit['adapter_sha256']:
        raise ValueError('Existing exported adapter is missing, changed or corrupt')
    return audit


def run(raw, *, api=None):
    # Set these before importing JAX through the canonical repository. Do not
    # alter the source config, the checkpoint identity, CUDA installation or XLA.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['JAX_PLATFORMS'] = 'cpu'
    job = resolve_job(raw)
    adapter = Path(job['adapter']); adapter.parent.mkdir(parents=True, exist_ok=True)
    audit_path = adapter.with_name(adapter.name+'.export.json')
    pending_path = adapter.with_name(adapter.name+'.export.pending.json')
    with DirectoryLock(adapter.with_name(adapter.name+'.export.lock')):
        sources = checked_sources(job)
        api = canonical_api(job['repo']) if api is None else api
        config = api.load_config(job['source_config'])
        expected_metadata = api.identity_for(config)
        alpha = config['lora']['alpha']
        contract = dict(version=1, checkpoint=job['checkpoint'],
            checkpoint_id=job.get('checkpoint_id',Path(job['checkpoint']).name),
            checkpoint_manifest_sha256=sources[0], source_config=job['source_config'],
            source_config_sha256=sources[1], checkpoint_payload_sha256=sources[2],
            repo=job['repo'], expected_metadata_sha256=digest_bytes(canonical(expected_metadata)),
            alpha=alpha, style='diffusion_model', adapter=str(adapter))
        if audit_path.is_file():
            return verify_export(adapter, read_json(audit_path), contract)
        if pending_path.is_file():
            pending = read_json(pending_path)
            if pending.get('contract') != contract:
                raise ValueError('Unfinished export belongs to another checkpoint/config')
            if not adapter.exists():
                staged = Path(pending['staged_adapter'])
                verify_export(staged, pending, contract)
                staged.replace(adapter)
            verify_export(adapter, pending, contract)
            immutable_json(audit_path, pending)
            return pending
        if adapter.exists():
            raise ValueError('Refusing to replace an unaudited pre-existing adapter')
        params, optimizer, metadata = api.load_checkpoint(job['checkpoint'], expected_metadata=expected_metadata)
        del optimizer
        gc.collect()
        staged = adapter.with_name('.'+adapter.name+'.'+uuid.uuid4().hex+'.staging')
        # Canonical signature verified in moss_art/lora.py:
        # export_safetensors(path, adapters, *, alpha, metadata=None, style=...).
        api.export_safetensors(staged, params, alpha=alpha, style='diffusion_model', metadata=dict(
            checkpoint_id=contract['checkpoint_id'], checkpoint_manifest_sha256=sources[0],
            source_config_sha256=sources[1], checkpoint_payload_sha256=sources[2],
            source_step=job.get('source_step', metadata.get('update'))))
        del params
        gc.collect()
        if not staged.is_file() or staged.stat().st_size == 0:
            raise ValueError('Canonical export produced an empty adapter')
        if checked_sources(job) != sources:
            raise ValueError('Source checkpoint/config changed during export')
        # Pin the publication proof before the final rename. A crash after rename
        # resumes the SAME bytes and never silently replaces an existing adapter.
        audit = dict(version=1, contract=contract, contract_sha256=digest_bytes(canonical(contract)),
            expected_metadata=expected_metadata, checkpoint_metadata=metadata,
            adapter_sha256=digest_file(staged), adapter_size=staged.stat().st_size,
            staged_adapter=str(staged), created_at=time.time(), cpu_only=True,
            export_function='moss_art.lora.export_safetensors',
            strict_load='moss_art.checkpoint.load_checkpoint(expected_metadata=identity_for(source_config))')
        immutable_json(pending_path,audit)
        staged.replace(adapter)
        # Flush the output before advertising its completed audit.
        with adapter.open('r+b') as stream: os.fsync(stream.fileno())
        immutable_json(audit_path,audit)
        return audit


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    args=parser.parse_args(argv)
    result=run(read_json(args.config))
    print(json.dumps(dict(event='checkpoint_adapter_exported',
        checkpoint_id=result['contract']['checkpoint_id'], adapter=result['contract']['adapter'],
        adapter_sha256=result['adapter_sha256'], cpu_only=True)),flush=True)


if __name__=='__main__': main()
