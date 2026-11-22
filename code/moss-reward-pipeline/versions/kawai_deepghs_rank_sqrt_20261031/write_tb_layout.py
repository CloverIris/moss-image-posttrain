#!/usr/bin/env python3
"""Write only TensorBoard dashboard layout metadata, never synthetic scalars."""
import argparse
import json
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    root=Path(config['logging']['log_dir']);root.mkdir(parents=True,exist_ok=True)
    from tensorboard.plugins.custom_scalar import layout_pb2,metadata
    from tensorboard.compat.proto import event_pb2,summary_pb2,tensor_pb2,types_pb2
    from tensorboard.summary.writer.record_writer import RecordWriter
    layout=layout_pb2.Layout()
    def category(title,charts):
        c=layout.category.add(title=title)
        for name,tags in charts:
            chart=c.chart.add(title=name)
            chart.multiline.tag.extend(tags)
    category('01 Reward overview',[
        ('Aesthetic and final reward',['reward_components/aesthetic_reward/mean','reward_components/final_reward/mean']),
        ('SDG artifact deduction',['reward_components/artifact_penalty/mean']),
        ('Within-prompt reward spread',['reward_components/final_reward/group_mad_mean','reward_components/final_reward/pooled_within_prompt_std']),
        ('SDG detection coverage',['reward_sdg/artifact_fraction','reward_sdg/artifact_observed_fraction'])])
    category('02 Actual weighted contributions',[
        ('Aesthetic contributions',[f'reward_components/contributions/{x}/mean' for x in ('kawai','deepghs','hps')]),
        ('Artifact contribution',['reward_components/contributions/sdg_artifact/mean']),
        ('Normalized within-prompt diversity',[f'reward_components/normalized_scores/{x}/group_mad_mean' for x in ('kawai','deepghs','hps')])])
    category('03 Base L2 magnitude',[
        ('Weighted losses',['train/art_loss','train/base_reference_loss','train/loss']),
        ('Raw base deviation',['train/base_reference_mse','train/base_reference_rms']),
        ('L2 share of objective',['train/base_reference_loss_fraction','train/base_reference_loss_to_art_ratio'])])
    category('04 Base L2 output-space competition',[
        ('Output cotangent norms',['train/art_output_grad_norm','train/base_reference_output_grad_norm','train/combined_output_grad_norm']),
        ('L2 to ART output-gradient ratio',['train/base_reference_output_grad_to_art_ratio']),
        ('Direction cosine',['train/base_reference_output_grad_cosine']),
        ('Cancellation',['train/base_reference_output_grad_cancellation_fraction'])])
    category('05 Diagnostic validity',[
        ('Defined flags',['train/base_reference_loss_ratio_defined','train/base_reference_output_grad_ratio_defined','train/base_reference_output_grad_cosine_defined']),
        ('Zero output-gradient fractions',['train/art_zero_output_grad_fraction','train/base_reference_zero_output_grad_fraction']),
        ('Collection axis mapping',['collection/source_update','collection/rollout_start'])])
    category('06 Logical batch',[
        ('Prompts and trajectories',['batch/prompts_per_collection','batch/trajectories']),
        ('Trajectory points and A/B window size',['batch/training_points_per_collection','batch/points_per_update_window']),
        ('Microbatch and devices',['batch/microbatch_per_device','batch/compute_devices']),
        ('Points per trajectory and update windows',['batch/training_points_per_trajectory','batch/update_windows'])])
    summary=summary_pb2.Summary(value=[summary_pb2.Summary.Value(
        tag=metadata.CONFIG_SUMMARY_TAG,metadata=metadata.create_summary_metadata(),
        tensor=tensor_pb2.TensorProto(dtype=types_pb2.DT_STRING,string_val=[layout.SerializeToString()]))])
    out=root/('events.out.tfevents.%s.reward_dashboard'%time.time_ns())
    with out.open('wb') as stream:
        writer=RecordWriter(stream)
        writer.write(event_pb2.Event(wall_time=time.time(),file_version='brain.Event:2').SerializeToString())
        writer.write(event_pb2.Event(wall_time=time.time(),step=0,summary=summary).SerializeToString());writer.flush()
    print(json.dumps(dict(layout=str(out),categories=len(layout.category),synthetic_scalar_points=0)))


if __name__=='__main__':main()
