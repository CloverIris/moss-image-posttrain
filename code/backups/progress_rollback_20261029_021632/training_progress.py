"""CPU-only provisional progress records; never publishes training metrics."""


def window_batch_counts(buckets, microbatch_size):
    """Count the already-built buckets without materializing replay tensors."""
    keys=sorted({key for bucket in buckets for key in bucket})
    return {key:max((len(bucket.get(key,[]))+microbatch_size-1)//microbatch_size
                    for bucket in buckets) for key in keys}


def microbatch_progress(window, bucket, completed, total, bucket_completed,
                        bucket_total, elapsed, base_reference_seconds, real_items):
    return dict(event='training_microbatch_progress',window=int(window),bucket=list(bucket),
                completed_microbatches=int(completed),total_microbatches=int(total),
                completed_bucket_microbatches=int(bucket_completed),total_bucket_microbatches=int(bucket_total),
                elapsed_seconds=float(elapsed),base_reference_seconds=float(base_reference_seconds),
                processed_valid_items=int(real_items),committed=False,provisional=True)


def window_finished(event):
    """A log record only; the official event stays pending until checkpoint."""
    fields=('update','rollout','window','optimizer_updates','optimizer_step_applied',
            'gradient_update_seconds','items','loss','art_loss','base_reference_mse',
            'base_reference_loss','base_reference_loss_to_art_ratio',
            'base_reference_output_grad_to_art_ratio','base_reference_output_grad_cosine')
    record={key:event[key] for key in fields if key in event}
    return dict(record,event='window_finished',committed=False,provisional=True,
                publication='Official metrics and TensorBoard wait for the completed A/B checkpoint')
