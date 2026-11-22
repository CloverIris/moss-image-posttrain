"""CPU-only diagnostics for frozen-base MSE; never changes an objective."""
import math


def reference_strength_metrics(art_loss, reference_mse, beta, denominator, output_grad_sums=None):
    if denominator <= 0 or any(not math.isfinite(float(v)) or float(v) < 0 for v in (art_loss, reference_mse, beta)):
        raise ValueError('Expected finite nonnegative losses/beta and positive denominator')
    weighted = beta*reference_mse
    total = art_loss+weighted
    metrics = dict(base_reference_loss_fraction=weighted/total if total > 0 else 0.,
                   base_reference_loss_ratio_defined=float(art_loss > 0),
                   base_reference_only_loss=float(art_loss == 0 and weighted > 0))
    if art_loss > 0:
        metrics['base_reference_loss_to_art_ratio'] = weighted/art_loss
    if output_grad_sums is None:
        return metrics
    policy_sq = max(float(output_grad_sums['policy_sq']), 0.)
    reference_sq = max(float(output_grad_sums['reference_sq']), 0.)
    dot = float(output_grad_sums['dot'])
    if any(not math.isfinite(x) for x in (policy_sq, reference_sq, dot)):
        raise ValueError('Nonfinite output cotangent statistics')
    policy = math.sqrt(policy_sq)
    reference = math.sqrt(reference_sq)
    combined = math.sqrt(max(policy_sq+reference_sq+2*dot, 0.))
    metrics.update(art_output_grad_norm=policy/denominator,
                   base_reference_output_grad_norm=reference/denominator,
                   combined_output_grad_norm=combined/denominator,
                   base_reference_output_grad_ratio_defined=float(policy > 0),
                   base_reference_output_grad_cosine_defined=float(policy > 0 and reference > 0),
                   art_zero_output_grad_fraction=output_grad_sums['policy_zero_count']/denominator,
                   base_reference_zero_output_grad_fraction=output_grad_sums['reference_zero_count']/denominator,
                   base_reference_only_output_grad=float(policy == 0 and reference > 0),
                   base_reference_output_grad_cancellation_fraction=max(0.,min(1.,1-combined/(policy+reference))) if policy+reference > 0 else 0.)
    if policy > 0:
        metrics['base_reference_output_grad_to_art_ratio'] = reference/policy
    if policy > 0 and reference > 0:
        cosine = max(-1.,min(1.,dot/(policy*reference)))
        metrics['base_reference_output_grad_cosine'] = cosine
        metrics['base_reference_output_grad_opposes_art'] = float(cosine < 0)
    return metrics
