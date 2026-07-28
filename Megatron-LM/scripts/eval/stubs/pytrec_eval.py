"""Import-only stub for OLMES tasks that do not use HELMET retrieval metrics.

The official OLMES task registry imports HELMET eagerly.  The OLMo/TULU tasks
prepared by this project never instantiate ``RelevanceEvaluator``; fail loudly
if that assumption changes instead of silently returning a fake metric.
"""


class RelevanceEvaluator:
    def __init__(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError(
            "pytrec_eval is unavailable in the request-preparation environment; "
            "HELMET tasks are outside this frozen evaluation suite."
        )
