"""safe_prefetch.py: prefetch() that re-raises a loader failure in the consumer.

Found with a package whose token was sabotaged to 208: the upstream pysrc.data.prefetch
puts its end marker in the `finally` of the producer thread, so an exception while a micro batch is built ends the
iteration SILENTLY (loader_probe.py reported PASSED for 61 of 120 micro batches, rc 0); in training an epoch would end
short without any error.  This prefetch hands the exception to the consumer, which raises it.  install() replaces
pysrc.data.prefetch for every user of the module attribute (CollatedArticlesDataLoader.__iter__,
iter_micro_batches, hence pysrc.train.train, initialize_scales, fisher_alloc_a); rx7q/rx7r import prefetch from here.
"""
import queue
import threading


class _Failure:
    __slots__ = ("exc",)

    def __init__(self, exc):
        self.exc = exc


def prefetch(iterator, num_prefetch: int = 2):
    q: queue.Queue = queue.Queue(maxsize=num_prefetch)
    end = object()

    def producer():
        try:
            for item in iterator:
                q.put(item)
        except BaseException as e:  # noqa: BLE001  (handed to the consumer)
            q.put(_Failure(e))
            return
        q.put(end)

    threading.Thread(target=producer, daemon=True).start()
    while True:
        item = q.get()
        if item is end:
            return
        if isinstance(item, _Failure):
            raise RuntimeError("prefetch: the data loader failed while building a batch") from item.exc
        yield item


ORIGINAL = None   # the upstream prefetch, kept for the positive control (werkzeug/pk_prefetch.py)


def install():
    global ORIGINAL
    import pysrc.data as pd
    if getattr(pd.prefetch, "__module__", "") != __name__:
        ORIGINAL = pd.prefetch
        pd.prefetch = prefetch
    return pd.prefetch is prefetch
