CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"


class _Metric:
    def __init__(self, *args, **kwargs):
        pass

    def labels(self, *args, **kwargs):
        return self

    def observe(self, *args, **kwargs):
        return None

    def inc(self, *args, **kwargs):
        return None

    def dec(self, *args, **kwargs):
        return None

    def set(self, *args, **kwargs):
        return None

    def time(self):
        metric = self

        class _Timer:
            def __enter__(self):
                return metric

            def __exit__(self, exc_type, exc, tb):
                return False

        return _Timer()


Histogram = _Metric
Counter = _Metric
Gauge = _Metric


def generate_latest(*args, **kwargs):
    return b""
