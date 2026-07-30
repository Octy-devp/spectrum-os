class OSC:
    """Spectrum OS API entry point. All operations accessed via `osc.<op>()`."""

    @property
    def sectors(self):
        from . import sector
        return sector

    @property
    def wave(self):
        from . import wave_ops
        return wave_ops

    @property
    def template(self):
        from . import template
        return template

    @property
    def branch(self):
        from . import branch
        return branch

    @property
    def cluster(self):
        from . import cluster
        return cluster

    @property
    def verify(self):
        from . import verify
        return verify

    @property
    def quantum(self):
        from .. import quantum
        return quantum

    @property
    def contracts(self):
        from .. import contracts
        return contracts

    @property
    def sources(self):
        from .. import sources
        return sources


osc = OSC()
