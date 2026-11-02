"""Baseline semantics: retain every fixed learned query state."""
class FixedQueryCompression:
    def __init__(self, num_queries): self.num_queries=num_queries
    def __call__(self, states): return states[:, :self.num_queries]
