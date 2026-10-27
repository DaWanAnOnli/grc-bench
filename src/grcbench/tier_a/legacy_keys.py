"""Summary keys of Tier-A smoke runs stored before pre-registration §8 decision 1, when the
grammar-constrained server default was called B2 and the unconstrained setting the B0 candidate.
smoke.load_summary_dict reads such a summary with the current keys."""

OLD_SUMMARY_KEYS = {"B2": "B0", "B0_candidate": "unconstrained_candidate",
                    "B0_request_candidate": "unconstrained_request_candidate"}
OLD_SUMMARY_VALUES = {"--jinja (the B2 server)": "--jinja (the B0 server)"}
