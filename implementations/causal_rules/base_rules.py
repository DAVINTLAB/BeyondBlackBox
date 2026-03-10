from templates.CausalRules import CausalRules

class base_rules(CausalRules):
    def __init__(self, *args):
        pass

    def apply_rules_tracker(self, *args):
        from .tracker.base_tracker_rules import apply_rules as base_tracker_rules
        return base_tracker_rules(*args)
    
    def apply_rules_estimator(self, *args):
        from .estimator.base_estimator_rules import apply_rules as base_estimator_rules
        return base_estimator_rules(*args)
