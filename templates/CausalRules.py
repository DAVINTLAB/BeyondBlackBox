# Wrapper class for applying rules to the detections to achieve the final decisions
class CausalRules:
    def __init__(self, args):
        pass

    def apply_rules_detector(self, *args):
        raise NotImplementedError("Method not implemented")

    def apply_rules_tracker(self, *args):
        raise NotImplementedError("Method not implemented")

    def apply_rules_estimator(self, *args):
        raise NotImplementedError("Method not implemented")
