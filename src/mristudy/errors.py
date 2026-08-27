class MRIStudyError(Exception):
    """Base error with a safe, user-facing message."""


class ApprovalError(MRIStudyError):
    pass


class StateError(MRIStudyError):
    pass


class PrivacyError(MRIStudyError):
    pass
