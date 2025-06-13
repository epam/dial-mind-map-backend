class BaseGeneratorException(Exception):
    """Exception raised for errors in Generator process."""

    def __init__(self, msg, details=None):
        super().__init__(msg)
        self.msg = msg
        self.details = details


class GenerationException(BaseGeneratorException):
    """Exception raised for errors in generation process."""


class ApplyException(BaseGeneratorException):
    """Exception raised for errors in apply process."""


class GenPipeException(BaseGeneratorException):
    """Exception raised for errors in generation pipeline."""


class DelPipeException(BaseGeneratorException):
    """Exception raised for errors in del pipeline."""


class AddPipeException(BaseGeneratorException):
    """Exception raised for errors in add pipeline."""
