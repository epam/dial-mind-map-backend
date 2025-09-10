class GeneratorException(Exception):
    """Base class for all exceptions raised by the generator module."""

    def __init__(self, message: str, details: str | None = None):
        """
        Initializes the GeneratorError.

        Args:
            message: The error message.
            details: Optional additional details about the error.
        """
        super().__init__(message)
        self.details = details


class EmptyDataException(GeneratorException):
    """Raised when essential data is empty or missing."""

    pass


class GenerationException(GeneratorException):
    """Raised for a generic error during the generation process."""

    pass


class ApplyException(GeneratorException):
    """Raised when an error occurs during the apply process."""

    pass


class GenPipeException(GeneratorException):
    """Base class for exceptions related to pipeline operations."""

    pass


class DelPipeException(GeneratorException):
    """Raised for an error within the deletion pipeline."""

    pass


class AddPipeException(GeneratorException):
    """Raised for an error within the addition pipeline."""

    pass
