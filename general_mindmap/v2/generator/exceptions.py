from typing import Any


class AnswerCountMismatchError(Exception):
    """
    Custom exception for when LLM answer count doesn't match question
    count.
    """

    def __init__(
        self, n_questions: int, n_answers: int, raw_output: Any = None
    ):
        self.n_questions = n_questions
        self.n_answers = n_answers
        self.raw_output = raw_output
        super().__init__(
            f"Mismatch in question/answer count. Sent {n_questions} "
            f"questions, received {n_answers} answers."
        )
