"""
    Query parameters
"""

# pylint: disable=C0301,C0103,C0303,C0411,W1203,C0412

from dataclasses import dataclass
from dataclasses_json import dataclass_json

from dial_rag.retrievers.description_retriever.image_details import ImageDetails
from dial_rag.retrievers.description_retriever.answer_details import AnswerDetails

@dataclass_json
@dataclass
class QueryParams:
    """
        QueryParams class
    """
    image_max_size  : int = 800
    answer_details : AnswerDetails = AnswerDetails.DETAILED
    answer_context  : str = ""
    image_detailed_quality : ImageDetails = ImageDetails.AUTO
    max_top_results_image : int = -1
    max_top_results_text : int = -1
    date_information : bool = True
    
