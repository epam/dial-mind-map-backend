# Doc values
class DocType:
    LINK = "LINK"
    FILE = "FILE"


class DocCategories:
    LINK = "LINK"
    PPTX = "PPTX"
    HTML = "HTML"
    PDF = "PDF"
    TXT = "TXT"
    UNSUPPORTED = "UNSUPPORTED"

    PDF_AS_A_WHOLE = "PDF_AS_A_WHOLE"
    PPTX_AS_A_WHOLE = "PPTX_AS_A_WHOLE"
    HTML_AS_A_WHOLE = "HTML_AS_A_WHOLE"
    LINK_AS_A_WHOLE = "LINK_AS_A_WHOLE"
    TXT_AS_A_WHOLE = "TXT_AS_A_WHOLE"


class DocContentType:
    PRESENTATION = (
        "application/vnd.openxmlformats-officedocument."
        "presentationml.presentation"
    )
    HTML = "text/html"
    PDF = "application/pdf"
    TEXT = "text/plain"


# A short, two-page report analyzing developments in re/insurance markets and the global economy, covering topics like interest rates, inflation, and emerging risks.
DEFAULT_DOC_DESC = ""

HEADER_TO_SPLIT_ON = "Header 1"

MAX_CHUNK_SIZE = 2048
