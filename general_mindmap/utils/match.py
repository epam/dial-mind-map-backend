import os
from typing import Dict, List, Optional, Sequence

import numpy
import torch
from langchain.schema import Document
from langchain_core.runnables import chain
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from xgboost import XGBClassifier

from general_mindmap.models.attachment import graph_to_attach
from general_mindmap.models.graph import Graph, GraphData

E5_EMBEDDINGS_MODEL_NAME_OR_PATH = os.environ.get(
    "E5_EMBEDDINGS_MODEL_PATH", "intfloat/e5-small-v2"
)

embedding_model = SentenceTransformer(E5_EMBEDDINGS_MODEL_NAME_OR_PATH)

match_model = XGBClassifier(
    n_estimators=100, max_depth=2, learning_rate=1, objective="binary:logistic"
)
match_model.load_model("models/match.model")


def calculate_embedding(input: str) -> numpy.ndarray:
    embedding = embedding_model.encode(input)
    assert isinstance(embedding, numpy.ndarray)

    result = embedding[None, :]
    assert isinstance(result, numpy.ndarray)
    return result


def calculate_token_embedding(input: str) -> numpy.ndarray:
    embedding = embedding_model.encode(input, output_value="token_embeddings")
    assert isinstance(embedding, torch.Tensor)

    return embedding.cpu().numpy()


def get_sorted_node_triggers(
    question_embedding: numpy.ndarray, triggers: List[str]
) -> List[Dict]:
    triggers_with_similarity = []
    for trigger in triggers:
        trigger_embedding = calculate_embedding(trigger)

        similarity = cosine_similarity(trigger_embedding, question_embedding)[
            0
        ][0]

        triggers_with_similarity.append(
            {
                "text": trigger,
                "similarity": similarity,
            }
        )
    triggers_with_similarity.sort(
        key=lambda trig: trig["similarity"], reverse=True
    )
    return triggers_with_similarity


def get_sorted_best_triggers_for_each_node(
    question_embedding: numpy.ndarray, nodes: Sequence[Document]
) -> List[Dict]:
    best_triggers = []
    for node in nodes:
        if "question" in node.metadata:
            if (
                "questions" not in node.metadata
                or not node.metadata["questions"]
            ):
                node.metadata["questions"] = [node.metadata["question"]]
            del node.metadata["question"]

        triggers = get_sorted_node_triggers(
            question_embedding, node.metadata["questions"]
        )
        best_triggers.append(triggers[0] | {"node": node})
    best_triggers.sort(key=lambda trigger: trigger["similarity"], reverse=True)
    return best_triggers


def calculate_features(question: str, best_triggers: List[Dict]) -> List[float]:
    question_token_embeddings = calculate_token_embedding(question)
    trigger_token_embeddings = calculate_token_embedding(
        best_triggers[0]["text"]
    )
    similarity = cosine_similarity(
        question_token_embeddings, trigger_token_embeddings
    )
    features = [
        best_triggers[0]["similarity"],
        (
            best_triggers[0]["similarity"] - best_triggers[1]["similarity"]
            if len(best_triggers) > 1
            else 1.0
        ),
        similarity.max(0).min(),
        similarity.max(-1).min(),
        similarity.max(0).mean(),
        similarity.max(-1).mean(),
    ]
    return features


async def find_matched_node(input: Dict) -> Optional[Document]:
    question = input["question"]
    nodes: Sequence[Document] = input["cards"]

    if not len(nodes):
        return None

    for node in nodes:
        if node.metadata["question"] == question:
            return node

    # question_embedding = calculate_embedding(question)

    # best_triggers = get_sorted_best_triggers_for_each_node(
    #     question_embedding, nodes
    # )

    # features = calculate_features(question, best_triggers)

    # if match_model.predict([features])[0] == 1:
    #     return best_triggers[0]["node"]
    # else:
    #     return None


def create_matched_node_to_graph_attachment_chain(graph_data: Graph):
    @chain
    def matched_node_to_graph_attachment(node_doc: Document) -> Dict:
        return graph_to_attach(
            Graph(
                root=[
                    GraphData(
                        data=graph_data.get_node_by_id(node_doc.metadata["id"])
                    )
                ]
            ),
            "Founded graph node",
        )

    return matched_node_to_graph_attachment
