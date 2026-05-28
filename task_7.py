import json
import os

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


os.environ["TOKENIZERS_PARALLELISM"] = "false"

tickets = [
    "I really like this software!",
    (
        "I received a recommendation to take a left turn, but the navigation system "
        "showed a 'No Left Turn' sign in the photo. Such discrepancy is not acceptable!"
    ),
    (
        "The photo analysis for speed limit signs is inconsistent, sometimes showing "
        "an older, incorrect limit after a recent road change. Is there a way to "
        "update this data faster?"
    ),
    "What are the plans for night vision integration?",
]

model_name = "typeform/distilbert-base-uncased-mnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

urgency_pipeline = pipeline(
    "zero-shot-classification",
    model=model,
    tokenizer=tokenizer,
)
sentiment_pipeline = pipeline(
    "zero-shot-classification",
    model=model,
    tokenizer=tokenizer,
)
topic_pipeline = pipeline(
    "zero-shot-classification",
    model=model,
    tokenizer=tokenizer,
)

urgency_candidate_labels = ["low", "medium", "high"]
sentiment_candidate_labels = ["positive", "neutral", "negative"]
topic_candidate_labels = [
    "general praise",
    "navigation sign discrepancy",
    "speed limit data update",
    "future feature question",
]


def classify_ticket(classifier, ticket, candidate_labels):
    result = classifier(ticket, candidate_labels=candidate_labels)
    return {
        "label": result["labels"][0],
        "score": float(result["scores"][0]),
        "labels": result["labels"],
        "scores": [float(score) for score in result["scores"]],
    }


urgency_results = [
    classify_ticket(urgency_pipeline, ticket, urgency_candidate_labels)
    for ticket in tickets
]
sentiment_results = [
    classify_ticket(sentiment_pipeline, ticket, sentiment_candidate_labels)
    for ticket in tickets
]
topic_results = [
    classify_ticket(topic_pipeline, ticket, topic_candidate_labels)
    for ticket in tickets
]

for index, ticket in enumerate(tickets, start=1):
    print(f"ticket_{index}:", ticket)
    print("urgency:", urgency_results[index - 1])
    print("sentiment:", sentiment_results[index - 1])
    print("topic:", topic_results[index - 1])

your_answer_as_string = (
    "Wybrałam zero-shot-classification dla pilności, sentymentu i tematu, ponieważ "
    "nie mamy oznaczonego zbioru treningowego dla tych konkretnych kategorii. "
    "Pipeline zero-shot pozwala podać własne etykiety kandydackie dla każdego "
    "zgłoszenia bez dodatkowego trenowania modelu. Text-classification wybrałabym "
    "wtedy, gdy dostępny byłby model już wytrenowany dokładnie na naszych etykietach "
    "lub gdy mielibyśmy dane do dostrojenia takiego modelu."
)

urgency_sen1 = urgency_results[0]
urgency_sen2 = urgency_results[1]
urgency_sen3 = urgency_results[2]
urgency_sen4 = urgency_results[3]

output = {
    "urgency_candidate_labels": urgency_candidate_labels,
    "urgency_sen1": urgency_sen1,
    "urgency_sen2": urgency_sen2,
    "urgency_sen3": urgency_sen3,
    "urgency_sen4": urgency_sen4,
    "sentiment_candidate_labels": sentiment_candidate_labels,
    "sentiment_results": sentiment_results,
    "topic_candidate_labels": topic_candidate_labels,
    "topic_results": topic_results,
    "choice_reasons": your_answer_as_string,
}

with open("task_7.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
