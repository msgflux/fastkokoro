from __future__ import annotations

TEXT_CORPORA = {
    "tiny": [
        "Hello.",
        "Ola.",
        "Hi there.",
    ],
    "short": [
        "Hello, how are you?",
        "Ola, tudo bem?",
        "Good morning, thanks for joining us today.",
        "Bom dia, obrigado por participar deste teste agora.",
    ],
    "medium": [
        "Hello, how are you? This is a speech synthesis benchmark with "
        "punctuation, pauses, and more realistic phrasing.",
        "Ola, tudo bem? Este benchmark de sintese de voz usa frases com "
        "pontuacao, pausas e cadencia mais realista.",
        "We are measuring latency to the first chunk and the total generation "
        "time across varied sentence structures.",
        "Estamos medindo a latencia ate o primeiro chunk e o tempo total com "
        "estruturas de frase variadas e menos repetitivas.",
    ],
    "long": [
        "Hello, how are you? This benchmark is measuring latency to the first "
        "chunk and total generation time. It also mixes short and long clauses, "
        "commas, numbers like twenty four and ninety six, and more natural "
        "transitions to avoid masking the real runtime behavior.",
        "Ola, tudo bem? Este benchmark mede a latencia ate o primeiro chunk e o "
        "tempo total de geracao. Ele tambem mistura frases curtas e longas, "
        "virgulas, numeros como vinte e quatro e noventa e seis, e transicoes "
        "mais naturais para nao mascarar o comportamento real do runtime.",
        "For streaming in a terminal interface, the ideal is to deliver audio "
        "early without waiting for the entire text to be processed. Because of "
        "that, this corpus intentionally varies punctuation, cadence, and phrase "
        "boundaries between iterations.",
        "Para streaming em uma interface de terminal, o ideal e entregar audio "
        "cedo sem esperar o texto inteiro ser processado. Por isso, este corpus "
        "varia pontuacao, cadencia e limites de frase entre as iteracoes.",
    ],
}


def corpus_choices() -> list[str]:
    return list(TEXT_CORPORA.keys())


def get_text(corpus: str, iteration: int = 0) -> str:
    items = TEXT_CORPORA[corpus]
    return items[iteration % len(items)]


def get_texts(corpus: str) -> list[str]:
    return list(TEXT_CORPORA[corpus])
