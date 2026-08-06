from loguru import logger

from app.generation.answer_service import AnswerService
from app.retrieval.retriever import Retriever


EXIT_COMMANDS = {
    "q",
    "quit",
    "exit",
    "koniec",
}


def run_retrieval_debug(
    retriever: Retriever,
) -> None:
    print()
    print("=" * 70)
    print("RETRIEVAL DEBUG")
    print("Wpisz pytanie do dokumentów.")
    print("Aby zakończyć, wpisz: q")
    print("=" * 70)

    while True:
        query = input("\nPytanie: ").strip()

        if query.lower() in EXIT_COMMANDS:
            logger.info("Retrieval debug session finished.")
            break

        if not query:
            print("Pytanie nie może być puste.")
            continue

        try:
            results = retriever.retrieve(query)
        except Exception:
            logger.exception("Retrieval request failed.")
            print("Wystąpił błąd podczas wyszukiwania.")
            continue

        if not results:
            print("\nNie znaleziono żadnych wyników.")
            continue

        print(f"\nZnalezione wyniki: {len(results)}")

        for position, result in enumerate(results, start=1):
            page_display = (
                result.page_number
                if result.page_number is not None
                else "brak"
            )

            unit_display = (
                result.unit_number
                if result.unit_number is not None
                else "brak"
            )

            text_preview = (
                result.text[:600]
                .replace("\n", " ")
                .strip()
            )

            print()
            print("-" * 70)
            print(f"Wynik #{position}")
            print(f"Score: {result.score:.4f}")
            print(f"Dokument: {result.document_name}")
            print(f"Typ: {result.file_type}")
            print(f"Strona: {page_display}")
            print(f"Jednostka: {unit_display}")
            print(f"Chunk index: {result.chunk_index}")
            print(f"Chunk ID: {result.chunk_id}")
            print(f"Źródło: {result.source_path}")
            print(f"Tekst: {text_preview}")


def run_answer_debug(
    answer_service: AnswerService,
) -> None:
    print()
    print("=" * 70)
    print("KNOWLEDGE ASSISTANT — ANSWER MODE")
    print("Zadaj pytanie do dokumentów.")
    print("Aby zakończyć, wpisz: q")
    print("=" * 70)

    while True:
        question = input("\nPytanie: ").strip()

        if question.lower() in EXIT_COMMANDS:
            logger.info("Answer session finished.")
            break

        if not question:
            print("Pytanie nie może być puste.")
            continue

        try:
            generated_answer = answer_service.answer(
                question
            )

        except Exception as error:
            logger.exception(
                "Answer generation failed."
            )

            print(f"\nWystąpił błąd: {error}")
            continue

        print()
        print("ODPOWIEDŹ:")
        print(generated_answer.answer)

        if generated_answer.citations:
            print()
            print("ŹRÓDŁA:")

            for citation in generated_answer.citations:
                page_display = (
                    citation.page_number
                    if citation.page_number is not None
                    else "brak"
                )

                print()
                print(
                    f"[{citation.source_id}] "
                    f"{citation.document_name}"
                )
                print(f"Strona: {page_display}")
                print(f"Chunk ID: {citation.chunk_id}")
                print(f"Score: {citation.score:.4f}")
                print(f"Fragment: {citation.excerpt}")