# AI Knowledge Assistant

Lokalny system RAG do indeksowania dokumentów, wyszukiwania hybrydowego
i generowania odpowiedzi z cytowaniami przy użyciu Qdrant i Ollama.

## Obsługiwane corpusy

- `v1` — podstawowy zestaw dokumentów i testów,
- `v2` — stress test obejmujący konflikty wersji, odmowy i prompt injection.

Każdy corpus ma osobny katalog dokumentów, kolekcję Qdrant i pliki
ewaluacyjne. Corpus wybiera się argumentem `--corpus`.

## Najważniejsze polecenia

```powershell
.\.venv\Scripts\python.exe main.py index --corpus v2 --rebuild
.\.venv\Scripts\python.exe main.py ask --corpus v2
.\.venv\Scripts\python.exe main.py evaluate --corpus v2
.\.venv\Scripts\python.exe main.py evaluate-answers --corpus v2
.\.venv\Scripts\python.exe main.py release-check
.\.venv\Scripts\python.exe -m streamlit run app\interfaces\streamlit_app.py
```

`release-check` kolejno testuje retrieval i odpowiedzi dla v1 oraz v2.
Wyniki porównuje z `data/eval/release_baseline_v1.json` i zapisuje raport
w `data/eval/release_check_results.json`. Kod zakończenia różny od zera
oznacza regresję albo błąd środowiska.

## Interfejs przeglądarkowy

Interfejs Streamlit pozwala wybierać corpus v1/v2, prowadzić osobną historię
rozmowy dla każdego zestawu, wyświetlać źródła i bezpiecznie prezentować
kontrolowane odmowy.

Przed pierwszym uruchomieniem zainstaluj zależności:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Następnie uruchom aplikację:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\interfaces\streamlit_app.py
```

Przeglądarka powinna otworzyć adres `http://localhost:8501`. Zatrzymanie
aplikacji: `Ctrl+C` w oknie PowerShell.

## Ważne ograniczenie lokalnego Qdrant

Nie uruchamiaj jednocześnie dwóch poleceń korzystających z Qdrant.
Przed ewaluacją zakończ tryb `ask`, wpisując `q` i naciskając Enter.

## Zamrożony baseline 0.13.0

- v1 retrieval: 9 przypadków,
- v1 answers: 6 przypadków,
- v2 retrieval: 30 przypadków,
- v2 answers: 35 przypadków,
- model: `gemma3:4b`,
- wymagany Recall@1 i MRR: 100%,
- wymagany Phrase Recall@1 dla v2: minimum 90%,
- wymagane odpowiedzi ugruntowane i cytowania: 100%,
- dopuszczalne halucynacje i błędy wykonania: 0%.

## Wersja 0.14.0

Dodano lokalny interfejs Streamlit z wyborem corpusu, historią rozmowy,
cytowaniami, statusem indeksu i komunikatami dla blokady Qdrant/Ollama.
