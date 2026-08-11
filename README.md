# AI Knowledge Assistant

Lokalny system RAG do indeksowania dokumentów, wyszukiwania hybrydowego
i generowania odpowiedzi z cytowaniami przy użyciu Qdrant i Ollama.

## Obsługiwane corpusy

- `production` — prywatne dokumenty dodawane przez użytkownika,
- `v1` — podstawowy zestaw dokumentów i testów,
- `v2` — stress test obejmujący konflikty wersji, odmowy i prompt injection.

Każdy corpus ma osobny katalog dokumentów i kolekcję Qdrant. Pliki
ewaluacyjne są przypisane wyłącznie do v1 i v2. Corpus wybiera się
argumentem `--corpus` albo w interfejsie Streamlit.

## Najważniejsze polecenia

```powershell
.\.venv\Scripts\python.exe main.py index --corpus production --rebuild
.\.venv\Scripts\python.exe main.py ask --corpus production
.\.venv\Scripts\python.exe main.py index --corpus v2 --rebuild
.\.venv\Scripts\python.exe main.py ask --corpus v2
.\.venv\Scripts\python.exe main.py evaluate --corpus v2
.\.venv\Scripts\python.exe main.py evaluate-answers --corpus v2
.\.venv\Scripts\python.exe main.py release-check
.\.venv\Scripts\python.exe -m streamlit run app\interfaces\streamlit_app.py
python -m unittest discover -s tests -v
```

`release-check` kolejno testuje retrieval i odpowiedzi dla v1 oraz v2.
Wyniki porównuje z `data/eval/release_baseline_v1.json` i zapisuje raport
w `data/eval/release_check_results.json`. Kod zakończenia różny od zera
oznacza regresję albo błąd środowiska.

## Interfejs przeglądarkowy

Interfejs Streamlit pozwala korzystać z prywatnego corpusu `production`,
wybierać testowe corpusy v1/v2, prowadzić osobną historię rozmowy dla każdego
zestawu, wyświetlać źródła i bezpiecznie prezentować kontrolowane odmowy.

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

### Własne dokumenty

1. W panelu bocznym wybierz `MOJE DOKUMENTY`.
2. Dodaj pliki TXT, PDF lub DOCX. Limit pojedynczego pliku wynosi 20 MB.
3. Kliknij `Zapisz dokumenty`.
4. Kliknij `Przebuduj indeks`.
5. Po komunikacie o gotowym indeksie zadaj pytanie w oknie rozmowy.

Usunięcie dokumentu wymaga potwierdzenia. Po dodaniu albo usunięciu pliku
należy przebudować indeks. Jeżeli usunięto wszystkie dokumenty, przycisk
`Wyczyść pusty indeks` usuwa także ich stare fragmenty z Qdrant.

Pliki w `data/user_documents` oraz lokalna baza `data/qdrant` są ignorowane
przez Git. Prywatne dokumenty nie są wysyłane do repozytorium GitHub.

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

## Szybkie testy i CI — wersja 0.15.0

Szybkie testy jednostkowe nie uruchamiają Ollama, Qdrant ani modelu
embeddingowego. Sprawdzają kontrakt odpowiedzi, profile corpusów, reranking,
budowę kontekstu oraz zamrożony baseline jakości.

Uruchomienie lokalne:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q app main.py tests
```

Workflow `.github/workflows/ci.yml` uruchamia te same kontrole dla każdego
push do gałęzi `main` oraz dla pull requestów. Używa lekkiego pliku
`requirements-ci.txt`, dlatego nie pobiera modeli i nie wymaga sekretów.

Pełna bramka jakości nadal pozostaje lokalna:

```powershell
python main.py release-check
```

## Wersja 0.17.0 — prywatna baza dokumentów

Dodano odizolowany profil `production`, bezpieczny upload i usuwanie plików,
przebudowę indeksu z poziomu Streamlit, status Ollamy oraz ochronę prywatnych
dokumentów przed przypadkowym dodaniem do Git. Corpusy v1 i v2 nadal służą
wyłącznie jako zamrożone zestawy jakościowe.

## Wersja 0.18.0 — aplikacja Windows bez terminala

Gotowa paczka znajduje się w katalogu `release` jako
`KnowledgeAssistant-Windows-x64-v0.18.0.zip`. Użytkownik rozpakowuje cały ZIP
i uruchamia `KnowledgeAssistant.exe`; Python, PowerShell i PyCharm nie są
potrzebne. Plik EXE musi pozostać w tym samym katalogu co dołączony folder
`_internal`.

Ollama pozostaje lokalnym silnikiem generującym odpowiedzi. Jeżeli nie jest
zainstalowana, aplikacja pokazuje przycisk prowadzący do instalatora Windows.
Jeżeli brakuje modelu `gemma3:4b`, można pobrać go bezpośrednio z panelu
bocznego. Pierwsze pobranie wymaga połączenia z internetem i kilku GB wolnego
miejsca.

Przebieg dla użytkownika:

1. Rozpakuj cały ZIP.
2. Uruchom `KnowledgeAssistant.exe`.
3. Zainstaluj Ollamę albo pobierz model, jeżeli aplikacja o to poprosi.
4. Dodaj pliki TXT, PDF lub DOCX i kliknij `Zapisz dokumenty`.
5. Kliknij `Przebuduj indeks`, a następnie zadawaj pytania.
6. Zakończ program przyciskiem `Zamknij aplikację`.

Dokumenty, indeks Qdrant i logi są zapisywane w
`%LOCALAPPDATA%\KnowledgeAssistant`. Przy paczce znajduje się również prosty
plik `START_TUTAJ.txt`.

Budowanie paczki deweloperskiej:

```powershell
.\packaging\build_windows.ps1
```

## Wersja 0.19.0 — Full Offline

Wariant Full Offline zawiera aplikację, przenośną Ollamę `v0.32.8`, lokalny
model `gemma3:4b` oraz embeddingi. Nie wymaga instalowania Ollamy ani dostępu
do internetu. Funkcje chmurowe Ollamy są wymuszone wyłączone przez
`OLLAMA_NO_CLOUD=1`, a silnik działa na osobnym lokalnym porcie i jest zamykany
razem z aplikacją.

Artefakty lokalne:

- `dist/KnowledgeAssistant-Full-Offline` — gotowy folder, około 6,19 GB,
- `release/KnowledgeAssistant-Full-Offline-Windows-x64-v0.19.0.zip` — pełny
  ZIP do przeniesienia np. pendrivem, około 4,98 GB,
- `release/KnowledgeAssistant-Full-Offline-v0.19.0-parts` — trzy części
  poniżej limitu 2 GiB pojedynczego pliku GitHub Release.

Użytkownik pobierający części umieszcza je w jednym folderze i uruchamia
`SCAL_I_ROZPAKUJ.cmd`. Skrypt sprawdza SHA-256 każdej części, scala pełne
archiwum, ponownie sprawdza SHA-256 i rozpakowuje aplikację. Podczas tej
operacji zalecane jest co najmniej 20 GB wolnego miejsca.

Budowanie pełnego wariantu:

```powershell
.\packaging\build_full_offline.ps1 -SkipInstall
```

Skrypt pobiera oficjalny przenośny runtime Ollamy tylko przy pierwszym buildzie
i sprawdza jego ustaloną sumę SHA-256. Z lokalnego magazynu Ollamy eksportuje
wyłącznie manifest `gemma3:4b` oraz wskazane przez niego bloby. Do dystrybucji
dołączane są licencja MIT Ollamy, wymagany `GEMMA_NOTICE.txt` i kopia warunków
użycia Gemmy.
