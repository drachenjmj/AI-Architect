# Setup-Report & RAG-Erklärung

## Teil 1 – Struggles beim Setup (Kurzreport)

**1. Falsches Embedding-Modell (404 Not Found)**
Das ursprünglich vorgesehene Modell `text-embedding-004` war für den verwendeten API-Key gar nicht freigeschaltet. Lösung: Über `ListModels` die tatsächlich verfügbaren Modelle abfragen → Umstieg auf `gemini-embedding-001` bzw. später `gemini-embedding-2`.

**2. Free-Tier-Quota des API-Keys (429 Resource Exhausted)**
Der Key läuft im Free-Tier mit einem kontingentierten, stark umkämpften Limit (zunächst 100 Anfragen/Min, später 1000/Tag pro Modell). Ein einzelner Bulk-Aufbau scheiterte sofort an `429`. Lösung: chargenweises Einfügen mit automatischem Retry + Backoff statt eines einzigen `from_documents`-Aufrufs.

**3. Duplikate in der Vektordatenbank (642 statt 321 Vektoren)**
`Rag_Setup.ipynb` war nicht idempotent – jeder erneute Lauf hat die Chunks zusätzlich eingefügt. Lösung: vor dem Einfügen `delete_collection()` aufrufen.

**4. Windows-Dateisperre**
Der erste Idempotenz-Ansatz (`shutil.rmtree`) scheiterte an `PermissionError`, weil Chroma die Indexdatei `data_level0.bin` exklusiv sperrt. Lösung: Löschung über die Chroma-API (`delete_collection()`) statt über das Dateisystem.

**5. Transiente Netzwerkfehler (502 Bad Gateway / DNS-Ausfälle)**
Die Umgebung zeigte zeitweise instabiles Netzwerk. Lösung: Retry-Logik auf alle transienten Fehler erweitert (429, 5xx, DNS/Timeout).

**6. Doppelte Codepflege (Notebook vs. App)**
Logik existierte parallel in `agent.ipynb` und `app.py`. Lösung: Auslagerung in ein gemeinsames Modul `architect.py` (Single Source of Truth) – Änderungen nur noch an einer Stelle.

**7. Erschöpfung des Tageskontingents durch Tests**
Die vielen Rebuilds haben `gemini-embedding-001` für den Tag aufgebraucht (1000/Tag). Lösung: Wechsel auf `gemini-embedding-2` (separates Kontingent, gleiche Vektor-Dimension) – konsistent in Build- und Query-Richtung.

**8. Veraltete / Deprecation-Warnungen**
`google.generativeai` und `langchain-community` sind deprecated; `langchain-chroma` war nicht installiert. Rein kosmetisch, keine Funktionsbeeinträchtigung.

---

## Teil 2 – RAG einfach erklärt (für jemanden mit Coding-Basics)

### Die Grundidee in einem Satz
RAG (Retrieval-Augmented Generation) ist eine „Open-Book-Prüfung" für ein Sprachmodell – statt nur aus dem Gedächtnis zu antworten, darf es vorher in einem eigenen Wissensspeicher nachlesen.

### Warum braucht man das?
Ein normales LLM (wie Gemini) weiß viel, aber nicht deine firmenspezifischen Dokumente – und es kann nicht up-to-date sein. RAG holt genau die passenden Stellen aus deinen PDFs ins Modell, bevor es antwortet.

### Der Ablauf – wie eine Pipeline mit 5 Stufen

```
PDFs ──► 1. Zerkleinern ──► 2. Übersetzen ──► 3. Speichern
                                                    │
              5. Antworten ◄── 4. Suchen ◄──────────┘
```

1. **Zerkleinern (Chunking):** Ein langes PDF wird in kleine, handliche Stücke (~1000 Zeichen) geschnitten – wie ein Buch in Absätze.
2. **Übersetzen (Embedding):** Jedes Stück wird zu einem Vektor = einer Liste von Zahlen, die den Inhalt/Bedeutung erfasst. Ähnliche Texte bekommen ähnliche Zahlen – quasi GPS-Koordinaten im Bedeutungsraum.
3. **Speichern (Vektordatenbank / Chroma):** Alle Vektoren landen in einer Datenbank, die besonders schnell „ähnlichste Stücke finden" kann.
4. **Suchen (Retrieval):** Bei einer Frage wird die Frage selbst zu einem Vektor übersetzt – und die DB liefert die 3 ähnlichsten Textstücke zurück.
5. **Antworten (Generation):** Diese Stücke werden zusammen mit der Frage an Gemini geschickt: „Beantworte das auf Basis dieser Quellen." Das Modell formuliert die Antwort und kann die Quelle angeben.

### Die Bibliotheks-Metapher
Stell dir vor, du fragst einen Bibliothekar (Vektordatenbank): „Was ist gut an Microservices?" Er sucht die relevantesten Buchseiten heraus und gibt sie einem Experten (LLM). Der liest sie durch und erklärt es dir in eigenen Worten – mit Quellenangabe. Genau das macht `search_patterns()` + Gemini in unserem Code.

### Konkret in unserem Projekt
- Wissensbasis: 2 AWS-Architektur-PDFs → 321 Chunks → Chroma-DB (`./chroma_db`)
- Stellt der Nutzer eine Architektur-Frage, holt `search_patterns()` die Top-3-Passagen, und der Agent antwortet damit fundiert statt aus dem Bauch heraus.
