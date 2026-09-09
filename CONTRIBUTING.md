# Mitwirken

Das Projekt ist in der Pre-Alpha-Phase: Layer 1 steht, Layer 2 und 3 sind
spezifiziert, aber nicht geschrieben. Am hilfreichsten sind derzeit:

- **Gegenargumente zum Zuschnitt.** Wenn Sie ein Modell schon einmal über
  gewöhnliche Rechner verteilt haben und wissen, woran es scheitert, ist das
  wertvoller als Code.
- **Hinweise auf übersehene Vorarbeiten.** Siehe `docs/prior-art.md`. Wenn dort
  etwas fehlt, bitte ein Issue.
- **Berichte von echter Hardware.** Gemischte GPUs, Consumer-Anschlüsse, NAT auf
  beiden Seiten. Genau dieser Fall lässt sich bei uns nicht vollständig
  nachstellen.

## Worum es geht

`swarmpipe` — die Maschinerie, um ein Modell über mehrere Rechner zu verteilen
und die Tensoren dazwischen zu bewegen, herausgelöst aus dem Projekt
SilentSwarm. Eine Bibliothek, keine Plattform: kein Scheduler, keine
Steuerungsebene, keine Oberfläche.

Vor der ersten Änderung [docs/design.md](docs/design.md) lesen, dann
[docs/requirements.md](docs/requirements.md) — die zweite Hälfte dieser Seite ist
eine Liste von Dingen, die bereits einmal schiefgegangen sind.

## Einrichtung

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,torch,lora]"     # GPU-Maschine, oder CPU-torch für Tests
pytest -q                              # alles
pytest -m "not torch" -q               # nur das, was ohne DL-Framework laufen muss
ruff check .
```

Für Arbeiten an der Wire- oder Transportschicht bitte eine **zweite Umgebung ohne
installiertes torch** verwenden — nur so fällt auf, wenn der torch-freie Vertrag
gebrochen wurde:

```bash
pip install -e ".[dev]"
pytest -m "not torch" -q
```

## Aufbau

| Pfad | Was dort liegt |
|---|---|
| `src/swarmpipe/split/` | L1. Sieht Modelle. `spec.py`/`api.py`/`plan.py` sind torch-frei; `torch/` ist das Backend. |
| `src/swarmpipe/wire/` | L2. Sieht Tensor-Frames. **Torch-frei per Regel.** Noch nicht implementiert (T2). |
| `src/swarmpipe/link/` | L3. Sieht `list[bytes]`. **Torch-frei per Regel.** Noch nicht implementiert (T1). |
| `tests/unit/` | Spiegelt `src/swarmpipe/`. |
| `seed/port/` | Code aus dem Ursprungsprojekt, wartet auf die Portierung in T0. Noch nicht importierbar — siehe [docs/porting-guide.md](docs/porting-guide.md). |
| `seed/origin/` | Der ursprüngliche Extraktionsplan, wörtlich. Nur zur Herkunft; der gültige Plan ist `docs/roadmap.md`. |

## Die Regeln, die kein Stil sind

1. **Nichts unterhalb von `wire/` oder `link/` darf torch importieren**, weder
   direkt noch über ein anderes Modul. Dafür existiert die ganze Schichtung: Ein
   Koordinator soll Datenverkehr weiterreichen können, ohne GPU-Stack. In der CI
   erzwungen; wer es bricht, repariert es mit einem Adapter in L1, nicht mit
   einer Ausnahme. Geprüft wird über einen Subprozess mit Import-Blocker, nicht
   in-process — die Testsuite importiert torch an anderer Stelle, und eine
   In-Process-Prüfung liefe gegen ein bereits geladenes Modul und ginge durch.
2. **L1 fasst niemals einen Socket an.** Wenn eine Splitting-Funktion einen Kanal
   braucht, ist der Entwurf falsch.
3. **Abhängigkeiten zeigen in eine Richtung**: `split` → `wire` → `link`. Die
   obere Schicht darf die untere importieren, niemals umgekehrt.
4. **Nichts hier importiert das Ursprungsprojekt.** `grep -rn "silent_swarm" src/
   tests/` muss leer bleiben.
5. **Wire-Formate sind Kompatibilitätsflächen.** Framing und
   Kompressions-Spec-Strings stehen in fremden Job-Records. Ihre *Bedeutung* zu
   ändern ist ein Breaking Change, auch wenn lokal nichts fehlschlägt.
6. **Eine leere Liste trainierbarer Parameter ist ein harter Fehler**, keine
   Warnung. Ein Optimierer über nichts läuft in jedem Schritt, meldet einen Loss
   und ändert nichts.
7. **Jedes Feature kommt mit Tests und Dokumentation in derselben Änderung.**
   Vom Ursprungsprojekt geerbt, und der Grund, warum dessen Refactorings
   überlebbar sind.

## Tests

Den Quellbaum spiegeln: `tests/unit/<paket>/` für `src/swarmpipe/<paket>/`. Alles,
was das torch-Extra braucht, mit `@pytest.mark.torch` markieren.

Tests bevorzugen, die keine GPU brauchen. Das geht weiter, als es aussieht:
Geräteplatzierung ist gegen das `meta`-Device prüfbar, das es auf jeder Maschine
gibt, und die Splitting-Logik braucht überhaupt keinen Cluster.

## Stil

Zeilenlänge 100. `ruff` für Linting und Importreihenfolge. NumPy-Docstrings auf
jedem öffentlichen Symbol, beginnend mit einer Zeile Zweck. Code, Docstrings und
Kommentare auf Englisch; Dokumentation in `docs/` und `seed/` auf Deutsch,
`README.md` auf Englisch.

Neue Abhängigkeiten nur mit Begründung — die Basisinstallation bleibt leicht,
siehe `THIRD-PARTY.md`. Lieber ein Modul erweitern als eine Abstraktion
hinzufügen: Die Transportschicht des Ursprungsprojekts hat auf genau diesem Weg
zwei konkurrierende Implementierungen bekommen.

Kommentare erklären das *Warum*, besonders dort, wo der Code defensiv ist: Die
meisten Wächter in dieser Bibliothek existieren, weil im Betrieb etwas
fehlgeschlagen ist, und ein Kommentar, der den Fehlschlag benennt, verhindert,
dass der Wächter später „vereinfacht" wird.

## Sprachregelung

Nicht „so schnell wie ein Cluster" und nicht „unbegrenzt skalierbar". Korrekt
ist: Arbeit, die auf einem einzelnen Rechner nicht möglich war, wird auf mehreren
möglich, bei einem Durchsatz, den die Netzverbindung begrenzt und den wir messen.
Kompression an der Stage-Grenze ist verlustbehaftet. Das gilt für
Code-Kommentare, Docstrings und Fehlermeldungen genauso wie für die
Dokumentation.

## Fallstricke

- Module immer bedingungslos auf ihr Zielgerät verschieben, nicht nur bei
  mehreren GPUs — sonst bleiben die Embeddings auf der CPU, während die Eingaben
  schon auf `cuda:0` liegen.
- bfloat16 hat keinen numpy-dtype. Ein torch-freies L2 braucht dafür eine
  ausdrückliche Darstellung; Tests nur auf fp32 finden das nicht.
- Den dtype nie hart kodieren — er ist Eigenschaft des Plans, nicht des Codes.
- Das Relay-Framing und die Kompressions-Spec-Strings nicht ändern: Sie stehen in
  Job-Records laufender Systeme.

## Was nicht hineingehört

Keine Modellgewichte, keine Checkpoints, keine personenbezogenen Daten. Nichts
aus dem Ursprungsprojekt außer dem, was in `seed/` bereits steht — und dieses
bleibt bis zur Portierung unverändert.
