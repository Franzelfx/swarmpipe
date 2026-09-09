# Hinweise für Claude Code

## Was das Projekt ist

`swarmpipe` — die Maschinerie, um ein Modell über mehrere Rechner zu verteilen
und die Tensoren dazwischen zu bewegen, herausgelöst aus dem Projekt
SilentSwarm. Eine Bibliothek, keine Plattform: kein Scheduler, keine
Steuerungsebene, keine Oberfläche.

Zuerst [docs/design.md](docs/design.md) lesen, dann
[docs/requirements.md](docs/requirements.md) — die zweite Hälfte dieser Seite ist
eine Liste von Dingen, die bereits einmal schiefgegangen sind.

## Befehle

```bash
pytest -q                  # alles
pytest -m "not torch" -q   # nur das, was ohne Deep-Learning-Framework laufen muss
ruff check .
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

## Konventionen

- Python, `src/`-Layout, Typannotationen, `ruff` und `pytest`. Zeilenlänge 100,
  NumPy-Docstrings auf allen öffentlichen Symbolen.
- **Abhängigkeiten zeigen in eine Richtung**: `split` → `wire` → `link`. Niemals
  aufwärts.
- Neue Abhängigkeiten nur mit Begründung. Die Basisinstallation bleibt leicht,
  siehe `THIRD-PARTY.md`.
- Lieber ein Modul erweitern als eine Abstraktion hinzufügen — die
  Transportschicht des Ursprungsprojekts hat auf genau diesem Weg zwei
  konkurrierende Implementierungen bekommen.
- Dokumentation in `docs/` und `seed/` auf Deutsch, `README.md` und Code auf
  Englisch.

## Harte Regeln

- **Nichts unterhalb von `wire/` oder `link/` importiert torch.** Dafür existiert
  die ganze Schichtung: Ein Koordinator soll Datenverkehr weiterreichen können,
  ohne GPU-Stack. Prüfung über einen Subprozess mit Import-Blocker, nicht
  in-process — die Testsuite importiert torch an anderer Stelle, und eine
  In-Process-Prüfung läuft dann gegen ein bereits geladenes Modul und geht durch.
- **Nichts hier importiert das Ursprungsprojekt.** `grep -rn "silent_swarm" src/
  tests/` muss leer bleiben.
- **Relay-Framing und Kompressions-Spec-Strings nicht ändern.** Sie stehen in
  Job-Records laufender Systeme.
- **Eine leere Liste trainierbarer Parameter ist ein harter Fehler**, keine
  Warnung. Ein Optimierer über nichts läuft in jedem Schritt, meldet einen Loss
  und ändert nichts.
- **Jede Änderung kommt mit Tests und Dokumentation.**

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
- Geräteplatzierung ist ohne GPU testbar: Das `meta`-Device gibt es auf jeder
  Maschine.
- Keine Modellgewichte, keine Checkpoints und keine personenbezogenen Daten
  committen.
