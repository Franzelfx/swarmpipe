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

## Einrichtung

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,torch,lora]"     # GPU-Maschine, oder CPU-torch für Tests
pytest -q
ruff check .
```

Für Arbeiten an der Wire- oder Transportschicht bitte eine **zweite Umgebung ohne
installiertes torch** verwenden — nur so fällt auf, wenn der torch-freie Vertrag
gebrochen wurde:

```bash
pip install -e ".[dev]"
pytest -m "not torch" -q
```

## Die Regeln, die kein Stil sind

1. **Nichts unterhalb von `wire/` oder `link/` darf torch importieren**, weder
   direkt noch über ein anderes Modul. In der CI erzwungen; wer es bricht,
   repariert es mit einem Adapter in L1, nicht mit einer Ausnahme.
2. **L1 fasst niemals einen Socket an.** Wenn eine Splitting-Funktion einen Kanal
   braucht, ist der Entwurf falsch.
3. **Die obere Schicht darf die untere importieren. Niemals umgekehrt.**
4. **Wire-Formate sind Kompatibilitätsflächen.** Framing und
   Kompressions-Spec-Strings stehen in fremden Job-Records. Ihre *Bedeutung* zu
   ändern ist ein Breaking Change, auch wenn lokal nichts fehlschlägt.
5. **Jedes Feature kommt mit Tests und Dokumentation in derselben Änderung.**
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
Kommentare auf Englisch; Dokumentation auf Deutsch.

Kommentare erklären das *Warum*, besonders dort, wo der Code defensiv ist: Die
meisten Wächter in dieser Bibliothek existieren, weil im Betrieb etwas
fehlgeschlagen ist, und ein Kommentar, der den Fehlschlag benennt, verhindert,
dass der Wächter später „vereinfacht" wird.

## Was nicht hineingehört

Keine Modellgewichte, keine Checkpoints, keine personenbezogenen Daten. Nichts
aus dem Ursprungsprojekt außer dem, was in `seed/` bereits steht — und dieses
bleibt bis zur Portierung unverändert.
