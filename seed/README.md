# `seed/` — Material aus dem Ursprungsprojekt

Nichts hiervon liegt auf dem Importpfad. Es ist bereitgestellt, nicht
ausgeliefert.

## `port/`

Funktionierender, getesteter Code aus
[SilentSwarm](https://github.com/Franzelfx/nxpSilentSwarm) (Branch
`feat/quantized-lora`, Commit `1dec850`), **wörtlich kopiert, mit unveränderten
Original-Kopfzeilen und Lizenzzeilen**. T0 verschiebt ihn nach `src/swarmpipe/`
und schreibt seine Importe um; das Dateimanifest steht in
[../docs/porting-guide.md](../docs/porting-guide.md).

Er wird absichtlich bytegleich gehalten: Er ist die Referenz für die Prüfung, dass
die Portierung nichts außer Importpfaden geändert hat. Aus demselben Grund ist er
vom Linting ausgenommen.

Die Lizenzkopfzeilen nennen PolyForm Noncommercial — das ist die Lizenz des
Ursprungsprojekts. Die herausgelöste Bibliothek steht unter Apache-2.0
([D1](../docs/decisions.md)); die Kopfzeilen werden beim Verschieben ersetzt, nicht
vorher. Hintergrund: [../THIRD-PARTY.md](../THIRD-PARTY.md).

Das Kernstück ist `port/split_torch/stage_builder.py`: die Modellchirurgie,
bereits in einen einzelnen Bibliotheksaufruf mit reinem Datenargument
herausgezogen, samt `port/split_torch/test_stage_builder.py`, ihrer 19 Tests
umfassenden, reinen CPU-Suite.

## `origin/`

`pipeline-library-extraction-epics.md` — der Extraktionsplan, der innerhalb des
Ursprungsprojekts geschrieben wurde, wörtlich zur Herkunftsdokumentation. Seine
internen Links zeigen auf das Ursprungs-Repository und lösen hier nicht auf.

**Er ist nicht der gültige Plan.** Das ist
[../docs/roadmap.md](../docs/roadmap.md), und dort ist die Arbeit für ein
eigenständiges Repository neu geordnet. `origin/` für die Begründung lesen,
`docs/` für die Arbeit folgen.
