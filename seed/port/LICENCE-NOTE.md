# Zur Lizenzkopfzeile in diesen Dateien

Die `.py`-Dateien in diesem Verzeichnis tragen die Kopfzeile

> SilentSwarm — Copyright 2026 NexPatch AI UG.
> Licensed under the PolyForm Noncommercial License 1.0.0. See LICENSE for details.

Das „LICENSE" darin meint die `LICENSE` des **Ursprungs-Repositories**
[SilentSwarm](https://github.com/Franzelfx/nxpSilentSwarm), nicht die dieses
Repositories. Die Kopien hier werden bytegleich gehalten, damit sich nachweisen
lässt, dass die Portierung in T0 nichts außer Importpfaden geändert hat — deshalb
bleiben die Kopfzeilen bis dahin stehen.

**Die Bibliothek selbst steht unter Apache-2.0**, siehe `../../LICENSE` und
[D1](../../docs/decisions.md). Die NexPatch AI UG hält das Urheberrecht an diesem
Code und gibt ihn für die herausgelöste Bibliothek unter Apache-2.0 frei; beim
Verschieben nach `src/swarmpipe/` in T0 wird die Kopfzeile entsprechend ersetzt.

Nichts in diesem Verzeichnis liegt auf dem Importpfad, und nichts davon wird in
ein gebautes Paket aufgenommen. Wer den Code jetzt schon verwenden will, nimmt
ihn aus `src/` — oder wartet auf T0.
