# Abhängigkeiten und Lizenzen

Dieses Projekt steht unter Apache-2.0 und nutzt bestehende Arbeiten, statt sie
nachzubauen.

## Basisinstallation

Was `pip install swarmpipe` mitbringt. Diese Liste bleibt bewusst kurz: Sie ist
das, was ein koordinierender Prozess ohne GPU installieren muss.

| Projekt | Lizenz | Nutzung |
|---|---|---|
| NumPy | BSD-3 | Tensor-Frames in L2, ohne Deep-Learning-Framework |
| PyZMQ | BSD-3 (libzmq: MPL-2.0) | Transport in L3 |

## Optionale Extras

Nur die Modellchirurgie in L1 braucht ein Deep-Learning-Framework.

| Projekt | Extra | Lizenz | Nutzung |
|---|---|---|---|
| PyTorch | `[torch]` | BSD-3 | Modellchirurgie, Autograd-Grenze |
| Transformers | `[torch]` | Apache-2.0 | Decoder-Introspektion |
| Accelerate | `[torch]` | Apache-2.0 | Geräteplatzierung |
| PEFT | `[lora]` | Apache-2.0 | LoRA-Adapter |
| pytest, ruff | `[dev]` | MIT | Tests und Linting |

## Herkunft des Codes

Der Code unterhalb von `seed/port/` stammt aus
[SilentSwarm](https://github.com/Franzelfx/nxpSilentSwarm) (Branch
`feat/quantized-lora`, Commit `1dec850`) und trägt dort noch die
Lizenzkopfzeilen des Ursprungsprojekts.

**SilentSwarm ist dual lizenziert** (PolyForm Noncommercial 1.0.0 sowie eine
kommerzielle Lizenz) und fällt nicht unter Apache-2.0. Die Umlizenzierung des
herausgelösten Teils ist möglich, weil die NexPatch AI UG das Urheberrecht an
diesem Code hält und ihn unter abweichenden Bedingungen freigeben kann.

Praktisch heißt das:

- Die Dateien unter `seed/port/` behalten ihre alten Kopfzeilen, bis T0 sie nach
  `src/swarmpipe/` verschiebt. Sie sind Referenz für den Abgleich, dass die
  Portierung nichts außer Importpfaden geändert hat.
- Beim Verschieben wird die Kopfzeile durch den Apache-2.0-Hinweis ersetzt, wie
  ihn die Dateien in `src/swarmpipe/` bereits tragen.
- Das Ursprungsprojekt selbst bleibt unverändert dual lizenziert. Eine permissive
  Bibliothek und eine nichtkommerzielle Anwendung sind miteinander vereinbar.

Diese Datei ist keine Rechtsberatung. Bei Unsicherheit juristischen Rat
einholen.
