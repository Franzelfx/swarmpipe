# Portierungsleitfaden (T0)

Alles unter [`seed/port/`](../seed/port/) ist funktionierender, getesteter Code
aus dem Ursprungsprojekt, wörtlich kopiert samt Original-Dateikopfzeilen. T0
verschiebt ihn nach `src/swarmpipe/` und schreibt seine Importe um. Es ist eine
mechanische Arbeit — kein Neuentwurf — und diese Seite ist die Checkliste.

**Herkunft.** Ursprungs-Repository: `nxpSilentSwarm`, Branch
`feat/quantized-lora`, Commit `1dec850`. Alles Folgende liegt dort unter
`src/silent_swarm/runtime/`.

## Warum das mechanisch ist

Die Abhängigkeitshülle von `build_stage` wurde vor der Extraktion geprüft: Diese
Dateien importieren **nichts** aus der Steuerungsebene des Ursprungsprojekts —
keinen Leader, keinen Worker-Lebenszyklus, keine CLI, keine gemeinsamen
Contracts. Jeder interne Import zeigt auf eine andere Datei aus derselben Liste.
Das ist es, was die Bibliothek überhaupt abtrennbar macht, und es lohnt sich, das
nach dem Umzug erneut zu prüfen:

```bash
grep -rn "silent_swarm" src/ tests/     # muss leer bleiben
```

## Das Manifest

| Seed-Datei | Zeilen | Ziel | Anmerkungen |
|---|---:|---|---|
| `split_torch/stage_builder.py` | 283 | `src/swarmpipe/split/torch/stage_builder.py` | Die T4-Extraktion: `build_stage()`. Der Grund, warum es dieses Repository gibt. |
| `split_torch/test_stage_builder.py` | 304 | `tests/unit/split/torch/test_stage_builder.py` | 19 Tests, nur CPU. Mit `@pytest.mark.torch` markieren. |
| `arch/loader.py` | 332 | `src/swarmpipe/split/torch/loader.py` | **Nur `DecoderTopology` + `introspect_decoder` samt Hilfsfunktionen übernehmen.** Der Rest — `load_causal_lm`, `build_quantization_config`, `full_precision_modules` — ist Checkpoint-Laden, zieht `accelerate`/`bitsandbytes` mit und ist Sache der Aufrufenden. Siehe „Was nicht portiert wird". |
| `arch/arch_adapter.py` | 166 | `src/swarmpipe/split/torch/arch_adapter.py` | Der Klebstoff für Embedding, kausale Maske und Rotary. Bitgenau gegen den Forward des Frameworks geprüft, für beide Familien — die Tests mitportieren. |
| `arch/framework_guard.py` | 85 | `src/swarmpipe/split/torch/_guard.py` | Optional. Schützt davor, zwei Deep-Learning-Frameworks in einem Prozess zu importieren; nur sinnvoll, falls je ein zweites Backend existiert ([D4](decisions.md)). Wegzulassen ist vertretbar — entscheiden, nicht driften lassen. |
| `compression/modules.py` | 134 | `src/swarmpipe/split/compression/modules.py` | `NoCompression`, `FixedQuantization`, `LearnedBottleneck`. **`FixedQuantization` wandert in T3 nach L2** — siehe unten. |
| `compression/factory.py` | 66 | `src/swarmpipe/split/compression/factory.py` | Spec-Strings sind eine öffentliche Schnittstelle. Ihre Bedeutung nicht ändern. |
| `compression/split.py` | 56 | `src/swarmpipe/split/compression/split.py` | `split_bottleneck` — Sender- und Empfängerhälfte mit geteilten Parametern. Genau richtig, wie es ist. |
| `compression/hook.py` | 37 | `src/swarmpipe/split/compression/hook.py` | Einhängen per Forward-Hook. Nur nötig, wenn Kompressoren ohne `build_stage` angehängt werden; portieren oder bewusst weglassen. |
| `finetune/lora.py` | 380 | `src/swarmpipe/split/lora.py` | Auf Modulebene bereits torch-frei (peft wird verzögert importiert), bleibt also in der Basisinstallation importierbar. `save_lora_adapters` / `load_lora_adapters` sind Export-Hilfen — mitportieren, Aufrufende brauchen sie. |
| `finetune/freeze.py` | 42 | `src/swarmpipe/split/freeze.py` | Friert den Backbone wirklich ein, sodass kein Optimierer-Zustand belegt wird. |

Insgesamt rund 1.600 Zeilen zu verschieben, zusätzlich zu den etwa 300, die
bereits in `src/` liegen.

## Import-Umschreibungen

| Von | Nach |
|---|---|
| `silent_swarm.runtime.split.spec` | `swarmpipe.split.spec` |
| `silent_swarm.runtime.split.api` | `swarmpipe.split.api` |
| `silent_swarm.runtime.sharding.plan` | `swarmpipe.split.plan` |
| `silent_swarm.runtime.torch.stage_builder` | `swarmpipe.split.torch.stage_builder` |
| `silent_swarm.runtime.torch.loader` | `swarmpipe.split.torch.loader` |
| `silent_swarm.runtime.torch.arch_adapter` | `swarmpipe.split.torch.arch_adapter` |
| `silent_swarm.runtime.compression.*` | `swarmpipe.split.compression.*` |
| `silent_swarm.runtime.finetune.lora` | `swarmpipe.split.lora` |
| `silent_swarm.runtime.finetune.freeze` | `swarmpipe.split.freeze` |
| `silent_swarm.runtime.framework_guard` | `swarmpipe.split.torch._guard` |

Die drei Dateien, die bereits in `src/swarmpipe/split/` liegen (`spec.py`,
`api.py`, `plan.py`), wurden auf diese Weise schon umgeschrieben — sie sind die
Referenz für Ton und Kopfzeilenformat.

## Lizenzkopfzeilen

Die Dateien in `seed/port/` tragen noch die PolyForm-Kopfzeilen des
Ursprungsprojekts. Beim Verschieben werden sie durch den Apache-2.0-Hinweis
ersetzt, wie ihn die Dateien in `src/swarmpipe/` bereits tragen; die Kopien unter
`seed/` bleiben bis zum Abschluss der Portierung unverändert, damit der Abgleich
möglich bleibt. Hintergrund: [D1](decisions.md) und
[../THIRD-PARTY.md](../THIRD-PARTY.md).

## Was nicht portiert wird

**Checkpoint-Laden.** `load_causal_lm`, `load_stage_shard`, der Bauer der
Quantisierungskonfiguration, der dtype-Auflöser. Ein Modell zu laden — von wo, in
welcher Präzision, wie quantisiert, in welches Verzeichnis ausgelagert — ist eine
Richtlinienentscheidung mit schweren Abhängigkeiten. Die Bibliothek nimmt ein
bereits geladenes Modell entgegen. Das hält `transformers` aus dem
Pflichtinstallationspfad heraus und lässt Aufrufenden die Freiheit, zu laden, wie
sie wollen.

Allerdings ist **das reine Shard-Laden eine echte Anforderung** (§„Bei reinem
Shard-Laden" in [requirements.md](requirements.md)), und Aufrufende werden es
brauchen. Es wird als Dokumentation und Beispiel ausgeliefert, nicht als API, bis
ein zweiter Nutzer die Form bestätigt.

**Platzierungsrichtlinie.** Welche Maschine welche Stage bekommt, wie viel VRAM
frei ist, ob ein Modell passt. Die Bibliothek führt einen Plan aus; ihn zu
entscheiden ist Sache der Aufrufenden.

**Alles zu Jobs.** Job-Records, Fortschrittsmeldungen, Telemetrie, Heartbeats.

## Hinweis zur Reihenfolge

T3 verschiebt `FixedQuantization` aus `split/compression/` nach `wire/codec.py`
als zustandslosen Codec. Das während T0 nicht vorwegnehmen: dort portieren, wo es
liegt, die Suite grün bekommen und es dann bewusst mit dem Äquivalenztest
verschieben, den T3 verlangt. Zwei Refactorings auf einmal sind der Weg, auf dem
die Äquivalenzprüfung ausfällt.

## Abnahme

```bash
pip install -e ".[dev,torch,lora]"
pytest -q                      # alles, einschließlich der portierten Suite
grep -rn "silent_swarm" src/ tests/   # nichts

# und, in einer separaten Umgebung ohne installiertes torch:
pip install -e ".[dev]"
python -c "import importlib.util; assert importlib.util.find_spec('torch') is None"
pytest -m "not torch" -q
```
