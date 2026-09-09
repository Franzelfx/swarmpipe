# Stand der Technik und Abgrenzung

Stand: September 2026. Dieses Dokument wird vor jeder größeren Veröffentlichung
neu geprüft — das Feld bewegt sich schnell.

## Einschätzung

Das Feld ist **gut besetzt, aber am anderen Ende**. Pipeline-Parallelität im
Rechenzentrum ist ausgereift und exzellent gelöst; verteilte Inferenz auf
Alltagsgeräten hat in den letzten Jahren mehrere brauchbare Werkzeuge bekommen.
Dazwischen liegt eine Lücke: **Training über wenige, ungleiche, schlecht
angebundene Rechner — als Bibliothek, nicht als Plattform.**

Zwei Unterschiede tragen die Lücke:

1. **Vorwärts genügt nicht.** Die Werkzeuge für Alltagshardware sind fast
   ausnahmslos Inferenzwerkzeuge. Der Rückwärtspfad ist die schwierigere Hälfte:
   Er braucht Aktivierungen, die aufbewahrt und gepaart werden, Gradienten, die
   zurückfließen, und eine Autograd-Grenze über den Netzwerkschnitt.
2. **Bibliothek statt Netzwerk.** Wo verteiltes Training über das offene Internet
   existiert, kommt es als Netzwerk mit Protokoll, Client und Schwarm. Wer nur
   die Schneide- und Transportmaschinerie im eigenen Werkzeug braucht, muss das
   Ganze übernehmen.

## Werkzeuge und Bibliotheken

| Projekt | Lizenz / Reife | Deckt ab | Unterschied |
|---|---|---|---|
| [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) | Apache-2.0, sehr reif | Pipeline- und Tensor-Parallelität, ZeRO, Offloading | Für homogene Cluster mit schnellem Interconnect gebaut; ein Framework, das man übernimmt, keine Bibliothek, die man importiert. Keine Grenzkompression, kein NAT-Fall. |
| [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | permissiv (NVIDIA), sehr reif | Referenz für großskalige Pipeline- und Tensor-Parallelität | Setzt eine Rechenzentrumstopologie voraus: einheitliche GPUs, gemeinsames Dateisystem, InfiniBand-Klasse Interconnect. |
| [Colossal-AI](https://github.com/hpcaitech/ColossalAI) | Apache-2.0, reif | Mehrere Parallelitätsstrategien unter einem Dach | Gleiche Zielumgebung; ebenfalls Plattform statt Baustein. |
| [`torch.distributed.pipelining`](https://docs.pytorch.org/docs/stable/distributed.pipelining.html) (ehem. PiPPy) | BSD-3, in PyTorch aufgenommen | Modellaufteilung und Pipeline-Zeitpläne nativ in PyTorch | Setzt eine `torch.distributed`-Prozessgruppe voraus — Ranks, die einander erreichen, ein Rendezvous, kein Relay. Und torch überall, auch dort, wo nur Bytes bewegt werden. **Naheliegender Vergleichspunkt, kein Ersatz.** |
| [Petals](https://github.com/bigscience-workshop/petals) | MIT, aktiv, ACL-2023-Demo | Schichten großer Modelle über einen öffentlichen Schwarm, Inferenz und parametereffizientes Feintuning, NAT-Traversal über hivemind | **Der nächste Verwandte.** Aber ein Netzwerk mit Protokoll und Client, auf einen öffentlichen Schwarm und ein festes Feintuning-Modell ausgelegt. swarmpipe ist die Maschinerie ohne Schwarm — zwei bekannte Rechner, keine Fremden, keine DHT. |
| [hivemind](https://github.com/learning-at-home/hivemind) | MIT, aktiv | Dezentrales Training, DHT, NAT-Traversal, Averaging | Schwerpunkt datenparallel und Mixture-of-Experts über viele Gegenstellen; nicht Pipeline-Chirurgie an einem Modell über wenige Rechner. **Mögliches Vorbild für den Transport.** |
| [exo](https://github.com/exo-explore/exo) | GPL-3.0, sehr sichtbar | Modell über Alltagsgeräte verteilen, automatische Erkennung im LAN | **Nur Inferenz**, keine Gradienten. Und GPL-3.0 — als Abhängigkeit in fremden Werkzeugen problematisch. |
| [distributed-llama](https://github.com/b4rtaz/distributed-llama), llama.cpp mit RPC-Backend | MIT | Inferenz über mehrere Heimgeräte, sehr schlank | Nur Inferenz; kein Trainingspfad, keine gelernte Kompression an der Grenze. |
| [Ray Train](https://github.com/ray-project/ray), [Alpa](https://github.com/alpa-projects/alpa) | Apache-2.0 | Orchestrierung und automatische Parallelisierung | Steuerungsebene und Scheduler — genau die Schicht, die swarmpipe ausdrücklich nicht ist, und die es unter sich haben möchte. |
| [SilentSwarm](https://github.com/Franzelfx/nxpSilentSwarm) | PolyForm-NC + kommerziell | Das Ursprungsprojekt: Scheduler, Steuerungsebene, Relay, verteiltes Training | Eine Plattform, und nicht kommerziell nutzbar. swarmpipe ist ihr abtrennbarer Kern unter Apache-2.0 ([D1](decisions.md)). |

## Wissenschaftliche Arbeiten

**Pipeline-Parallelität**

- **GPipe** — Huang et al., NeurIPS 2019, [1811.06965](https://arxiv.org/abs/1811.06965).
  Mikrobatches und Re-Materialisierung; die Grundform des Verfahrens.
- **PipeDream** — Narayanan et al., SOSP 2019, [1806.03377](https://arxiv.org/abs/1806.03377).
  Asynchrone Zeitpläne und die Buchführung über Gewichtsversionen. Relevant für
  die Schritt-ID: Ohne Paarung von Aktivierung und Gradient ist ein
  Pipeline-Update stillschweigend falsch.
- **Megatron-LM** — Shoeybi et al., 2019, [1909.08053](https://arxiv.org/abs/1909.08053).
  Die Referenz für den Rechenzentrumsfall.

**Training über langsame und heterogene Verbindungen** — der Teil, der diesem
Projekt am nächsten liegt.

- **Learning@home / hivemind** — Ryabinin & Gusev, NeurIPS 2020,
  [2002.04013](https://arxiv.org/abs/2002.04013). Dezentrales Training über
  unzuverlässige Gegenstellen.
- **Decentralized Training of Foundation Models in Heterogeneous Environments** —
  Yuan et al., NeurIPS 2022, [2206.01288](https://arxiv.org/abs/2206.01288).
  Zuordnung von Pipeline-Stages auf ungleiche Knoten mit ungleicher Bandbreite.
- **AC-SGD** — Wang et al., NeurIPS 2022, [2206.01299](https://arxiv.org/abs/2206.01299).
  Kompression der Aktivierungen beim Feintuning über langsame Netze, mit
  Konvergenzaussage. Die theoretische Rückendeckung für die Kompressionsgrenze.
- **SWARM Parallelism** — Ryabinin et al., ICML 2023,
  [2301.11913](https://arxiv.org/abs/2301.11913). Modellparalleles Training über
  unzuverlässige, schlecht verbundene Geräte.
- **Petals** — Borzunov et al., ACL 2023 (System-Demo),
  [2209.01188](https://arxiv.org/abs/2209.01188). Inferenz und Feintuning über
  einen kooperativen Schwarm.

**Parametereffizientes Feintuning**

- **LoRA** — Hu et al., ICLR 2022, [2106.09685](https://arxiv.org/abs/2106.09685).
  Der Grund, warum Feintuning auf gewöhnlicher Hardware überhaupt in Reichweite
  ist, und der Grund, warum zwei Stages disjunkte Hälften eines Modells
  trainieren können, ohne die vortrainierten Gewichte anzufassen.

> **Zu prüfen.** Die Angaben zu Veröffentlichungsjahr, Konferenz und arXiv-Nummer
> stammen aus der Vorrecherche und sind vor der Veröffentlichung dieses Dokuments
> gegen die Primärquellen abzugleichen. Ebenso der aktuelle Stand der oben
> genannten Werkzeuge — Lizenzen und Funktionsumfang ändern sich, und die Tabelle
> behauptet über fremde Projekte etwas, das stimmen muss.

## Wo die Lücke genau liegt

Zusammengefasst deckt jede Zeile oben etwas ab, aber keine deckt die Kombination
ab, um die es hier geht:

| Anforderung | Rechenzentrums-Frameworks | Inferenz auf Alltagsgeräten | Schwarm-Netzwerke | swarmpipe |
|---|---|---|---|---|
| Rückwärtspfad über die Maschinengrenze | ✅ | ✕ | ✅ | ✅ |
| Wenige, ungleiche Rechner | ✕ | ✅ | teils | ✅ |
| Gegenstellen hinter NAT | ✕ | LAN | ✅ | ✅ |
| Gelernte Kompression an der Stage-Grenze | ✕ | ✕ | teils | ✅ |
| Als Bibliothek importierbar, ohne Plattform | ✕ | ✕ | ✕ | ✅ |
| Steuerungsebene ohne GPU-Stack betreibbar | ✕ | — | ✕ | ✅ |

Die letzte Zeile ist die unauffälligste und die folgenreichste. Sobald das
Wire-Format ein Deep-Learning-Framework zum Verstehen braucht, muss jede Stelle,
die Bytes weiterreicht, mehrere Gigabyte Abhängigkeiten installieren. Genau das
entscheidet, ob Selbstbetrieb praktisch oder nur theoretisch ist.
