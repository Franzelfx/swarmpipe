# Entwurf

Drei Schichten. Der Vertrag jeder Schicht ist enger als der der Schicht darüber,
und die Enge wird durch Tests erzwungen, nicht durch Übereinkunft.

```
┌─ L1  split/ ──────────────────────────────────────────────────────────────┐
│  sieht Modelle. Zerteilt eines in Stages, setzt Kompressionsgrenze und     │
│  Adapter ein, platziert die Module, wählt aus, was trainiert.              │
│  Fasst niemals einen Socket an.                     torch (optionales Extra)│
├─ L2  wire/ ───────────────────────────────────────────────────────────────┤
│  sieht TensorFrame (Shape, dtype, Nutzlast, Schritt-ID) und das            │
│  Rollenprotokoll: Aktivierung vorwärts, (Loss, Gradient) zurück.           │
│                                                            numpy, kein torch│
├─ L3  link/ ───────────────────────────────────────────────────────────────┤
│  sieht list[bytes]. Sendet sie, empfängt sie, zählt sie.                   │
│  Keine Tensoren, keine dtypes, keine Schritte.           pyzmq, kein torch │
└───────────────────────────────────────────────────────────────────────────┘
```

## Warum die Schnitte genau dort liegen

Die Grenzen sind nicht willkürlich; jede wurde dort gezogen, wo ein reales System
bereits ächzte.

**L3 sieht nur Bytes.** In dem System, aus dem dies herausgelöst wurde, war der
Transport in zwei Implementierungen auseinandergelaufen — ein alter PUSH/PULL-
Stack und ein aktueller PAIR-Stack — weil keiner von beiden einen Vertrag hatte,
der sie austauschbar gemacht hätte. Ein Link, der nur `send_multipart` /
`recv_multipart` / `close` kennt, lässt sich gegen ein Relay, ein Mock oder eine
prozessinterne Pipe tauschen, und ein einziger Vertragstest deckt alle ab.

**L2 ist torch-frei.** Das ist die tragende Regel. Ein Koordinator, der Bytes
zwischen zwei Gegenstellen hinter NAT weiterreicht, muss das *Framing* verstehen,
um zu routen — aber niemals einen GPU-Stack dafür brauchen. Sobald der
Frame-Encoder torch importiert, erbt der Koordinator eine mehrere Gigabyte große
Abhängigkeit für das Privileg, Puffer zu kopieren. Dass L2 auf numpy bleibt, ist
der Grund, warum ein selbst betriebener Koordinator auf einer kleinen Maschine
läuft.

**L1 fasst niemals einen Socket an.** Die Modellchirurgie — einen Decoder
introspizieren, seine Blöcke schneiden, Adapter einsetzen, einen Flaschenhals
über die Grenze aufteilen, Module auf Geräte legen, entscheiden, welche Parameter
ein Optimierer sieht — ist der wertvollste und fehleranfälligste Teil des ganzen
Stacks. Im Ursprungssystem lag sie eingebettet in einer 300-zeiligen
Trainingsfunktion, wo sie ohne zwei Prozesse und eine GPU nicht testbar war.
Herausgezogen ist sie ein Funktionsaufruf mit einem reinen Datenargument, und ihre
Regressionssuite läuft auf einem Laptop.

## Die Grenze, die das Ursprungssystem anders gezogen hat

**Kompression sind zwei Anliegen, nicht eines**, und sie gehören auf
verschiedene Schichten:

* **Trainierbare Kompression** — ein gelernter Flaschenhals, dessen
  Abwärtsprojektion beim Sender und dessen Aufwärtsprojektion beim Empfänger
  läuft — hat Parameter, empfängt Gradienten und ist Teil des Modellgraphen. Das
  ist **L1**.
* **Zustandslose Codecs** — int8-/int4-Packung, ein fp16-Cast — sind reine
  Wire-Formate ohne Parameter und ohne etwas zu lernen. Das ist **L2**.

Das Ursprungssystem gab beides aus einem Factory-Aufruf zurück und stattete jeden
Kompressor mit dem Tripel `encode`/`decode`/`wire_nbytes` aus, ob er es brauchte
oder nicht. Die Trennung macht L2 als Bytes-rein/Bytes-raus testbar und lässt L1
ohne Meinung zur Bandbreite zurück.

Eine Folge lohnt die ausdrückliche Erwähnung: Der Straight-Through-Estimator, der
feste Quantisierung differenzierbar macht, ist nur für *trainierbare*
Quantisierung relevant. Ein reiner Wire-Codec hat keinen Gradienten zu schätzen.
Vor dem Verschieben ist zu prüfen, dass keine Konfiguration darauf angewiesen ist,
dass der feste Quantisierer differenzierbar ist — sonst ist der Umzug stillschweigend
eine Verhaltensänderung.

## L1 im Einzelnen

L1 teilt sich noch einmal, und diese Teilung ist es, die den Plan portabel macht.

### Der Plan (framework-frei, kein torch)

`SplitSpec` — wie das Modell geschnitten wird. Grenzschicht, Kompressionsprofil,
LoRA-Konfiguration, ob der Backbone eingefroren wird, sowie dtype und
Quantisierung, für die der Lauf geplant wurde.

`Placement` — wohin die Module dieser Stage kommen. Ein Standardgerät, optional
Geräte pro Block, damit eine Stage mehrere lokale GPUs überspannen kann, und
Sonderfälle für Embeddings und Head.

`StageSpec` — der Anteil einer Stage: Stage-Index, ein `SplitSpec` und ein
`Placement`. Geht durch JSON hin und zurück, damit der Prozess, der den Schnitt
entscheidet, ihn dem Prozess übergeben kann, der ihn ausführt.

`plan.py` — Planung Schicht→Stage und Schicht→GPU. `weighted_split` verteilt
Schichten proportional zum freien VRAM nach dem Largest-Remainder-Verfahren, damit
die Größen exakt aufgehen; `stage_layer_ranges` schneidet ein Modell in
zusammenhängende Pipeline-Bereiche; `plan_block_devices` macht aus einem
VRAM-Profil eine Geräteliste pro Block.

Bemerkenswert ist, was hier *nicht* steht: die Richtlinie, welche Maschine welche
Stage bekommt. Die Bibliothek führt einen Plan aus; ihn zu wählen ist Sache der
Aufrufenden.

### Die Chirurgie (torch-Backend)

`build_stage(model, spec) -> StageBundle` tut der Reihe nach: den Decoder
introspizieren, die Blöcke dieser Stage herausschneiden, LoRA-Adapter **auf diese
Blöcke beschränkt** einsetzen, den Grenzkompressor bauen und über die Grenze
aufteilen, jedes Modul auf das im Spec genannte Gerät bewegen und die
trainierbaren Parameter auswählen.

Zurück kommt ein `StageBundle` — die Blöcke, die Embeddings oder der Head, die
eigene Hälfte des Kompressors, die Liste trainierbarer Parameter und das
gegebenenfalls adaptergekapselte Modell. Es baut keinen Optimierer, führt keine
Schleife aus und hält keinen Kanal.

Zwei Dinge lehnt es rundheraus ab:

* eine Pipeline, die nicht aus zwei Stages besteht (bis der N-Stage-Schnitt
  kommt — siehe Fahrplan);
* eine Stage, die am Ende **nichts zu trainieren** hat. Ein Optimierer über einer
  leeren Parameterliste läuft in jedem Schritt, meldet einen Loss und ändert
  nichts. Erreichbar ist das über ein LoRA-`target_modules`, das auf den Blöcken
  *dieser* Stage auf nichts passt — deshalb ein harter Fehler statt eines
  stillen.

## L2 im Einzelnen (T2, T3)

`TensorFrame` — Shape, dtype, Nutzlast und eine **Schritt-ID**. Die Schritt-ID ist
keine Zierde: Ohne sie lässt sich eine Aktivierung nicht mit dem Gradienten
paaren, der sie beantwortet, und eine Verwechslung nach einem Timeout erzeugt ein
stillschweigend falsches Gewichtsupdate statt eines Fehlers.

`Session` — das Rollenobjekt (`upstream` / `downstream`), dem der Austausch
gehört: Aktivierung senden, `(Loss, Gradient)` empfangen. Im Ursprungssystem war
dieses Protokoll zwischen Kanal und Autograd-Grenze aufgeteilt, sodass keines von
beidem für sich verständlich war.

`codec.py` — die zustandslosen Codecs, arbeitend auf Frames.

**Die bf16-Falle.** bfloat16 hat keinen numpy-dtype. Ein torch-freies L2 braucht
dafür eine ausdrückliche Darstellung — die Nutzlast als rohe Bytes tragen, den
dtype in den Metadaten benennen und L1s Adapter uminterpretieren lassen. Jeder
Entwurf, der annimmt, `numpy.dtype(name)` löse jeden verwendeten dtype auf,
besteht seine Tests auf fp32 und scheitert an genau dem dtype, den diese Cluster
üblicherweise fahren.

## L3 im Einzelnen (T1)

`Link` — `send_multipart(list[bytes])`, `recv_multipart() -> list[bytes]`,
`close()` sowie Byte-Zähler, damit Aufrufende Bandbreite messen können.

`ZmqLink` — direkt von Rechner zu Rechner, wenn die Gegenstellen sich erreichen.

`RelayLink` — über einen HTTP-Koordinator, für den Normalfall, dass mindestens
eine Gegenstelle hinter NAT sitzt. Die Aufgabe des Relays ist, pro Kanal eine
Warteschlange zu halten und die Bytes weiterzureichen; es sieht nie hinein.

`resolve.py` — Endpunktauflösung und Ausfallsicherung: erst direkt versuchen,
dann auf das Relay zurückfallen.

## Der Adapter zwischen L1 und L2

Genau ein Stück muss sowohl Modelle als auch Frames kennen: die Autograd-Grenze.
Sie ist eine `torch.autograd.Function`, deren Vorwärtspfad die Aktivierung sendet
und deren Rückwärtspfad den zurückgekommenen Gradienten liefert — sodass ein
einzelnes `loss.backward()` auf der Upstream-Stage deren Parameter über eine
Gegenstelle auf einer anderen Maschine trainiert. Sie wandelt torch-Tensoren in
Frames und zurück, und sie ist die einzige Stelle im Stack, die das tut.
