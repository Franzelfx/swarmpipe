# swarmpipe

**Ein Modell über mehrere Rechner aufteilen und die Tensoren dazwischen
bewegen.**

Eine Bibliothek für Pipeline-Parallelität auf der Hardware, die tatsächlich
vorhanden ist: zwei oder drei gewöhnliche Rechner, unterschiedliche GPUs, ein
Consumer-Internetanschluss und keine Cluster-Verwaltung.

> **Status: Pre-Alpha.** Die Planungsschicht von Layer 1 ist implementiert und
> getestet. Das torch-Backend existiert, wartet aber auf seine Portierung (siehe
> [seed/](seed/README.md)); Layer 2 und 3 sind spezifiziert, aber noch nicht
> geschrieben.

[English version](README.md) · [Mitwirken](CONTRIBUTING.md) ·
[Dokumentationsregeln](doc/README.md)

## Das Problem

Ein Modell, das nicht auf eine Grafikkarte passt, lässt zwei Auswege: eine
größere Karte mieten oder ein komplettes Framework für verteiltes Training
übernehmen, das für Rechenzentren gebaut ist. Beides drängt die Arbeit in
zentralisierte Rechenkapazität, und das zweite ist meist schwerer als das
Problem — ein vollständiger Cluster-Stack, um ein Modell über zwei
Arbeitsplatzrechner zu verteilen.

Die Folgen sind nicht bloß unbequem. Forschende ohne institutionelles Budget,
kleine Organisationen, öffentliche Stellen mit Daten, die das Haus nicht
verlassen dürfen, und Interessierte mit zwei Spiele-PCs sind von Arbeiten
ausgeschlossen, welche die vorhandene Hardware zusammengenommen leisten könnte.
Die Hardware ist da; ungenutzte GPUs gibt es überall. Was fehlt, ist die
Verbindungstechnik, die mehrere gewöhnliche Rechner wie einen größeren wirken
lässt — und zwar so verpackt, dass ihre Nutzung nicht bedeutet, jemandes
Plattform zu übernehmen.

## Der Ansatz

Ein Modell wird in *Stages* zerschnitten: Jeder Rechner führt einen
zusammenhängenden Abschnitt des Decoders aus und reicht die Aktivierung an der
Schnittstelle vorwärts und deren Gradienten zurück. Drei streng geschichtete
Bausteine, jede Schicht mit einem engeren Vertrag als die darüber:

```
swarmpipe/
  split/   L1  sieht Modelle.        Zerteilt eines, setzt die Kompressions-
                                     grenze ein, wählt aus, was trainiert.
                                     Fasst niemals einen Socket an.
  wire/    L2  sieht Tensor-Frames.  Shape, dtype, Nutzlast, Schritt-ID — und
                                     das Rollenprotokoll für Vorwärts- und
                                     Rückwärtspfad. Torch-frei.
  link/    L3  sieht list[bytes].    Direkt von Rechner zu Rechner oder über
                                     ein Relay für Gegenstellen hinter NAT.
                                     Nichts oberhalb von Bytes.
```

**Nur das Backend von L1 importiert torch.** Das ist keine Ordnungsliebe: Es ist
die Voraussetzung dafür, dass ein koordinierender Prozess — die Stelle, die Bytes
zwischen zwei Gegenstellen weiterreicht, die sich nicht direkt erreichen können —
auf einer kleinen, dauerhaft laufenden Maschine ohne CUDA, ohne PyTorch und mit
fünf Sekunden Startzeit läuft.

```bash
pip install swarmpipe          # L2 + L3 + Planungsschicht von L1. Ohne torch.
pip install swarmpipe[torch]   # ...zusätzlich die Modellchirurgie, für GPU-Knoten.
```

Das Ergebnis ist eine Bibliothek, keine Plattform: kein Scheduler, keine
Steuerungsebene, keine Oberfläche und keine Meinung dazu, wem die Rechner gehören
oder wo die Gewichte liegen.

## Der eigentliche Beitrag

Pipeline-Parallelität ist nichts Exotisches, und die Einzelteile sind ausgereift.
Was es nicht gibt, ist diese Kombination als abtrennbare Bibliothek: Die
funktionierenden Implementierungen sind mit den Systemen verschweißt, in denen
sie gewachsen sind — einem Trainings-Framework, einem Cloud-Scheduler, einer
Steuerungsebene. Wer ein Modell über zwei gewöhnliche Rechner verteilen will,
übernimmt entweder jemandes gesamten Stack oder schreibt die Chirurgie erneut.

Die Lücke ist der kleine, heterogene, schlecht angebundene Fall, und er braucht
anderes als das Rechenzentrum: **Kompression an der Stage-Grenze**, weil die
Bandbreite der Engpass ist und nicht der Interconnect; **ein Relay**, weil NAT
bei Consumer-Anschlüssen die Regel ist; **Präzision pro Maschine**, weil die
Hardware gemischt ist; und **gar keinen Scheduler**, weil die Nutzerin selbst
plant.

swarmpipe ist genau diese Maschinerie für sich, herausgelöst aus einem laufenden
System ([SilentSwarm](https://github.com/Franzelfx/nxpSilentSwarm)) statt am
Reißbrett entworfen, mit den Nahtstellen dort, wo die Erfahrung sie hinlegt.

## Umfang

**Enthalten:** eine Python-Bibliothek zum Einbetten in bestehende Stacks; die
Modellchirurgie mit Schnitt, Kompressionsgrenze, LoRA und Geräteplatzierung; ein
torch-freies Wire-Format samt Transport einschließlich NAT-Relay; ein
Ende-zu-Ende-Beispiel, das in zwei Prozessen auf einem Laptop läuft; ein zweiter,
unabhängiger Nutzer der API als Nachweis, dass die Nahtstellen tragen.

**Nicht enthalten:** Scheduler, Steuerungsebene, Oberfläche, Modell-Hosting.
Keine Meinung zu Optimierern, Zeitplänen oder Checkpointing — das sind
Anwendungen, und dies ist die Schicht darunter. Kein zweites
Deep-Learning-Backend auf Vorrat.

**Formulierungsdisziplin:** nicht „ein 70B-Modell auf zwei Spiele-PCs, so schnell
wie in der Cloud". Pipeline-Parallelität über einen Consumer-Anschluss ist durch
diesen Anschluss begrenzt. Die ehrliche Aussage ist enger: Arbeit, die auf einem
Rechner unmöglich war, wird auf mehreren möglich — bei einem Durchsatz, den wir
veröffentlichen und nicht andeuten. Kompression an der Grenze ist verlustbehaftet,
und ihre Wirkung auf die Konvergenz ist eine Messung, die wir schulden, keine
Annahme. Nichts hiervon macht ein langsames Netz schnell.

## Lizenz

Apache-2.0. Abhängigkeiten siehe [THIRD-PARTY.md](THIRD-PARTY.md). Das
Ursprungsprojekt SilentSwarm bleibt dual lizenziert (PolyForm Noncommercial) und
fällt ausdrücklich nicht unter diese Lizenz; nur die herausgelöste Bibliothek
tut es.
