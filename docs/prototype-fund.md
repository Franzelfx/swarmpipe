# Prototype Fund — Arbeitsfassung des Antrags

> **Vor Verwendung die aktuellen Programmregeln prüfen.** Rundentermine,
> Fördersummen, Förderfähigkeit und die durchführende Organisation haben sich im
> Lauf des Programms mehrfach geändert, und diese Fassung entstand ohne Zugriff
> auf die aktuelle Ausschreibung. Auf der offiziellen Seite abgleichen.
>
> [D1](decisions.md) ist erledigt: Die Bibliothek steht unter Apache-2.0 und
> erfüllt damit die Anforderung an eine OSI-anerkannte Lizenz.

## In einem Satz

Eine freie Bibliothek, mit der sich ein großes Sprachmodell über die Rechner
verteilen lässt, die ohnehin vorhanden sind — und dort ausführen oder
feinabstimmen —, statt eine Maschine zu mieten, die groß genug ist, es
aufzunehmen.

## Das Problem

Wer mit einem Modell arbeiten will, das nicht auf eine Grafikkarte passt, hat
heute zwei Antworten: eine größere Karte mieten oder ein komplettes Framework für
verteiltes Training übernehmen, das für Rechenzentren gebaut ist. Beides drängt
in Richtung zentralisierter Rechenkapazität, und das zweite ist oft schwerer als
das Problem — ein vollständiger Cluster-Stack, um ein Modell über zwei
Arbeitsplatzrechner zu verteilen.

Das hat Folgen über die Bequemlichkeit hinaus. Forschende ohne institutionelles
Budget, kleine Organisationen, öffentliche Stellen mit Daten, die nicht in eine
kommerzielle Cloud dürfen, und Interessierte mit zwei Spiele-PCs sind von Arbeit
ausgeschlossen, die ihre Hardware zusammengenommen leisten könnte. Die Hardware
ist da — ungenutzte GPUs gibt es überall. Was fehlt, ist die Verbindungstechnik,
die mehrere gewöhnliche Rechner wie einen größeren wirken lässt, verpackt so,
dass ihre Nutzung nicht bedeutet, jemandes Plattform zu übernehmen.

## Was wir bauen wollen

Die Maschinerie für pipeline-paralleles Ausführen von Modellen, als kleine
installierbare Bibliothek ohne angehängte Plattform:

1. **Ein Modell schneiden.** Einen Transformer an einer gewählten Schicht
   zerteilen, jeden Teil auf das richtige Gerät legen, optional eine gelernte
   Kompressionsgrenze einsetzen, damit weniger Daten über das Netz gehen, und
   parametereffizientes Feintuning (LoRA) einrichten, sodass Training ohne das
   Speicherbudget eines Rechenzentrums möglich ist.
2. **Die Tensoren rahmen.** Ein minimales Wire-Format für die Aktivierungen und
   Gradienten, die zwischen den Maschinen wandern, einschließlich der
   Buchführung, die aus einer Verwechslung einen Fehler macht statt eines
   stillschweigend falschen Ergebnisses.
3. **Die Bytes bewegen.** Direkter Transport von Rechner zu Rechner, wo möglich,
   und ein Relay für den Normalfall, dass ein Heim- oder Büroanschluss hinter
   einem Router sitzt, der keine eingehenden Verbindungen annimmt.

Die drei Teile sind streng geschichtet, und nur der erste braucht ein
installiertes Deep-Learning-Framework. Ein koordinierender Prozess — die Stelle,
die Datenverkehr zwischen zwei Gegenstellen weiterreicht, die einander nicht
erreichen — läuft auf einer kleinen, dauerhaft eingeschalteten Maschine ohne GPU
und ohne PyTorch. Diese Randbedingung ist es, die Selbstbetrieb realistisch statt
theoretisch macht, und sie wird durch Tests erzwungen, nicht durch gute Absicht.

## Warum das glaubwürdig ist

Das ist keine Entwurfsskizze. Die Maschinerie existiert und läuft in einem
größeren, selbst betriebenen System, das wir gebaut haben
([SilentSwarm](https://github.com/Franzelfx/nxpSilentSwarm)), wo ein
Zwei-Maschinen-Schnitt nachweislich gradientenweise zu einer
Einzelprozess-Baseline passt. Das Problem ist, dass sie mit diesem System
*verschweißt* ist: Die Modellchirurgie lag innerhalb einer Trainingsfunktion,
zusammen mit Optimierer, Data-Loader und Netzwerkkanal, und war ohne den Rest der
Anwendung weder nutzbar noch testbar.

Der erste Extraktionsschritt ist bereits getan und in diesem Repository
enthalten: Die Splitting-Logik ist jetzt ein einzelner Bibliotheksaufruf mit
einem reinen Datenargument, mit einer Regressionssuite, die auf einem Laptop ohne
GPU läuft. Die geförderte Arbeit ist, die Sache ordentlich zu Ende zu bringen:
die beiden unteren Schichten, der N-Stage-Schnitt, die Dokumentation und ein
zweiter, unabhängiger Nutzer der API als Nachweis, dass die Nahtstellen tragen.

Wir wissen, wo die Fallen liegen, weil wir hineingetreten sind: ein Wire-Format,
das ein Koordinator ohne GPU-Stack nicht lesen kann; eine Aktivierung, die nach
einem Timeout mit dem falschen Gradienten gepaart wird und einen plausiblen Loss
bei falschem Modell erzeugt; ein hart kodiertes Zahlenformat, das auf jeder etwas
älteren Karte scheitert — also genau auf der gewöhnlichen Hardware, für die das
Ganze gedacht ist. Das steht in [requirements.md](requirements.md) und ist
Anforderung an die geförderte Arbeit.

## Wie es sich von Vorhandenem unterscheidet

Großskalige Pipeline-Parallelität ist von Frameworks gut abgedeckt, die auf
homogene Cluster mit schnellem Interconnect und einen Scheduler zielen. Darin sind
sie ausgezeichnet, und sie setzen es voraus: einheitliche Hardware, ein
gemeinsames Dateisystem, Maschinen, die einander Verbindungen öffnen können. Am
anderen Ende gibt es inzwischen brauchbare Werkzeuge, um **Inferenz** über
Alltagsgeräte zu verteilen — aber ohne Rückwärtspfad, und der ist die schwierigere
Hälfte.

Die Lücke ist der kleine, heterogene, schlecht angebundene Fall mit Training: zwei
oder drei gewöhnliche Rechner, verschiedene GPUs, Consumer-Internet, keine
Cluster-Verwaltung. Dieser Fall braucht anderes — Kompression an der Grenze, weil
die Bandbreite der Engpass ist, ein Relay, weil NAT die Regel ist, Präzision pro
Maschine, weil die Hardware gemischt ist, und gar keinen Scheduler, weil die
Nutzerin selbst plant. Die vollständige Abgrenzung mit Quellen steht in
[prior-art.md](prior-art.md).

Es unterscheidet sich außerdem im Zuschnitt: Dies ist eine Bibliothek, keine
Plattform. Sie hat keine Meinung dazu, wem die Maschinen gehören, wer die Arbeit
plant oder wo die Gewichte liegen. Das ist Absicht — es ist der Grund, warum sie
in fremden Werkzeugen verwendbar ist.

## Zielgruppen und wie sie erreicht werden

* **Forschende und Studierende ohne Rechenbudget** — erreichbar über die
  Dokumentation und ein lauffähiges Zwei-Prozess-Beispiel, das auf einem Laptop
  läuft.
* **Kleine Organisationen und öffentliche Stellen** mit Daten, die das Haus nicht
  verlassen dürfen und die Inferenz oder Feintuning auf eigener Hardware
  brauchen.
* **Entwicklerinnen und Entwickler selbst betriebener KI-Werkzeuge**, die die
  Splitting-Maschinerie wollen, ohne eine Anwendung drumherum zu übernehmen.
  Diese Gruppe entscheidet, ob die Bibliothek ihre Herkunft überlebt: Die API
  wird während der Förderphase gegen einen zweiten, unabhängigen Nutzer geprüft,
  gerade damit sie nicht von den Gewohnheiten eines einzigen Aufrufers geformt
  wird.
* **Die Community rund um mehrere GPUs im Eigenbau**, die groß und laut ist und
  das Problem derzeit mit Skripten löst.

## Arbeitsplan

Sechs Monate. Die Aufgaben sind in [roadmap.md](roadmap.md) spezifiziert; dies
ist ihr Zeitplan.

| Monat | Meilenstein | Fertig, wenn |
|---|---|---|
| 1 | **Fundament.** Splitting-Schicht portieren; Lizenz, Repository, CI, torch-freie Garantie unter Test. | `build_stage()` läuft aus der Bibliothek; die Basisinstallation hat keine Deep-Learning-Abhängigkeit, und die CI weist es nach. |
| 2 | **Transport.** Ein Link-Protokoll, direkt und über Relay, ein Vertragstest für beide. | Beide Transporte bestehen denselben Test; ein Lauf über Relay funktioniert zwischen zwei Maschinen hinter NAT. |
| 3 | **Wire-Format.** Frames, Session-Protokoll, Schritt-Paarung, jedes real genutzte Zahlenformat. | Round-Trips für fp32/fp16/bf16; ein nicht passender Schritt ist ein Fehler, kein falsches Update. |
| 4 | **Es läuft.** Zustandslose Codecs; das Zwei-Prozess-Beispiel; Bandbreitenmessungen mit und ohne Kompression. | Wer das Repository klont, trainiert in unter zehn Minuten über zwei Prozesse. |
| 5 | **Mehr als zwei Maschinen.** N-Stage-Schnitt; Dokumentationsseite und API-Referenz. | Ein Drei-Stage-Schnitt passt zu einer Einzelprozess-Baseline. |
| 6 | **Nachweis und Veröffentlichung.** Ein unabhängiger zweiter Nutzer; die API-Korrekturen, die er erzwingt; `v0.1.0` veröffentlicht. | Ein Programm, das nichts aus unserem anderen Projekt importiert, trainiert ein geschnittenes Modell. Per Namen installierbar. |

Bewusst nicht im Umfang: ein Scheduler, eine Oberfläche, Modell-Hosting und
jegliche Meinung zu Optimierern. Das sind Anwendungen, und dies ist die Schicht
darunter.

## Risiken

* **Die API wird von ihrem einzigen Nutzer geformt.** Die Gegenmaßnahme ist,
  Monat 6 wie beschrieben durchzuführen — ein zweiter Nutzer *vor* der
  Versionsnummer, nicht danach. Jede Stelle, an der dieser Nutzer um die API
  herumgreifen muss, ist ein Fehler, der dann noch billig zu beheben ist.
* **Abdriften in Richtung Plattform.** Das Ursprungsprojekt ist eine Plattform,
  und die Versuchung, seine Bequemlichkeiten mitzunehmen, ist ständig. Die
  Schichtverträge stehen geschrieben und werden durch Import-Tests erzwungen.
* **Bandbreite über Consumer-Anschlüsse.** Es ist möglich, dass der
  Trainingsdurchsatz über eine gewöhnliche Leitung selbst mit Kompression zu
  niedrig ausfällt. Das wird früh gemessen (siehe Abbruchkriterien in
  [roadmap.md](roadmap.md)), und das Ergebnis wird veröffentlicht — auch ein
  negatives ist ein nützliches Ergebnis für alle anderen, die es versuchen.
* **Hardwarezugang für realistische Tests.** Verhalten über mehrere GPUs und
  mehrere Maschinen lässt sich auf einem Laptop nicht vollständig prüfen.
  Abgemildert dadurch, dass alles, was CPU-testbar *sein kann*, auch CPU-getestet
  wird — Geräteplatzierung, Chirurgie, Frames, Transporte —, damit Cluster-Zeit
  nur für das aufgewendet wird, was sie wirklich braucht.

## Hinweis zum Budget

Das Programm hat historisch etwa sechs Monate individueller Arbeit bis zu einer
festen Obergrenze gefördert, wobei Teams sich ein Projektbudget teilen. Die
aktuelle Höhe und Struktur vor dem Eintragen aus der Ausschreibung bestätigen.
