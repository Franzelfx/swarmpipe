# Fahrplan

## Stand heute (Vorarbeit)

- [x] Die Modellchirurgie im Ursprungssystem in einen Bibliotheksaufruf
      herausgezogen (`build_stage`), mit unverändert laufender Regressionssuite
- [x] Nachweis, dass ein Zwei-Maschinen-Schnitt gradientenweise zu einem
      Einzelprozess-Baseline passt (im Ursprungssystem)
- [x] Planungsschicht (`spec.py`, `api.py`, `plan.py`) torch-frei in diesem Repo,
      mit Tests
- [x] Lizenz entschieden: Apache-2.0 ([D1](decisions.md))
- [ ] Physischer Umzug des Seed-Codes nach `src/` (T0)
- [ ] Bandbreitenmessung an einem realen Consumer-Anschluss, mit und ohne
      Kompression — trägt der Ansatz außerhalb des LAN?

Die Bandbreitenmessung ist die Machbarkeitsfrage. Fällt sie negativ aus,
verschiebt sich der Zuschnitt in Richtung Kompression und Inferenz, bevor Aufwand
in die Trainingspfade fließt.

## Geplante Umsetzung

Umgeschrieben für ein eigenständiges Repository aus dem Extraktionsplan in
[`seed/origin/`](../seed/origin/pipeline-library-extraction-epics.md). Der dortige
Plan ging davon aus, dass die Bibliothek innerhalb ihres Ursprungsprojekts wächst
und erst später auszieht; sie von Anfang an im eigenen Repository zu bauen ordnet
die Arbeit um: **T0 ersetzt T6/T9**, und das Ursprungsprojekt wird am Ende zum
Nutzer statt durchgehend zum Wirt.

Größen: S ≈ ein Tag, M ≈ 2–4 Tage, L ≈ eine Woche oder mehr.

| # | Aufgabe | Hängt ab von | Größe | Liefert |
|---|---|---|---|---|
| **T0** | Seed portieren: das torch-Backend von L1 und seine Abhängigkeitshülle | — | M | `build_stage()` läuft hier; `pip install -e .[torch]` grün |
| **T1** | `Link`-Protokoll + zwei Transporte gegen einen Vertragstest | — | M | L3 existiert |
| **T2** | Torch-freie Wire-Frames + Session | T1 | M | L2 existiert, ohne torch importierbar |
| **T3** | Zustandslose Codecs aus den Kompressoren herauslösen | T2 | S | int4/int8-Packung ist ein Wire-Codec, kein Modellmodul |
| **T4** | N-Stage-Schnitt (heute: genau 2) | T0 | M | Pipelines über mehr als zwei Maschinen |
| **T5** | Ende-zu-Ende-Beispiel: zwei Prozesse, ein Modell, echte Loss-Kurve | T2, T0 | S | das, was Neue zuerst ausführen |
| **T6** | Doku-Seite + API-Referenz | T5 | S | nutzbar ohne Kenntnis des Ursprungsprojekts |
| **T7** | Zweiter Nutzer, der nichts aus dem Ursprungsprojekt importiert | T5 | M | Nachweis, dass die Nahtstellen tragen |
| **T8** | Ursprungsprojekt migriert auf die veröffentlichte Bibliothek | T7 | M | Extraktion abgeschlossen, eine Implementierung statt zwei |
| **T9** | `v0.1.0` auf PyPI | T6, T7 | S | per Namen installierbar |

**Reihenfolge: T0 → T1 → T2 → T3 → T5, dann neu bewerten.** T4 und T7 können
parallel laufen, sobald T5 grün ist; T8 steht bewusst am Ende.

---

## T0 — Seed portieren

Die L1-Chirurgie des Ursprungsprojekts ist bereits in einen bibliotheksförmigen
Aufruf (`build_stage`) herausgezogen, und ihre Regressionssuite läuft unverändert.
Was fehlt, ist der physische Umzug: Der Code in [`seed/port/`](../seed/port/)
importiert noch aus seiner alten Heimat.

Das Dateimanifest steht in [porting-guide.md](porting-guide.md). Die gesamte
Abhängigkeitshülle umfasst rund 1.600 Zeilen und berührt nichts aus der
Steuerungsebene des Ursprungsprojekts — das macht daraus eine mechanische Arbeit
und keinen Neuentwurf.

**Abnahme.** `pip install -e ".[torch,lora]"`, danach besteht die portierte
`test_stage_builder.py` unverändert. `pytest -m "not torch"` läuft weiterhin in
einer Umgebung ohne installiertes torch.

## T1 — `Link` und ein Transport-Stack

L3 festlegen: `send_multipart(list[bytes])`, `recv_multipart() -> list[bytes]`,
`close()` und die Byte-Zähler, die die Telemetrie der Aufrufenden speisen. Zwei
Implementierungen — direktes ZMQ und ein HTTP-Relay — und **ein** Vertragstest,
den beide bestehen müssen.

Das Ursprungssystem trägt zwei Transport-Stacks aus zwei Epochen mit sich, und sie
sind bereits einmal auseinandergelaufen. Nur einer kommt mit.

**Abnahme.** Beide Implementierungen bestehen denselben Vertragstest. Nichts
unterhalb von `link/` importiert torch. Das Relay-Framing ist bytegleich zu dem
des Ursprungsprojekts, damit ein gemischter Cluster während T8 weiterläuft.

## T2 — Wire-Frames und Session-Protokoll

`TensorFrame` auf numpy, darauf das Rollenprotokoll und eine **Schritt-ID** in den
Metadaten, damit eine Aktivierung mit ihrem Gradienten gepaart werden kann.

**Abnahme.** Ein Test stellt sicher, dass `swarmpipe.wire` importiert, während
torch nicht in `sys.modules` ist. Round-Trip-Tests für fp32, fp16 und **bf16** —
letzterer hat keinen numpy-dtype und braucht eine ausdrückliche Darstellung. Zu
dokumentieren ist, ob der Loss in den Metadaten der Antwort mitreist oder einen
eigenen Frame bekommt; beides ist vertretbar, undokumentiert nicht.

## T3 — Zustandslose Codecs wandern nach L2

Die Skalier- und Packlogik der festen Quantisierung wird ein Codec über Frames.
Die Factory gibt weiterhin nur trainierbare Module zurück; `fixed_intN` löst auf
einen Wire-Codec auf. **Spec-Strings ändern sich nicht** — bestehende Job-Records
verwenden sie.

**Abnahme.** Numerische Äquivalenz: Derselbe Spec erzeugt vorher und nachher
dieselben Bytes. Zuerst prüfen, ob irgendetwas darauf beruht, dass der feste
Quantisierer differenzierbar ist; ein reiner Codec hat keinen Gradienten zu
schätzen, und falls doch etwas darauf beruhte, ist dies stillschweigend eine
Verhaltensänderung.

## T4 — N Stages

Heute besteht der Schnitt aus genau zwei Stages, und die Annahme steckt ebenso im
Rollenprotokoll wie in der Chirurgie: „Upstream" und „Downstream" sind Rollen,
keine Positionen. Eine mittlere Stage ist beides.

**Abnahme.** Ein Drei-Stage-Schnitt trainiert, und seine Gradienten passen zu
einer Einzelprozess-Baseline. Der Zwei-Stage-Pfad bleibt unverändert.

## T5 — Das Beispiel, das es beweist

Zwei Prozesse, ein kleines Modell, ein echter ZMQ-Link, ein Loss, der sinkt.
Lauffähig auf einem Laptop ohne GPU.

Das ist das Erste, was Neue ausführen, und das Erste, was bricht, wenn eine API
driftet — also gehört es in die CI.

## T6 — Dokumentation und API-Referenz

NumPy-Docstrings auf jedem öffentlichen Symbol, eine Architekturseite und das
Beispiel aus T5 als Startseite. Der Prüfstein ist, ob jemand ohne Kenntnis des
Ursprungsprojekts ein Modell schneiden kann.

## T7 — Ein zweiter Nutzer

Etwas, das swarmpipe verwendet und **nichts** aus dem Ursprungsprojekt importiert
— die ehrliche Probe darauf, ob die Nahtstellen tragen. Der naheliegende
Kandidat: ein Modell über zwei Prozesse schneiden, mit handgeschriebenem
Koordinator, ohne Steuerungsebene, ohne Job-Records.

Jede Stelle, an der dieser Nutzer um die API herumgreifen muss, ist ein Fehler aus
T0–T3. Hier werden sie behoben. Nach `v0.1.0` kosten sie einen Release-Zyklus.

## T8 — Das Ursprungsprojekt migriert

Das Ursprungsprojekt löscht seine eigene Kopie und hängt von der
veröffentlichten Bibliothek ab. Seine Suite muss gegen das Wheel bestehen, nicht
gegen eine Pfadinstallation. Das steht mit Absicht am Ende: Solange T7 die API
nicht gebogen hat, friert eine Migration nur die heutige Form ein.

## T9 — `v0.1.0`

Taggen, veröffentlichen, pinnen.

## Abbruch- und Umsteuerkriterien

- **Bandbreite.** Erreicht ein Zwei-Maschinen-Schnitt über einen typischen
  Consumer-Anschluss selbst mit gelerntem Flaschenhals keinen Durchsatz, der die
  Aufteilung gegenüber Gradient-Checkpointing auf einer Maschine lohnt, dann ist
  Training über WAN nicht der Anwendungsfall. Umsteuern auf den LAN- und
  Inferenzfall; Schichtung und Transport bleiben unverändert gültig.
- **Konvergenz.** Kostet die Kompression an der Grenze mehr Qualität, als die
  Bandbreite einspart, wird die Kompressionsgrenze optional statt vorgesehen —
  und das Ergebnis wird veröffentlicht, statt es wegzulassen.
- **API-Form.** Muss der zweite Nutzer in T7 an mehreren Stellen um die API
  herumgreifen, verschiebt sich der Schwerpunkt von T4 (mehr Stages) auf die
  Reparatur der Nahtstellen. Eine falsche API mit drei Stages ist schlechter als
  eine richtige mit zweien.
- **Fremdes Werkzeug.** Erscheint eine permissiv lizenzierte Bibliothek, die
  Schnitt, Grenzkompression und NAT-Transport abtrennbar bündelt, verlagert sich
  der Schwerpunkt auf das, was dann noch fehlt — voraussichtlich die
  Grenzkompression und der heterogene Fall — statt eine zweite Umsetzung
  danebenzustellen.
