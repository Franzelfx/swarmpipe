# Anforderungen

Was die Bibliothek leisten muss und welche Randbedingungen sie einhalten muss.
Die zweite Liste wiegt schwerer als die erste: Sie enthält die Dinge, die eine
vernünftige Person falsch machen würde — herausgefunden, indem wir sie in einem
laufenden System falsch gemacht haben.

## Funktional

### L1 — Schneiden

1. **Einen Decoder in zusammenhängende Stages schneiden**, an einer gewählten
   Grenzschicht oder standardmäßig in der Mitte, und die Grenze **aufgelöst**
   zurückmelden, damit wer „Mitte" gesagt hat sehen kann, wo die Mitte lag.
2. **Über Architekturfamilien hinweg funktionieren, ohne Sonderfälle an der
   Aufrufstelle.** In der Praxis gibt es zwei Familien von
   Positionseinbettungen: gelernt-absolut (GPT-2-Stil, ein `wpe`-Modul) und
   rotierend (Llama/Qwen/Mistral, innerhalb der Attention angewandt). Ein
   segmentierter Forward muss den Forward des Frameworks für beide bitgenau
   reproduzieren — also Embedding, kausale Maske und, bei rotierend, den
   cos/sin-Kontext pro Block.
3. **Module pro Gerät platzieren.** Eine Stage kann mehrere lokale GPUs
   überspannen: Gerätezuweisung pro Block, mit eigenen Plätzen für die Embeddings
   und für Abschlussnorm und Head.
4. **Eine Kompressionsgrenze einsetzen** und einen trainierbaren Flaschenhals
   darüber aufteilen, sodass der schmale *Code* über die Leitung geht und nicht
   die Rekonstruktion. Das ist der Unterschied zwischen einer Bandbreitenersparnis
   und einer sinnlosen Rundreise.
5. **LoRA-Adapter einsetzen, beschränkt auf die eigenen Blöcke dieser Stage**, bei
   eingefrorenen vortrainierten Gewichten, sodass zwei Stages disjunkte Hälften
   eines Modells trainieren und ihre Adapter sich anschließend sauber
   zusammenführen lassen.
6. **Die trainierbaren Parameter ausdrücklich auswählen**, in drei Modi:
   vollständiges Fine-Tuning, eingefrorener Backbone (nur der Grenzkompressor
   trainiert) und LoRA.
7. **Ein reines Objekt zurückgeben**, keinen laufenden Prozess: Module, Parameter
   und die wirksame Konfiguration. Kein Optimierer, keine Schleife, kein Kanal.
8. **Den Plan serialisieren.** Der Prozess, der den Schnitt entscheidet, ist oft
   nicht der, der ihn ausführt, und dazwischen liegen womöglich eine
   HTTP-Grenze und ein Job-Record.

### L2 — Die Leitung

9. Einen Tensor als Shape + dtype + Nutzlast rahmen und dabei jeden real
   verwendeten dtype verlustfrei durchreichen: fp32, fp16, **bf16**.
10. Eine **Schritt-ID** mitführen, damit eine Aktivierung und der sie
    beantwortende Gradient gepaart werden können.
11. Zustandslose Codecs bereitstellen — int8, int4, fp16-Cast — als reine
    Wire-Formate, und die Nutzlastgröße für die Bandbreitenrechnung melden.
12. Das Rollenprotokoll besitzen: Upstream sendet eine Aktivierung und empfängt
    `(Loss, Gradient)`; Downstream umgekehrt.

### L3 — Der Transport

13. Ein `Link`-Protokoll: `send_multipart`, `recv_multipart`, `close`, dazu
    Byte-Zähler.
14. Zwei Implementierungen, die *denselben* Vertragstest bestehen: direkt von
    Rechner zu Rechner und über einen HTTP-Koordinator für Gegenstellen hinter
    NAT.
15. Endpunktauflösung mit Ausfallsicherung zwischen beiden.

### Paketierung

16. `pip install swarmpipe` installiert L2 + L3 + die Planungsschicht von L1
    **ohne Deep-Learning-Framework**. Die Chirurgie ist ein optionales Extra.
17. Die Bibliothek hängt von nichts aus dem System ab, aus dem sie herausgelöst
    wurde. Ein Wächtertest erzwingt, dass der Pfeil in eine Richtung zeigt.
18. Ein zweiter Nutzer — einer, der nichts aus dem Ursprungsprojekt importiert —
    muss damit etwas Echtes bauen können. Jede Stelle, an der dieser Nutzer um
    die API herumgreifen muss, ist ein API-Fehler, und der ist vor der
    Veröffentlichung sehr viel billiger zu beheben als danach.

## Randbedingungen, teuer gelernt

Jede davon hat im Ursprungssystem etwas gekostet. Es sind Anforderungen, keine
Ratschläge.

**Die torch-freie Regel ist tragend und zerfällt lautlos.** Alles unterhalb eines
Laufzeitpakets ist einen unachtsamen Import davon entfernt, einen GPU-Stack in
eine Steuerungsebene zu ziehen, und nichts schlägt dabei hörbar fehl — der
Prozess wird nur langsamer und die Installation schwerer. Erzwungen werden muss
das durch einen Test in einem Subprozess mit installiertem Import-Blocker, denn
die Testsuite selbst importiert torch an anderer Stelle, und eine
In-Process-Prüfung ginge gegen ein bereits geladenes Modul durch.

**Das Relay-Wire-Format niemals beiläufig ändern.** Das Framing teilen sich der
Transport und der Koordinator, der es weiterreicht. Eine Änderung bricht einen
laufenden Cluster mitten im Job, und der Fehler erscheint auf einer Maschine, die
jemandem gehört, der die Änderung nicht ausgerollt hat. Bytegleich oder
versioniert; niemals „verbessert".

**Ein Optimierer über einer leeren Parameterliste ist ein stiller No-op.** Er
läuft in jedem Schritt, meldet einen plausiblen Loss und ändert nichts. LoRA macht
das über ein `target_modules` erreichbar, das auf der betreffenden Stage auf
nichts passt. Zur Konstruktionszeit hart scheitern.

**Die Module bedingungslos verschieben.** Ein Platzierungspfad, der nur bei
mehreren konfigurierten GPUs lief, ließ die Embeddings auf der CPU zurück,
während die Eingaben bereits nach `cuda:0` bewegt worden waren — ein
dimensionsrichtiger, gerätefalscher Absturz im Embedding-Lookup, und zwar auf
genau der Einzel-GPU-Konfiguration, die am häufigsten ist.

**bf16 hat keinen numpy-dtype.** Eine torch-freie Wire-Schicht muss ihn
ausdrücklich darstellen. Tests, die nur fp32 abdecken, finden das nicht.

**Spec-Strings sind eine öffentliche Schnittstelle.** Kompressionsprofile wie
`learned_dim256_int8` stehen in Job-Records und Experimentkonfigurationen, die
bereits existieren. Umzubauen, worauf ein Spec *auflöst*, ist in Ordnung; zu
ändern, was ein Spec-String *bedeutet*, entwertet die Historie.

**Beide Hälften eines Schnitts müssen übereinstimmen.** Zwei Stages, die sich über
die Adapterkonfiguration uneinig sind, erzeugen Shards, die sich nicht
zusammenführen lassen — und die Uneinigkeit fällt erst beim Export auf, lange
nachdem die Rechenzeit ausgegeben ist. Die Übereinstimmung braucht einen
Fingerabdruck, den beide Seiten vor dem Training berechnen und vergleichen können.

**Bei reinem Shard-Laden sind die Gewichte der anderen Stage nicht da.** Damit der
Hauptspeicher proportional zur Stage bleibt und nicht zum Modell, materialisiert
jeder Knoten nur seinen eigenen Abschnitt; die Blöcke der anderen Stage tragen
Platzhalter-Tensoren. Jeder Code, der „alle Blöcke" anfasst — die Adapterinjektion
ist der verlockende Fall —, scheitert rundheraus, statt nur Arbeit zu verschwenden.

**Einen dtype wählen, den der Cluster wirklich kann.** bfloat16 hart zu kodieren
scheitert im ersten Schritt auf jeder Karte, die älter ist — also genau auf der
gewöhnlichen Hardware, für die eine solche Bibliothek existiert. Der dtype ist
Eigenschaft des Plans, nicht des Codes.

**Die Platzierung testen, nicht die Verrohrung.** Geräteplatzierung sieht ohne
GPUs untestbar aus und bleibt deshalb meist ungetestet. Sie ist es nicht: Das
`meta`-Device gibt es auf jeder Maschine, also ist „ist dieses Modul wirklich
dorthin gewandert, wo der Spec es hinschickte" eine reine CPU-Zusicherung.
