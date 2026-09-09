# Entscheidungen

Festgehalten, damit sie bewusst getroffen werden und nicht nebenbei. Erledigte
Entscheidungen bleiben stehen — mit Begründung, damit sie nicht in einem halben
Jahr erneut verhandelt werden.

## D1 — Lizenz ✅ entschieden: Apache-2.0

**Status: erledigt.** `LICENSE` enthält den vollständigen Text; die Kopfzeilen in
`src/` und `tests/` verweisen darauf.

Das Ursprungsprojekt ist dual lizenziert: PolyForm Noncommercial 1.0.0 plus eine
kommerzielle Lizenz. PolyForm-NC untersagt kommerzielle Nutzung, was diese
Bibliothek in einem kommerziellen Projekt unbrauchbar gemacht hätte — **auch im
eigenen** — und ebenso für jede Person, die ein Modell für ihr Unternehmen
feinabstimmen will, also für einen großen Teil derer, für die diese Bibliothek
gedacht ist.

Die NexPatch AI UG hält das Urheberrecht und kann die Bibliothek unter
abweichenden Bedingungen freigeben. Zur Wahl standen **Apache-2.0** (permissiv,
ausdrückliche Patentlizenz, Standard für Python-Infrastruktur) und **MPL-2.0**
(dateiweises Copyleft, hält Verbesserungen an der Bibliothek offen und erlaubt
zugleich kommerzielle Einbettung).

Zwei Gründe machten die Sache dringlich statt aufschiebbar: Eine Umlizenzierung
nach dem Eintreffen externer Beiträge braucht die Zustimmung **jeder**
beitragenden Person, und dieses Fenster schließt sich in dem Moment, in dem das
Repository öffentlich wird. Und öffentliche Förderprogramme setzen eine
OSI-anerkannte Lizenz in aller Regel voraus; PolyForm-NC ist keine.

**Entschieden für Apache-2.0**, bei unverändertem Ursprungsprojekt: Eine
permissive Bibliothek und eine nichtkommerzielle Anwendung sind miteinander
vereinbar, und die Bibliothek ist der Teil, dem breite Übernahme nützt. Die
Dateien unter `seed/port/` behalten bis zu ihrer Portierung in T0 die Kopfzeilen
des Ursprungsprojekts; siehe [../THIRD-PARTY.md](../THIRD-PARTY.md).

## D2 — Name

`swarmpipe` ist ein Arbeitstitel aus dem Ursprungsprojekt.

Vor jedem öffentlichen Auftritt die Verfügbarkeit auf PyPI prüfen. Ein Name ohne
„swarm" altert besser, falls die Bibliothek das Projekt überlebt, aus dem sie
stammt — es geht um Pipeline-Parallelität, nicht um Schwärme.

## D3 — Kommt die Trainingsschleife hinein?

Heute gibt `build_stage` ein Bundle zurück, und die Schleife schreiben die
Aufrufenden. Eine mitgelieferte Schleife hieße, eine Meinung zu Optimierern,
Zeitplänen, Gradientenakkumulation und Checkpointing mitzuliefern — und genau
darüber sind sich Aufrufende uneinig.

**Zurückgestellt auf T7.** Schreibt der zweite Nutzer eine Schleife, die im
Wesentlichen der des Ursprungsprojekts gleicht, gehört sie in die Bibliothek —
vermutlich als `train_stage(bundle, session, ...)` neben dem Bundle, nicht
anstelle.

## D4 — Umfang der Framework-Unterstützung

Das torch-Backend von L1 wird das einzige bleiben. Die Protokollnaht bleibt
trotzdem — sie kostet nichts und ist es, was L2/L3 torch-frei hält —, aber ein
zweites Backend wird nicht auf Vorrat gebaut.

## D5 — Wo das Planobjekt lebt

`SplitSpec` lebt in der Bibliothek, und Nutzer re-exportieren es. Die Alternative
— jeder Nutzer hält ein eigenes und konvertiert — vermeidet eine Abhängigkeit von
einer Steuerungsebene auf die Bibliothek; aber eine Steuerungsebene, die die
Bibliothek nicht installieren kann, ist genau das Problem, das die torch-freie
Basisinstallation bereits löst. Entschieden zugunsten der Bibliothek; erneut zu
prüfen nur, falls die Basisinstallation aufhört, leicht zu sein.

## D6 — Projektführung

Folgt auf D1 und ist jetzt fällig, da das Repository öffentlich werden kann.
Festzulegen: wer prüft, wo die Latte für Beiträge liegt und ob ein CLA oder ein
DCO gilt. Ein DCO (eine Sign-off-Zeile) ist die leichtere Variante und erzeugt
nicht die Umlizenzierungs-Verbindlichkeit, wegen der ein CLA sonst eingeführt
wird — er erhält allerdings auch nicht die Möglichkeit einer späteren
Umlizenzierung. Da D1 entschieden ist, wiegt das weniger schwer.

**Empfehlung: DCO**, plus eine kurze Aussage dazu, was hineingehört, in
[../CONTRIBUTING.md](../CONTRIBUTING.md).
