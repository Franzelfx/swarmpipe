# Dokumentation

Grundsatz für diese Phase: **weniger ist mehr.** Dokumentation, die dem Code
vorausläuft, ist veraltet, bevor sie gelesen wird. Was hier steht, muss stimmen;
was nicht mehr stimmt, wird gelöscht, nicht „später aktualisiert".

## Wo was steht

| Ort | Inhalt |
|---|---|
| `README.md`, `README.de.md` | Problem, Ansatz, Status, Abgrenzung zu bestehenden Projekten, Installation und Nutzung sobald es etwas zu nutzen gibt, Lizenz. Die einzige Stelle für Prosa über das Projekt. |
| `CONTRIBUTING.md` | Konventionen, Regeln, Fallstricke — alles, was jemand vor der ersten Änderung wissen muss. |
| Docstrings | Zweck und *Warum* jedes öffentlichen Symbols. Die API-Referenz ist der Code. |
| Tests | Das Verhalten. Ein Test ist die verbindliche Spezifikation, nicht ein Dokument. |
| Issues | Offene Fragen, Ideen, alles jenseits des aktuellen Meilensteins. |
| `doc/` | Diese Datei. Später höchstens `entscheidungen.md`, siehe unten. |

Meilensteine, Zeitplan und Förderlogik stehen in der Projektskizze, nicht im
Repository.

## Was in `doc/` gehört

Nur, was sich dem Code nicht ansehen lässt und trotzdem für alle verbindlich ist:

- **`entscheidungen.md`** — getroffene Entscheidungen mit Folgen über eine Datei
  hinaus (Lizenz, Abhängigkeit, Schnittstelle, die nicht mehr geändert werden
  darf). Pro Eintrag drei Sätze: Was wurde entschieden, warum, was wurde
  verworfen. Nur getroffene Entscheidungen; offene Fragen sind Issues.

Ein Dokument pro Thema, höchstens eine Seite, mit Datum der letzten Prüfung in
der ersten Zeile.

## Was nicht in `doc/` gehört

- Entwürfe für Code, den es noch nicht gibt: Design-Dokumente, Anforderungslisten,
  Portierungsanleitungen. Der Entwurf wird zu Docstrings und Tests, wenn der Code
  entsteht.
- Roadmaps über den aktuellen Meilenstein hinaus. Das sind Issues und Milestones
  beim Repository-Hoster.
- Literaturlisten. Ein Zitat gehört dorthin, wo es gebraucht wird: Docstring oder
  README-Tabelle.
- Zweitfassungen von README-Inhalten.
- Anleitungen und Tutorials vor einer stabilen API.
- Antragstexte und Fördernarrative.
- Generierte Inhalte: API-Referenz, Changelog vor dem ersten Release.

## Regeln

1. Widerspricht ein Dokument dem Code, gilt der Code. Das Dokument wird in
   derselben Änderung korrigiert oder gelöscht.
2. Dokumentation entsteht in derselben Änderung wie das, was sie beschreibt —
   nie davor, nie „nachgereicht".
3. Sprache: Deutsch in `doc/` und `README.de.md`; Englisch in `README.md`, Code,
   Docstrings und Commit-Nachrichten.
4. Im Zweifel nicht dokumentieren. Ein Issue kostet weniger als ein veraltetes
   Dokument.
5. Vor jedem Release und jeder Bewerbungsrunde `doc/` durchgehen: Was nicht mehr
   stimmt, fliegt.
