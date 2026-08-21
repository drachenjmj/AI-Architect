# E-Shop-Prototyp: Event-Choreografie im Kaufprozess (kuratorischer Auszug)

- **Source title:** Hin zu optimierten Microservice-Choreografien — Konzeption und Bewertung ereignisbasierter Kommunikationsstrukturen für cloud-native Microservices (Chapter 6: Prototyp Implementierung, kuratierte Auszüge)
- **Author(s):** Esad Hrbatović
- **Year:** 2025
- **Knowledge box:** 2
- **Domain:** e-commerce
- **Original PDF:** `Hin zu optimierten Microservice-Choreografien.pdf`

> Deterministic source curation: only the whitelisted Chapter-6 sections on the e-commerce prototype are retained. Section headings and source text are preserved faithfully (German original); only PDF-extraction artifacts (broken hyphens, stray spaces) were repaired. Page numbers refer to the printed thesis pagination. Framework comparisons, benchmarking results, runtime-toolchain material, and bibliography are deliberately excluded.

## 6.1.1. Überblick über funktionale Anforderungen *(printed pp. 31)*

Die in Anhang 1 enthaltene Übersicht dient als Referenz für alle funktionalen Anforderungen und unterteilt sich in ID, Teilanforderung und konkreter Zielsetzung. Die Benutzer-Authentifizierung und Autorisierung muss sowohl eine sichere Registrierung (ID 1.1–1.3) als auch eine rollenbasierte Zugriffskontrolle (ID 1.2.3) umfassen. Die passwortbasierte Authentifizierung soll durch den Einsatz von Hashing-Algorithmen umgesetzt werden.

Die Verwaltung der Benutzerprofile (ID 2) soll die automatisierte Profilerstellung bei der Registrierung und die nachträgliche Änderung und Validierung von eingegebenen Benutzerdaten abdecken. Zusätzlich sollen administrative Tätigkeiten wie die Produktverwaltung (ID 3) implementiert werden. Weitere Anforderungen betreffen die Darstellung, Suche und Filterung von Produkten (ID 4). Die Anbindung eines Warenkorbs und eines vollständigen Bestellprozesses (ID 5 und 6) muss ebenfalls gewährleistet sein. Ziel der Lizenz-Generierung (ID 8) ist, dass zu jedem bestellten digitalen Produkt eine eindeutige Seriennummer generiert wird.

Abschließend sollen E-Mail-Benachrichtigungen (ID 9) und Tracking (ID 10) umgesetzt werden, damit Nutzer:innen automatisiert über abgeschlossene oder fehlerhafte Bestellungen informiert werden. Zusätzlich sind Betriebs- und Nutzungsdaten für Analyse- und Monitoring-Zwecke vorhanden.

## 6.1.2. High Level Architecture *(printed pp. 31–32)*

Die Lösung des umgesetzten E-Shops setzt sich aus zehn Java Microservices zusammen, wobei alle bis auf das API-Gateway mit einer eigenen MongoDB Datenbank verbunden sind. Dazu kommt, dass jeder der Services, bis auf das API-Gateway über Apache Kafka als Message Bus miteinander kommunizieren. Das bietet die Grundlage für das Choreografie Muster mit asynchroner, Event-basierter Kommunikation.

Abbildung 8 zeigt die übergeordnete Architektur, die aus den folgenden Services besteht:

- **ApiGateway:** Bietet einen zentralen Einstiegspunkt für alle Clientanfragen und übernimmt das Routing zu nachgelagerten Diensten.
- **AuthService:** Verwaltet Benutzerauthentifizierung, Registrierung und JWT-Token-Generierung.
- **UserService:** Speichert und verwaltet Benutzerprofile und Informationen zu Rollen sowie Metadaten der Benutzer:innen.
- **ProductService:** Verwaltet den Shop-Katalog der angebotenen digitalen Waren.
- **CartService:** Ermöglicht Benutzer:innen das Hinzufügen, Ändern und Entfernen von Artikeln in ihren Warenkorb und startet den Bestellvorgang.
- **OrderService:** Verwaltet Kundenbestellungen und verfolgt deren Status.
- **PaymentService:** Validiert Zahlungsmethoden, verarbeitet Transaktionen und veröffentlicht Events bei erfolgreicher oder fehlgeschlagener Zahlung.
- **LicenseService:** Generiert und verwaltet digitale Lizenzen für gekaufte Waren.
- **NotificationService:** Versendet Benutzerbenachrichtigungen, z. B. Bestätigungs-E-Mails mit Lizenzen.
- **TrackingService:** Erfasst, speichert und analysiert alle wichtigen Geschäftsereignisse für Monitoring von Events und Benchmarking der Verarbeitungszeit.

## 6.2. Einsatz von Design Patterns *(printed p. 32, Einleitungssatz für Kontinuität)*

Um den Prototyp an moderne Cloud-native Best Practices anzupassen, integriert die Lösung bewusst ausgewählte Architekturmuster, welche im Kapitel davor diskutiert wurden.

## 6.2.1. Event-Messaging und Choreografie *(printed pp. 32–33)*

Die Choreografie ist hier zentraler Bestandteil. In Abbildung 9 wird die Umsetzung anhand des Kaufprozesses visualisiert. Ein:e registrierte Benutzer:in gibt eine Bestellung auf, indem der Checkout Prozess über den CartService gestartet wird. Dies löst eine Reihe von Events aus: CheckoutStartedEvent, OrderCreatedEvent, PaymentSuccessEvent (oder PaymentFailEvent), LicensesGeneratedEvent und NotificationSentEvent. Mehrere Dienste führen lokale Transaktionen durch und erreichen durch das Saga Pattern letztendliche Konsistenz (Eventual Consistency).

## 6.2.2. Saga Pattern *(printed pp. 33–34)*

Jede lokale Transaktion löst Domain Events aus, die nachfolgende lokale Transaktionen in anderen Services auslösen. Tritt während des Prozesses ein Fehler auf, zum Beispiel eine fehlgeschlagene Zahlung, machen kompensierende Transaktionen die vorherigen Vorgänge rückgängig oder passen sie an. Dieser Ansatz spielt mit der ereignisgesteuerten Choreografie zusammen und fördert die Gesamtkonsistenz. Eine fehlgeschlagene Zahlung in diesem System folgt einer Saga, die aus den folgenden Schritten besteht:

1. Der Client sendet eine POST /cart-products/checkout-Anfrage an das ApiGateway und leitet den Bestellvorgang ein.
2. Das ApiGateway leitet die Anfrage an den CartService weiter.
3. Der CartService liest die Warenkorbdaten aus der MongoDB cartdb, generiert eine Bestellnummer und gibt diese über das ApiGateway an den Client zurück.
4. Der CartService generiert ein CheckoutStartedEvent und veröffentlicht es über Kafka.
5. Der OrderService verarbeitet das CheckoutStartedEvent, erstellt eine Bestellung mit dem Status „OFFEN", speichert sie in der MongoDB orderdb und gibt ein OrderCreatedEvent aus.
6. Der CartService verarbeitet das OrderCreatedEvent und löscht den entsprechenden Warenkorb aus der Datenbank, da die Bestellung nun aussteht und der Warenkorb nicht mehr benötigt wird.
7. Der PaymentService verarbeitet das OrderCreatedEvent, validiert die Zahlungsmethode und versucht, die Zahlung abzuwickeln.

Wenn die Zahlung fehlschlägt, wird ein PaymentFailEvent ausgelöst. Von hier an werden Ausgleichstransaktionen, wie in Abbildung 10 gezeigt, durchgeführt:

1. Der OrderService verarbeitet das PaymentFailEvent, lehnt die Bestellung ab und aktualisiert den Bestellstatus in der MongoDB-OrderDB auf „PAYMENT_FAILED".
2. Der NotificationService verarbeitet das PaymentFailEvent, fragt Benutzerdaten (E-Mail und Name) aus der Datenbank ab, erstellt eine E-Mail-Benachrichtigung mit den Details zur fehlgeschlagenen Zahlung und sendet ein NotificationSentEvent.
3. Der NotificationService kommuniziert mit einem externen Mailserver, um den Benutzer:innen eine E-Mail über die fehlgeschlagene Zahlung zu senden.

Während des gesamten Prozesses verarbeitet der TrackingService alle relevanten Ereignisse und protokolliert sie für Analyse- und Auditzwecke in der MongoDB TrackingDb.

## 6.2.3. Database per Service – CQRS *(printed p. 34)*

Jeder Microservice wendet das Database per Service Pattern an und stellt sicher, dass er der Eigentümer seiner Daten bleibt. Diese Trennung unterstützt die unabhängige Bereitstellung und Versionierung von Services und minimiert gleichzeitig unbeabsichtigte Nebeneffekte, die durch gemeinsam genutzte Datenbanken entstehen können.

Um komplexe Abfragen zu verarbeiten und die Leseleistung zu verbessern, ohne die Autonomie der Dienste zu beeinträchtigen, nutzt die Lösung außerdem das CQRS-Muster. Beispielsweise speichert der OrderService eine lokale Kopie bestimmter Benutzerdaten, wie die Rolleninformationen, und aktualisiert diese durch das Abonnieren relevanter Domänenevents wie UserUpdatedEvent. Das vermeidet ineffiziente synchrone Suchvorgänge zur Laufzeit und stellt gleichzeitig sicher, dass die einzelnen Dienste lose gekoppelt bleiben.

## 6.2.4. API-Gateway *(printed p. 35)*

Das API-Gateway fungiert als zentraler Einstiegspunkt für alle externen Client-Anfragen. Es vereinfacht die Kommunikation mit den zugrunde liegenden Microservices, da die Clients die Adressen der einzelnen Dienste nicht kennen müssen. Durch Routing, Aggregation und Weiterleiten von Client-Anfragen verringert das API-Gateway insgesamt die Komplexität des verteilten Systems.

Zudem ermöglicht dieser Ansatz, die Dienste in einem privaten Netzwerk zu verbergen und sie nicht über das öffentliche Internet zugänglich zu machen. Schlussendlich können unterschiedliche API-Gateways eingesetzt werden, beispielsweise beim Erstellen einer Benutzeroberfläche für Administrator:innen und einer für Kund:innen.

## 6.3.3. Monitoring *(printed p. 35; nur E-Shop-/Business-Event-Tracking)*

Das Monitoring erfolgt über einen dedizierten TrackingService, der Events im gesamten System erfasst und Metadaten wie Zeitstempel, Session IDs und Korrelationskennungen aller Anfragen speichert. Das ist angelehnt an die zuvor diskutierten Patterns der Log-Aggregation und Distributed Tracings. Dieses Design ermöglicht eine Rückverfolgbarkeit von Benutzeraktionen und Workflows. Weiters ermöglicht es Leistungsanalysen, wie beispielsweise die Messung der Verarbeitungszeit von Bestellungen und den dazugehörigen Zwischenschritten.

## 6.4. Einschränkungen *(printed p. 36)*

Obwohl der E-Shop-Prototyp so konzipiert wurde, dass er eine realistische Microservice-Architektur widerspiegelt, sollen einige bewusste Vereinfachungen den Umfang reduzieren, damit das Projekt im Rahmen dieser Masterarbeit durchführbar bleibt.

Eine wesentliche Einschränkung besteht darin, dass bestimmte externe Abhängigkeiten, wie beispielsweise echte Zahlungsgateways nicht vollständig integriert sind. Stattdessen wird der Zahlungsvorgang simuliert, um einen realistischen Prozess ohne tatsächliche Finanztransaktionen nachzubilden (siehe Anforderung ID 7).

Eine weitere Einschränkung ist das Fehlen von Maßnahmen zur Ausfallsicherheit. Während das System beispielsweise für die Event-Verwaltung auf Kafka setzt, sind Mechanismen zur Bewältigung von Broker-Ausfällen, zur Nachrichtenwiedergabe oder zur Partitionsneuverteilung nicht implementiert. Dadurch liegt der Fokus auf der Choreografie-Logik und der Bewertung der Frameworks, anstatt auf dem Aufbau eines produktionsreifen verteilten Systems mit hohen Garantien an Verfügbarkeit.

Schließlich wurden einige nicht wesentliche Geschäftsfunktionen bewusst weggelassen. Das betrifft unter anderem benutzergenerierte Produktbewertungen, personalisierte Empfehlungsmechanismen oder ein Frontend mit Dashboards. Diese sind zwar für einen vollwertigen E-Shop relevant, tragen aber nicht direkt zum primären Forschungsziel bei.
