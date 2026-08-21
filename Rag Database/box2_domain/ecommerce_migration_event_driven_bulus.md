# Monolith-to-Microservices-Migration im E-Commerce: Zerlegung, Event-Driven Patterns und Leistung (kuratorischer Auszug)

- **Source title:** Migrating Monolithic E-Commerce Systems to Microservices: A Systematic Review of Event-Driven Architecture Approaches (curated excerpts)
- **Author(s):** Stephen W. Bulus, Olubukola D. Adekola, Folasade Y. Ayankoya, Oluwabamise J. Adeniyi, Ayodeji G. Abiodun
- **Year:** 2025
- **Knowledge box:** 2
- **Domain:** e-commerce
- **Original PDF:** `AJSTE_O1G0V4GO.pdf`

> Deterministic source curation: only the whitelisted Introduction paragraphs on e-commerce/event-driven migration, FINDINGS 1 (Decomposition Frameworks), FINDINGS 2 (Event-Driven Architecture Patterns), DISCUSSION 1 (Key Success Factors), and DISCUSSION 2 (Performance Considerations) are retained. Source text is preserved faithfully; only PDF-extraction artifacts (spaced hyphens, broken words) were repaired. Page numbers refer to the printed journal pagination (pp. 1–19). Headline percentage claims, the review-process sections, study-summary tables, generic organizational discussion, future research, and references are deliberately excluded.

## Introduction — relevante Absätze zur E-Commerce-Migration *(printed p. 2)*

The rapid evolution of e-commerce platforms has necessitated architectural paradigm shifts to meet increasing demands for scalability, reliability, and agility [2]. Traditional monolithic architectures, while providing simplicity in development and deployment, often become bottlenecks as e-commerce systems scale to handle millions of transactions and users [3]. The centralized nature of monolithic systems creates inherent limitations in terms of independent scaling, technology diversity, and fault isolation, making them unsuitable for modern e-commerce requirements.

Microservices architecture has emerged as a compelling alternative, offering the promise of enhanced scalability, improved fault tolerance, and greater development agility [1]. By decomposing monolithic applications into smaller, independently deployable services, organizations can achieve better resource utilization, faster deployment cycles, and improved system resilience.

Event-driven architecture (EDA) patterns have gained particular attention in the context of microservices migration, especially for e-commerce systems where business processes are inherently event-oriented. Customer actions such as placing orders, updating profiles, or processing payments naturally translate to domain events that can drive system behavior [5]. The asynchronous nature of event-driven communication provides better decoupling between services, enhanced fault tolerance, and improved scalability compared to synchronous request-response patterns [4].

Despite growing interest in microservices adoption, the migration from monolithic e-commerce systems remains a complex undertaking fraught with technical and organizational challenges. Abgaz et al. [1] note that monolith decomposition into microservices remains at an early stage, with insufficient standardized methods for combining static, dynamic, and evolutionary data analysis. The absence of established metrics, datasets, and baselines for evaluating migration success further complicates decision-making for practitioners.

## FINDINGS — 1. Decomposition Frameworks and Methodologies *(printed pp. 11–12)*

The reviewed studies reveal several distinct approaches to monolith decomposition, each with specific advantages for e-commerce contexts.

**a. Domain-Driven Design (DDD) Approaches:** Multiple studies emphasize Domain-Driven Design as a foundational approach for identifying service boundaries in e-commerce systems. Abgaz et al. [1] identify DDD as a critical component of their Monolith to Microservices Decomposition Framework (M2MDF), noting its effectiveness in creating business-aligned service boundaries. Kaloudis [6] reinforces this approach, demonstrating how DDD enables better service cohesion and reduced coupling. In e-commerce contexts, natural domain boundaries often align with business capabilities, such as customer management, product catalog, order processing, payment handling, and inventory management.

**b. Process Mining-Based Decomposition:** Taibi and Systä [7] present a novel 6-step framework utilizing process mining to reduce subjectivity in decomposition decisions. Their approach analyzes runtime execution traces to identify independent service candidates, offering a data-driven alternative to purely architectural analysis. The framework showed particular promise in industrial applications, helping to identify decomposition options that manual analysis missed. This objective approach addresses one of the key challenges identified by Abgaz et al. [1] regarding the lack of standardized decomposition methods.

**c. The M2MDF Framework:** Abgaz et al. [1] propose a comprehensive framework, identifying four major phases of decomposition:

1. Analysis Phase: Understanding the existing monolith through static and dynamic analysis.
2. Decomposition Phase: Identifying service boundaries using various techniques.
3. Implementation Phase: Creating new microservices and establishing communication patterns.
4. Evaluation Phase: Assessing the quality of the decomposed architecture.

**d. Automated Decomposition Tools:** Kalia et al. [10] present Mono2Micro, a practical tool performing spatio-temporal decomposition using business use cases and runtime call relations. Their evaluation against four existing techniques demonstrated significant improvements in decomposition quality metrics, with practitioners rating the tool highly for creating functionally cohesive microservice partitions. This addresses the tool support gap identified by Abgaz et al. [1].

## FINDINGS — 2. Event-Driven Architecture Patterns *(printed p. 12)*

Event-driven architecture patterns emerge as a critical enabler for successful microservices migration in e-commerce systems.

**a. Event Sourcing and CQRS:** Ghosh [5] emphasizes event sourcing as fundamental to event-driven microservices, storing all changes as a sequence of events. This approach provides complete auditability and state reconstruction capabilities particularly valuable in e-commerce contexts where transaction history and regulatory compliance are critical. The Command Query Responsibility Segregation (CQRS) pattern complements event sourcing by separating read and write operations, improving performance for high-volume e-commerce workloads.

**b. Saga Pattern for Distributed Transactions:** The saga pattern addresses one of the most challenging aspects of microservices migration: managing distributed transactions. Ghosh [5] describes how sagas coordinate local transactions through events, essential for e-commerce workflows spanning multiple services such as order processing, payment authorization, and inventory updates. Kaloudis [6] reinforces this pattern's importance, particularly referencing Airbnb's transaction management implementation.

**c. Event Collaboration Patterns:** Event collaboration enables services to work together without direct dependencies, crucial for e-commerce scalability. Services publish domain events (e.g., "OrderPlaced", "PaymentProcessed") that other services can subscribe to and react accordingly, creating loosely coupled systems that can evolve independently [5].

## DISCUSSION — 1. Key Success Factors *(printed p. 15)*

The synthesis of findings reveals several critical success factors for migrating monolithic e-commerce systems to event-driven microservices:

**a. Domain-Driven Decomposition:** The most successful migrations employ Domain-Driven Design principles to identify natural service boundaries aligned with business capabilities [1][6]. E-commerce systems benefit from well-defined domain contexts, such as customer management, product catalog, and order processing.

**b. Incremental Migration Strategies:** The Strangler Fig Pattern emerges as the preferred approach for complex e-commerce systems, providing risk mitigation and continuous operation during transformation [5]. This incremental approach allows organizations to learn and adapt migration strategies based on early experiences.

**c. Event-Driven Communication:** Asynchronous event-driven patterns prove superior for e-commerce workloads, providing better scalability under high load conditions and improved fault tolerance [4][5]. The natural event-oriented nature of e-commerce business processes aligns well with event-driven architectures.

**d. Comprehensive Tooling:** Organizations benefit significantly from automated decomposition tools like Mono2Micro [10], which provide objective analysis of service boundaries and reduce subjective architectural decisions.

## DISCUSSION — 2. Performance Considerations *(printed p. 16)*

The performance analysis reveals nuanced trade-offs between synchronous and asynchronous communication patterns:

**a. Low Load Scenarios:** Synchronous patterns (gRPC, REST) provide better performance with lower latency [4].

**b. High Load Scenarios:** Asynchronous patterns (message queues) demonstrate superior throughput and availability [4].

**c. Availability:** Event-driven architectures provide better fault tolerance and system availability under stress [4].

These findings suggest that e-commerce systems should employ hybrid approaches, using synchronous communication for real-time user interactions and asynchronous patterns for backend processing and inter-service communication.
