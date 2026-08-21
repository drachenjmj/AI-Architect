# E-Commerce Polyglot Persistence: Data Stores by Workload

- **Source title:** Polyglot Persistence with Azure Cosmos DB and Azure SQL Database
- **Publisher:** Microsoft Azure Architecture Center / Microsoft Learn
- **Year:** 2026
- **Knowledge box:** 2
- **Domain:** e-commerce
- **Original source:** https://learn.microsoft.com/en-us/azure/architecture/databases/idea/combine-relational-nosql
- **Source date:** 2026-05-29

> Deterministic source curation from the supplied Microsoft Learn Markdown. YAML/front matter, diagrams, Visio links, product-oriented component descriptions, unrelated industry use cases, cost/pricing material, and page furniture were removed. The retained source text is kept faithful to the original.

## Polyglot persistence and workload fit

Applications often handle diverse data workloads that have different characteristics. Structured, transactional data requires relational integrity and complex queries. Semistructured, rapidly changing, or high-volume data requires flexible schemas and horizontal scalability. Databases like Azure SQL Database and Azure Cosmos DB can support diverse workloads and multimodel requirements. However, in specific scenarios, organizations can achieve better outcomes by pairing SQL Database and Azure Cosmos DB in a polyglot persistence architecture.

Some workloads need the strict transactional guarantees and complex relational queries of a relational database, while other workloads need the flexible schemas and horizontal scalability of a NoSQL database. A polyglot approach assigns each workload to the platform designed for its dominant access pattern, rather than forcing a single platform to handle requirements it isn't optimized for.

This article describes a polyglot persistence approach that pairs SQL Database with Azure Cosmos DB so that you can configure each workload to use the database that best suits its requirements:

- SQL Database manages data that benefits from complex queries, relational integrity, and atomicity, consistency, isolation, and durability (ACID) transactions. SQL Database also supports multimodel capabilities like JSON, graph, spatial, and vector data, along with analytical workloads through columnstore indexes. Financial transaction records are well-suited to this database because they require consistent multiple-table transactions that span line items, inventory, and accounts.
- Azure Cosmos DB handles high-volume, schema-flexible, or globally distributed data that requires low-latency access and elastic scalability. E-commerce catalogs are well-suited to this database because their schemas evolve frequently and shoppers expect submillisecond reads regardless of region.

With a domain-driven microservices approach, each service uses the database that fits its data characteristics. Each microservice owns its private data store. This design prevents unintentional coupling between services and supports independent updates and deployments without coordinating changes across the system.

## Architecture — Data flow

1. Users and client applications connect to the system through Azure API Management, which provides a unified gateway for all back-end microservices.
2. API Management routes requests to the appropriate domain-driven microservices. Each microservice owns its data store independently.
3. Microservices that handle flexible-schema, high-volume, or globally distributed workloads, like user profiles, session state, product catalogs, and shopping carts, use Azure Cosmos DB. Azure Cosmos DB stores this data as JSON documents, provides single-digit millisecond response times, and scales horizontally.
4. Microservices that handle structured, transactional workloads, like order management, inventory, and payments, use SQL Database. SQL Database provides full ACID compliance, complex query support, and relational integrity for these operations.
5. Some microservices communicate with each other to fulfill cross-domain data requirements. For example, the shopping cart service queries the user session service for session context, and both the inventory and order management services interact with the product catalog service for product information. These calls between microservices use service APIs rather than directly accessing another service's database, which preserves data ownership boundaries.

## Scenario details and trade-offs

Applications often combine transactional operations with high-volume or rapidly evolving data. Structured, transactional data requires relational integrity and complex queries. Semistructured, rapidly evolving, or high-volume data requires flexible schemas and horizontal scalability. With a polyglot persistence approach, you can assign each workload to the database technology that best matches its requirements.

Domain-driven microservices enforce clear data-ownership boundaries, so each service manages its own data store independently. This approach introduces challenges like data redundancy across stores and eventual consistency between services. A polyglot architecture also increases operational complexity compared to a single database platform. Your team must develop and maintain expertise across both database technologies, which increases training and operational overhead.

The following advantages help offset those challenges:

- **Independent scalability:** Each database scales according to its workload. Azure Cosmos DB handles read/write bursts of millions of operations per second with guaranteed low latency. SQL Database provides both provisioned compute for steady workloads and serverless autoscaling, with scale-to-zero capabilities for unpredictable workloads.
- **Appropriate data modeling:** SQL Database provides relational schemas, foreign keys, and joins for data that has well-defined relationships. Azure Cosmos DB provides schema-agnostic storage with automatic indexing for data that evolves frequently.

## When to use each service

SQL Database and Azure Cosmos DB have overlapping capabilities. Both services can store JSON and deliver low-latency responses when configured appropriately. The decision depends on which service's primary design strengths align with your workload's dominant access patterns:

- Choose Azure Cosmos DB when your workload primarily requires schema-flexible document storage, automatic multiregion distribution with guaranteed single-digit millisecond reads, or elastic horizontal scaling across partitions. These characteristics are native strengths of Azure Cosmos DB and represent its optimized path.
- Choose SQL Database when your workload primarily requires enforced relational integrity across tables, multistatement ACID transactions, or complex joins and aggregations. These characteristics are native strengths of Azure SQL Database and represent its optimized path.

When a workload's requirements don't clearly favor one service, evaluate the dominant access pattern rather than secondary capabilities. For example, SQL Database supports JSON storage, but a workload that consists primarily of schema-flexible JSON documents with high-write throughput better suits Azure Cosmos DB.

## Potential use case — E-commerce and retail

- **E-commerce and retail:** Applications that use SQL Database for customer accounts, orders, and inventory, and Azure Cosmos DB for product catalogs, personalization, and real-time session data.
