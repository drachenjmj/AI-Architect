# E-Commerce Product Search: Read-Heavy Workloads, Indexing, Shards, and Scaling

- **Source title:** Amazon OpenSearch Service 101: T-shirt size your domain for e-commerce search
- **Publisher:** AWS Big Data Blog
- **Authors:** Abe Raghib, Aditya Challa, Harsh Bansal, Raaga N.G
- **Publication date:** 2026-02-16
- **Knowledge box:** 2
- **Domain:** e-commerce
- **Original file:** `Amazon OpenSearch Service 101 T-shirt size your domain for e-commerce search.pdf`
- **Source pages used:** PDF pages 1-4, 7-12 of 15

> Curated from the supplied AWS article. The retained material focuses on e-commerce search workload characteristics, catalog/index design, sharding/replicas, traffic spikes, scaling, and monitoring. Exact instance-family recommendations, T-shirt sizing tables, storage formulas, author biographies, comments, and AWS-console-specific operational detail were excluded.

## E-commerce search workload *(PDF pp. 1-2)*

E-commerce search must return fast and relevant product results while supporting capabilities such as full-text search, faceted search, tokenization, and autocomplete across large product catalogs and changing traffic levels.

AWS characterizes the workload as read-heavy, with advanced filtering and faceting, while product and catalog data continue to change through inventory updates, listing changes, pricing changes, and user activity such as clicks and reviews.

Sales and seasonal peaks increase query demand and require elasticity in compute and storage resources.

## Catalog ingestion and index organization *(PDF p. 2)*

Product and catalog updates can be ingested in bulk or through real-time streaming. Data is organized into logical indexes.

Index organization affects search, scalability, and operational flexibility. A small or medium catalog can use a comprehensive product index, while larger and more diverse catalogs can be split into category-specific indexes when separate mappings and scaling characteristics are useful.

## Shards, replicas, and distributed query execution *(PDF pp. 2-4)*

Indexes are divided into primary shards, with replicas used for availability and additional read capacity. Shards are distributed across nodes so that primary and replica copies of the same data are not placed together.

Search queries use a scatter-gather process:

1. A request reaches a coordinating node.
2. The query is distributed to the relevant primary or replica shards.
3. Each shard searches its local data and returns partial results.
4. The coordinating node merges and sorts the partial results into the final response.

The number and size of shards therefore affect performance, scaling, data distribution, coordination overhead, and query throughput.

## Read-heavy e-commerce search and replicas *(PDF pp. 7-8)*

For read-heavy e-commerce workloads, shard design should be monitored as indexes and catalogs grow. Replica shards can improve query throughput and fault tolerance because read requests can be served by additional shard copies across the cluster.

The source emphasizes balancing shard distribution and avoiding hot spots while keeping enough capacity for data growth and traffic variation.

## Scaling for traffic surges and catalog growth *(PDF p. 11)*

E-commerce platforms face unpredictable traffic surges and growing product catalogs.

The article describes two broad scaling approaches:

- **Vertical scaling:** increase the resources available to existing data nodes.
- **Horizontal scaling:** add more data nodes so indexing and search work can be distributed across more cluster capacity.

For traffic growth or increasing data volume, horizontal scaling can distribute load across nodes. Temporary additional replicas can also increase read throughput during high-traffic periods without requiring permanent capacity at the same level.

## Monitoring search quality and capacity *(PDF pp. 11-12)*

Search infrastructure should be monitored using workload and performance indicators rather than sized once and left unchanged.

The source highlights resource pressure, query latency, garbage-collection pressure, and request/thread-pool rejections as useful signals for detecting insufficient capacity or problematic queries. High-percentile query latency is specifically relevant for identifying slow user-facing searches.

The article recommends starting from workload requirements, testing with realistic catalog size and traffic, and iterating cluster/shard configuration as the business and workload grow.
