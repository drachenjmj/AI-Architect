# E-Commerce Inventory Concurrency: Optimistic Locking

- **Source title:** Optimistic locking with version number
- **Publisher:** Amazon Web Services — DynamoDB Developer Guide
- **Knowledge box:** 2
- **Domain:** e-commerce
- **Original file:** `BestPractices_OptimisticLocking.md`
- **Original source:** https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/BestPractices_OptimisticLocking.html

> Curated from the supplied AWS documentation. The retained material focuses on the concurrency pattern and the e-commerce inventory example. SDK-specific annotations and full implementation code were removed.

## Conflict detection at write time

Optimistic locking detects write conflicts rather than preventing concurrent access up front. Each item carries a version attribute that is incremented with every successful update.

When an application updates an item, it includes a condition that checks whether the stored version still matches the version that the application previously read. If another process changed the item in the meantime, the condition fails and the write is rejected with a conflict instead of silently overwriting the newer state.

## When optimistic locking fits

AWS describes optimistic locking as a good fit when:

- multiple users or processes might update the same item, but conflicts are relatively infrequent;
- retrying a failed write is inexpensive;
- the application should avoid the overhead and complexity of distributed locks.

The documented examples explicitly include e-commerce inventory updates.

## E-commerce inventory model

AWS illustrates the pattern with an inventory item containing:

- `ItemId` — the item identifier;
- `Version` — an integer version number;
- `QuantityLeft` — the remaining inventory quantity.

A new item starts with an initial version. Each successful update changes the inventory state and increments the version.

The update flow is:

1. Read the current inventory item and its version.
2. Attempt the quantity update with a condition that the stored version still equals the version just read.
3. If the condition succeeds, update `QuantityLeft` and increment `Version`.
4. If the condition fails because another process updated the item first, read the current state again and retry according to a bounded retry policy.

## Trade-offs

### Retry overhead under contention

As concurrency and contention increase, conflicts become more frequent. That can increase retries and write cost.

### Application complexity

The application must maintain version information and handle failed conditional writes and retries. The pattern avoids distributed locks, but it does not eliminate concurrency-handling logic.
