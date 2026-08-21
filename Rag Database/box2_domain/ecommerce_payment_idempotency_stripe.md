# E-Commerce Payment/API Retries: Idempotent Requests

- **Source title:** Idempotent requests
- **Publisher:** Stripe Docs
- **Knowledge box:** 2
- **Domain:** e-commerce
- **Original file:** `idempotent_requests.md`
- **Original source:** https://docs.stripe.com/api/idempotent_requests?lang=curl

> Curated from the supplied Stripe documentation. The retained material focuses on the idempotency behavior relevant to safe retries in payment and other write operations. Example credentials/CLI boilerplate were removed. Stripe-specific semantics remain explicitly identified as provider-specific details.

## Safe retries without duplicate operations

Stripe's API supports idempotency so that a request can be retried after a connection or transport failure without accidentally performing the same operation twice.

When creating or updating an object, the client supplies an idempotency key. If the same request must be retried, reusing that key allows Stripe to recognize the retry rather than treating it as an independent operation.

A client generates the idempotency key. Stripe recommends a unique value such as a UUID v4 or another sufficiently random string. Sensitive data such as email addresses or personal identifiers should not be used as idempotency keys.

## Result reuse and request consistency

Stripe saves the status code and response body of the first request made for a given idempotency key once endpoint execution begins. Subsequent requests with the same key return the same stored result, including server-error responses.

When a key is reused, Stripe compares the incoming parameters with the original request parameters and rejects inconsistent reuse. This prevents the same idempotency key from being accidentally applied to a different operation.

Results are not stored when request validation fails before endpoint execution begins, or when the request conflicts with another concurrently executing request. Those requests can be retried because no endpoint execution result was persisted.

## Stripe-specific operational details

- All `POST` requests accept idempotency keys.
- Idempotency keys have no effect on `GET` or `DELETE` requests because those request types are idempotent by definition in Stripe's API semantics.
- Stripe can automatically remove stored keys after they are at least 24 hours old; reuse after pruning is treated as a new request.
- Idempotency keys can be up to 255 characters long.
