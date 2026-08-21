# E-Commerce Limited Inventory: Checkout Reservation and Expiry

- **Source title:** Manage limited inventory
- **Publisher:** Stripe Docs
- **Knowledge box:** 2
- **Domain:** e-commerce
- **Original file:** `Manage limited inventory.pdf`
- **Original source:** https://docs.stripe.com/payments/checkout/managing-limited-inventory
- **Source pages used:** PDF pages 1-3 of 4

> Curated from the supplied Stripe documentation. Only the limited-inventory reservation, Checkout Session expiry, and inventory-release behavior is retained. Code snippets, account/UI chrome, help links, and the empty/footer page were excluded.

## Prevent long-held inventory reservations *(PDF p. 1)*

For limited-inventory businesses, customers should not be able to reserve scarce items for a long time without completing a purchase. Stripe documents expiring a pending Checkout Session as a way to end the pending sale and make the reserved inventory available again.

Checkout supports both manual and timed session expiry. When a Checkout Session expires, its status changes to `expired`.

## Manual and timed expiry *(PDF p. 2)*

An open Checkout Session can be expired immediately using Stripe's expiry endpoint when the pending purchase should be cancelled.

A Checkout Session can also be created with an `expires_at` timestamp. Stripe documents an allowed expiry window between 30 minutes and 24 hours after the current time. If `expires_at` is not set, the default value is 24 hours after the current time.

These values are Stripe-specific implementation limits; the general domain concern is that a reservation must have a bounded lifetime rather than holding limited inventory indefinitely.

## Return expired reservations to inventory *(PDF p. 3)*

When a Checkout Session expires, Stripe sends the `checkout.session.expired` event.

A webhook endpoint can listen for this event and return to inventory the items that were reserved in the expired session. This links checkout-session lifecycle management to release of temporarily reserved stock.
