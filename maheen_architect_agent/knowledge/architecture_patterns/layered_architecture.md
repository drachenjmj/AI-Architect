# Layered Architecture

## When to Use

Use this pattern when the system is simple, stable, and can be separated into clear layers.

## Good Fit For

- Simple business applications
- Internal dashboards
- CRUD systems
- Systems with limited real-time needs

## Typical Components

- User Interface Layer
- Application Layer
- Business Logic Layer
- Data Access Layer
- Database

## Strengths

- Easy to understand
- Easy to build
- Good for small teams
- Clear separation of responsibilities

## Risks

- Can become hard to scale if the system grows
- Not ideal for high real-time event processing
- Changes in one layer may affect other layers

## Selection Clues

Choose this pattern if the context mentions:
- simple system
- internal tool
- dashboard
- low complexity
- limited users
- stable requirements