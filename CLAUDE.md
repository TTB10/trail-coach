# Trail Coach — Claude Code instructions

This file gives context and conventions to Claude Code when working on this project.

---

## Project overview

Trail Coach is an AI-powered ultra-trail coaching platform. It analyzes a runner's training history (synced from Strava) and predicts race performance based on physiological load modeling, GPX-based course analysis, and adaptive pacing strategy.

The project is a TypeScript-based migration from a Streamlit prototype to a production-grade SaaS architecture.

## Stack

- **Backend:** FastAPI (Python 3.12) + Pydantic v2 + SQLAlchemy 2.0
- **Database:** PostgreSQL (hosted on Supabase) + Alembic migrations
- **Auth:** Supabase Auth (JWT verified server-side)
- **Frontend:** Next.js 16 (App Router) + TypeScript + Tailwind CSS + shadcn/ui
- **Data fetching:** TanStack Query (React Query)
- **External APIs:** Strava API (OAuth + activities sync)
- **Hosting:** Vercel (frontend), Railway (backend), Supabase (DB and storage)

## Repository structure (monorepo)

​```
trail-coach/
├── backend/          # FastAPI application
│   ├── main.py       # Entry point (will move to app/ later)
│   ├── pyproject.toml
│   └── uv.lock
├── frontend/         # Next.js application
│   ├── src/app/      # App Router pages
│   ├── src/components/
│   └── package.json
├── README.md
├── CLAUDE.md         # This file
└── .gitignore
​```

## Common commands

### Backend

​```bash
cd backend
uv run fastapi dev main.py     # Start dev server with hot reload (http://localhost:8000)
uv add <package>                # Add a dependency
uv run pytest                   # Run tests (when set up)
uv run ruff check .             # Lint
uv run ruff format .            # Format
uv run mypy .                   # Type check
​```

### Frontend

​```bash
cd frontend
npm run dev                     # Start dev server (http://localhost:3000)
npm run build                   # Production build
npm run lint                    # ESLint
npm install <package>           # Add a dependency
​```

### Git

We use Conventional Commits:
- `feat(scope): ...` for new features
- `fix(scope): ...` for bug fixes
- `chore(scope): ...` for tooling, deps, config
- `docs(scope): ...` for documentation
- `refactor(scope): ...` for code restructuring
- `test(scope): ...` for tests

Scope is usually `backend`, `frontend`, or `infra`.

## Code conventions

### Python (backend)

- Python 3.12+ syntax (use `dict[str, str]` not `Dict[str, str]`, `X | None` not `Optional[X]`)
- All functions must have type hints on parameters and return values
- Pydantic v2 models for request/response schemas (in `app/schemas/`)
- SQLAlchemy 2.0 declarative style with typed columns (`Mapped[...]`)
- Pure business logic goes in `app/domain/` (no HTTP, no DB) — easy to unit test
- API endpoints in `app/api/v1/` thin: validate input, call services, return response
- Services in `app/services/` orchestrate domain + DB
- Use `httpx.AsyncClient` for external HTTP calls (never `requests`)
- Async by default for I/O endpoints

### TypeScript (frontend)

- Strict mode enabled (no `any` unless justified with a comment)
- Functional React components only (no class components)
- Use shadcn/ui components when possible, customize via Tailwind
- Server Components by default, mark `'use client'` only when needed (interactivity, browser APIs)
- TanStack Query for all backend calls, with typed hooks generated from OpenAPI
- File naming: `kebab-case` for files, `PascalCase` for components
- Co-locate component-specific types with the component

### General

- Never commit secrets. Use `.env.local` (frontend) and `.env` (backend), both gitignored
- `.env.example` files at the root of `backend/` and `frontend/` document required vars
- Prefer composition over inheritance
- Small files (< 300 lines), single-responsibility
- Tests for business logic (especially physiological calculations)

## Domain knowledge — physiological model

The original Streamlit app implements a custom prediction model. Key formulas to preserve:

- **Effort équivalent:** `effort = distance_km + (denivele_positif_m / 100)`, with a small bonus for D-
- **TRIMP (Karvonen):** `duration_min * ratio * 0.64 * exp(1.92 * ratio)` where `ratio = (HR - HR_rest) / (HR_max - HR_rest)`
- **CTL (chronic training load):** EWMA of TRIMP with alpha = 2/43 (~42-day fitness)
- **ATL (acute training load):** EWMA of TRIMP with alpha = 2/8 (~7-day fatigue)
- **TSB (form):** CTL.shift(1) - ATL.shift(1)
- **ACWR (injury risk):** sum(7d charge) / (sum(28d charge) / 4)
- **Riegel-style time projection:** `T2 = T1 * (effort2 / effort1)^k` where k is dynamically adjusted based on experience, fitness deficit, gradient ratio, etc.
- **Night penalty:** +15% time between 8pm and 6am
- **Heart rate zones:** Karvonen percentages (Z1: <60% reserve, Z2: 60-75%, Z3: 75-85%, Z4: 85-95%, Z5: >95%)

These calculations live in `backend/app/domain/` and are pure functions — no I/O, fully unit-tested.

## Reference: original Streamlit prototype

The Streamlit prototype lives separately and is the reference implementation for the prediction model. When porting features, preserve numerical equivalence and add unit tests pinning the expected outputs against known inputs.

## Project status

🚧 Bootstrapping phase. Current state:
- Backend: FastAPI app with `/` and `/health` endpoints — no domain logic yet
- Frontend: Next.js app with default landing page — no real UI yet
- Database: Supabase project created — no schema yet
- Auth: not yet implemented
- Strava integration: not yet implemented

Next milestones (Sprint 1):
1. Move backend to proper `app/` structure
2. Set up Alembic and connect to Supabase Postgres
3. Implement Supabase Auth verification middleware
4. Port physiological calculations to `app/domain/` with tests
5. Replace Next.js default page with a real layout + login screen