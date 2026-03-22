# Frontend UX-First Two-Phase Optimization Plan

## Plan Metadata
- Project: `story_maker` (Streamlit frontend)
- Date: 2026-03-22
- Planner: Prometheus
- Source draft: `.sisyphus/drafts/frontend-module-optimization.md`

## Objective
Deliver a **UX-first frontend optimization** for the existing Streamlit application, while executing a **second-phase maintainability refactor** that improves iteration speed without changing backend behavior.

## Confirmed Decisions
1. Priority order: **UX first**, **maintainability refactor second**.
2. UI change intensity: **medium-to-heavy redesign**.
3. UX target focus:
   - A: readability & information hierarchy
   - C: interaction feedback clarity (loading/error/success/empty states)
   - D: visual polish & consistency
4. Delivery model: **Two phases**
   - Phase 1: quick wins (1–2 weeks)
   - Phase 2: structural optimization
5. KG panel scope: **interaction/layout redesign only**, no KG engine rewrite.
6. Style direction: keep cyberpunk-dark identity, upgrade quality and consistency.
7. Verification for quick wins: manual checklist + UI smoke scripts.

## Scope Boundaries

### In Scope
- Streamlit frontend structure and interaction flows in `app.py` and new frontend-facing modules.
- UX consistency for main chat flow and KG sidebar panel.
- Visual system hardening (tokens/rules/components) while preserving dark cyberpunk tone.
- Frontend smoke tests and quality gates needed to safely support refactoring.

### Out of Scope
- Rewriting frontend framework (no React/Vue migration).
- Modifying KG core engine algorithms, extraction logic, or backend API semantics.
- Changing game narrative logic or LLM prompt strategy as a primary objective.

## Guardrails (Metis-Informed)
1. No backend engine behavior change under “UI optimization” label.
2. No framework migration or custom Streamlit component platform in this plan.
3. Style changes must be centralized; no new large inline-style blocks.
4. Phase 2 cannot proceed until Phase 1 smoke checks are stable.
5. Refactor must preserve existing user-visible core flows.

## Assumptions / Defaults Applied
1. UI smoke framework default: **`pytest` + Streamlit `AppTest`** as primary path.
2. Optional browser E2E is deferred unless smoke gaps remain.
3. Device baseline: desktop-first plus basic narrow-width validation.
4. Accessibility baseline in this round: keyboard focus clarity + contrast checks + reduced-motion support for custom animations.

## Delivery Structure
- **Phase 1 (UX Quick Wins, 1–2 weeks)**: improve clarity, feedback, and visual consistency with minimal architectural risk.
- **Phase 2 (Maintainability Refactor)**: decompose monolith and centralize state/style structure for long-term UX velocity.

## Detailed Task Plan

### Phase 1 — UX Quick Wins (Primary)

#### Task P1-1 — Establish UX Baseline & Scenario Matrix
- **Goal**: freeze current UX baseline before redesign to avoid subjective regressions.
- **Primary files to create/update**:
  - `tests/ui/scenarios/test_chat_primary_flow.py`
  - `tests/ui/scenarios/test_kg_sidebar_flow.py`
  - `tests/ui/README.md` (scenario definitions and expected outcomes)
- **Implementation decisions**:
  1. Use `pytest + streamlit.testing.v1.AppTest` for smoke-level interaction checks.
  2. Define minimum scenario set:
     - Start game from landing state
     - Submit one free-text action and observe narrative response
     - Select one generated option button and observe state transition
     - Open/observe KG panel content refresh after turn
  3. Record current behavior snapshot (pass/fail + notes) as baseline evidence.
- **QA scenarios (must pass to complete task)**:
  - Run `pytest tests/ui/scenarios -q` and confirm tests execute end-to-end.
  - Baseline file documents expected loading/error/success behavior for each scenario.

#### Task P1-2 — Centralize Visual Tokens & Style Injection
- **Goal**: remove style fragmentation and create a single source of truth for visual consistency.
- **Primary files to create/update**:
  - `src/ui/theme_tokens.py` (color/spacing/typography/radius/shadow/motion tokens)
  - `src/ui/style_injector.py` (single style injection entrypoint)
  - `app.py` (replace large inline style blocks with injector call)
- **Implementation decisions**:
  1. Preserve cyberpunk-dark identity; improve hierarchy by tokenized contrast levels (surface-1/2/3, accent, semantic states).
  2. Add motion guard for reduced-motion preference in injected CSS.
  3. Do not change gameplay/business logic during style extraction.
- **QA scenarios (must pass to complete task)**:
  - Smoke scenarios from P1-1 continue passing after style centralization.
  - Manual check: buttons, inputs, cards, sidebar panels use consistent spacing/typography scale.

#### Task P1-3 — Standardize Feedback States (Loading/Error/Success/Empty)
- **Goal**: eliminate ambiguous UI status and reduce user uncertainty.
- **Primary files to create/update**:
  - `src/ui/feedback.py` (uniform rendering helpers)
  - `app.py` (replace scattered ad-hoc feedback calls)
  - `tests/ui/scenarios/test_feedback_states.py`
- **Implementation decisions**:
  1. Define canonical feedback patterns for:
     - request in progress (loading)
     - recoverable failure (error with retry guidance)
     - successful completion confirmation
     - empty/no-data informational state
  2. Apply these patterns to main chat actions and KG panel update points.
  3. Ensure textual guidance explains next action (retry / continue / refresh).
- **QA scenarios (must pass to complete task)**:
  - Trigger each state in controlled test path and assert expected feedback container appears.
  - Manual check confirms feedback copy is concise and action-oriented.

### Phase 2 — Maintainability Refactor (Secondary)

#### Task P2-1 — Extract Session State Contract
- **Goal**: prevent rerun-related state drift while enabling modularization.
- **Primary files to create/update**:
  - `src/ui/session_contract.py` (state keys, defaults, validation helpers)
  - `app.py` (replace duplicated key initialization/restore logic)
  - `tests/ui/scenarios/test_session_persistence.py`
- **Implementation decisions**:
  1. Define explicit state schema for all frontend-critical keys (engine/history/options/kg_html/nlu_debug/etc.).
  2. Expose single initialization path and single restore path.
  3. Add defensive fallbacks for missing/corrupt saved session fields.
- **QA scenarios (must pass to complete task)**:
  - Refresh/reload simulation preserves essential chat and panel context.
  - Missing optional session fields no longer cause fatal render failures.

#### Task P2-2 — Decompose `app.py` into UI Modules
- **Goal**: separate concerns so UX iteration no longer requires monolithic edits.
- **Primary files to create/update**:
  - `src/ui/layout/main_view.py`
  - `src/ui/layout/sidebar_view.py`
  - `src/ui/sections/chat_section.py`
  - `src/ui/sections/kg_section.py`
  - `src/ui/sections/evaluation_section.py`
  - `app.py` (entry orchestration only)
- **Implementation decisions**:
  1. Keep functional equivalence: no gameplay behavior change.
  2. `app.py` becomes orchestration shell calling section renderers.
  3. Shared helpers (feedback/theme/session contract) are imported instead of duplicated.
- **QA scenarios (must pass to complete task)**:
  - All P1 scenario tests pass unchanged.
  - Manual walkthrough confirms no missing controls/content after decomposition.

#### Task P2-3 — KG Sidebar Interaction Refinement Without Engine Changes
- **Goal**: improve KG panel readability and actionability within UI-only boundary.
- **Primary files to create/update**:
  - `src/ui/sections/kg_section.py`
  - `src/knowledge_graph/visualizer.py` (presentation-level hook points only; no algorithm changes)
  - `tests/ui/scenarios/test_kg_sidebar_flow.py` (extend assertions)
- **Implementation decisions**:
  1. Reorganize sidebar information hierarchy: title/status/context/actions order.
  2. Add clear refresh/loading/error/empty micro-states for graph panel container.
  3. Keep data source and KG computation untouched.
- **QA scenarios (must pass to complete task)**:
  - KG panel still renders with existing data pipeline.
  - Sidebar state transitions are explicit and test-assertable.

#### Task P2-4 — UX Consistency Hardening & Regression Gate
- **Goal**: lock in redesign quality and prevent future drift.
- **Primary files to create/update**:
  - `tests/ui/scenarios/test_visual_consistency_rules.py`
  - `tests/ui/ux_checklist.md`
  - `.github/workflows/frontend-smoke.yml` (if repository policy allows new CI workflow)
- **Implementation decisions**:
  1. Encode consistency checks (token usage, forbidden inline-style reintroduction, feedback component usage).
  2. Add CI smoke execution for UI scenario subset where feasible.
  3. If CI workflow addition is blocked, define local pre-merge command contract as fallback.
- **QA scenarios (must pass to complete task)**:
  - CI/local smoke command executes deterministically.
  - Consistency checklist is attached to verification report with evidence references.

## Risks & Mitigations
- CSS overreach risk → constrain selectors to scoped wrappers and validate key pages.
- Streamlit rerun/state instability → central state guards + smoke checks before/after each refactor slice.
- Visual inconsistency regression during split work → tokenized design rules and checklist gate.
- KG panel usability regressions → panel-specific scenario checks in every validation pass.

## Acceptance Criteria (Outcome Level)

### Phase 1 Must-Haves
1. Primary user flow readability improved with explicit hierarchy and sectioning.
2. Loading/error/success/empty feedback patterns are standardized in core UX paths.
3. KG panel interaction readability and layout clarity improved without backend logic changes.
4. UX manual checklist passes and UI smoke scripts pass for agreed scenarios.

### Phase 2 Must-Haves
1. `app.py` no longer acts as monolithic all-concern file.
2. State initialization/mutation entry points are centralized and test-covered.
3. Style/theming rules are centralized and reusable.
4. Phase 1 UX outcomes remain stable after refactor (no net regression on smoke scenarios).

## Final Verification Wave (Explicit User Approval Required)
1. Produce a verification report with:
   - UX checklist results (pass/fail + evidence)
   - Smoke script results (pass/fail + evidence)
   - Scope compliance audit (confirm no out-of-scope backend changes)
2. Present unresolved items (if any) with severity and remediation proposal.
3. **Stop and request explicit user confirmation (“okay”) before marking work complete.**

## Handoff
- Execution command after approval: `/start-work`
- Optional high-accuracy path: Momus review loop before execution.
