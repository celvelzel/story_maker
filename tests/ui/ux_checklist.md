# UX Verification Checklist

StoryWeaver UX 验证清单。
用于手动验证前端优化的完成情况。

## Phase 1: UX Quick Wins

### Visual Consistency
- [ ] All colors use theme tokens from `src/ui/theme_tokens.py`
- [ ] No hardcoded color values in component files
- [ ] Consistent spacing using token values
- [ ] Consistent border radius using token values
- [ ] Consistent shadow styles using token values

### Feedback States
- [ ] Loading states show spinner with clear message
- [ ] Error states show error message with retry hint
- [ ] Success states show success confirmation
- [ ] Empty states show helpful message with action hint
- [ ] Warning states show warning with action guidance

### Accessibility
- [ ] `prefers-reduced-motion` media query disables animations
- [ ] Focus outlines are visible on interactive elements
- [ ] Color contrast meets WCAG AA standards
- [ ] Keyboard navigation works for all interactive elements

### Interaction Feedback
- [ ] Button clicks provide visual feedback
- [ ] Form inputs show validation feedback
- [ ] Loading states prevent duplicate submissions
- [ ] Error states allow retry without page refresh

## Phase 2: Maintainability Refactor

### Code Structure
- [ ] `app.py` is under 300 lines
- [ ] Sidebar rendering is in `src/ui/layout/sidebar_view.py`
- [ ] Main area rendering is in `src/ui/layout/main_view.py`
- [ ] Chat section is in `src/ui/sections/chat_section.py`
- [ ] KG section is in `src/ui/sections/kg_section.py`
- [ ] Evaluation section is in `src/ui/sections/evaluation_section.py`

### Session State
- [ ] Session state is initialized via `session_contract.py`
- [ ] All session keys have defined defaults
- [ ] Session validation catches type mismatches
- [ ] Session persistence works across page refreshes

### Style Injection
- [ ] All CSS is injected via `style_injector.py`
- [ ] No large inline style blocks in `app.py`
- [ ] Theme tokens are used consistently
- [ ] Reduced motion support is included

### Test Coverage
- [ ] All UI smoke tests pass
- [ ] Feedback state tests pass
- [ ] Session persistence tests pass
- [ ] Visual consistency tests pass
- [ ] KG sidebar flow tests pass

## Manual Verification Steps

### 1. Landing Page
1. Open the application
2. Verify hero banner displays correctly
3. Verify "Start New Game" button is visible
4. Verify genre input field is visible
5. Verify sidebar shows empty KG state

### 2. Game Start
1. Enter a genre (e.g., "fantasy")
2. Click "Start New Game"
3. Verify loading spinner appears
4. Verify game initializes without errors
5. Verify opening narration appears in chat
6. Verify KG panel shows initial graph

### 3. Story Interaction
1. Enter a free-text action
2. Verify loading state appears
3. Verify story response appears
4. Verify KG updates with new entities
5. Verify consistency score updates
6. Verify option buttons appear (if generated)

### 4. Option Selection
1. Click an option button
2. Verify loading state appears
3. Verify story continues based on option
4. Verify KG updates accordingly
5. Verify consistency tracking continues

### 5. Evaluation
1. Click "Run Evaluation"
2. Verify loading spinner appears
3. Verify evaluation results display
4. Verify metrics are calculated correctly
5. Verify LLM judge scores appear

### 6. Session Persistence
1. Refresh the page
2. Verify game state is restored
3. Verify chat history is preserved
4. Verify KG visualization is restored
5. Verify evaluation results are preserved

### 7. Error Handling
1. Submit action without starting game
2. Verify warning message appears
3. Verify retry hint is displayed
4. Verify no application crash

### 8. Responsive Design
1. Resize browser window
2. Verify layout adapts to smaller screens
3. Verify sidebar collapses appropriately
4. Verify chat remains readable
5. Verify KG visualization scales

## Evidence Collection

### Screenshots
- [ ] Landing page
- [ ] Game in progress
- [ ] KG visualization
- [ ] Evaluation results
- [ ] Error state
- [ ] Mobile view

### Test Results
- [ ] `pytest tests/ui/scenarios/ -v` output
- [ ] All tests pass (except LLM-dependent skips)
- [ ] No new errors introduced

### Code Review
- [ ] `app.py` line count < 300
- [ ] No inline style blocks > 10 lines
- [ ] All imports are used
- [ ] No TODO/FIXME comments in new code

## Sign-off

- [ ] Phase 1 UX improvements verified
- [ ] Phase 2 refactor completed
- [ ] All tests passing
- [ ] Manual verification completed
- [ ] Ready for user approval
