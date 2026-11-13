# Phase 3 Verification & Simulation Report

## 1. Automated Verification Summary

All verification commands passed on the final Phase 3 build (`pixi.js@8.23.0`, Vite + TypeScript strict mode):

| Check | Command | Measured Result |
|---|---|---|
| Linter | `npm run lint` | Passed (0 errors, 0 warnings) |
| Typecheck | `npm run typecheck` | Passed (`tsc --noEmit`, strict mode) |
| Unit, Property, Vision & Sim Tests | `npm run test:coverage` | **18 / 18 files passed**, **69 / 69 tests passed** |
| Production Build | `npm run build` | Passed (`vite build`, 784 modules, 0 errors) |
| End-to-End Tests (Chromium) | `npm run test:e2e` | **5 / 5 Playwright specs passed**, **0 external runtime requests**, **0 page errors** |

### Selected Module Test Coverage (`vitest --coverage`)
- `src/ai/`: **94.91%** statements (`src/ai/hider.ts`: **95.87%**)
- `src/game/`: **92.47%** statements (`src/game/commitReveal.ts`: **89.58%**, `src/game/fsm.ts`: **92.56%**, `src/game/levels.ts`: **93.75%**)
- `src/game/rules/`: **95.00%** statements (`engine.ts`: **93.47%**, `schema.ts`: **96.29%**)
- `src/tracking/`: **97.26%** statements

---

## 2. Simulation Results: Levels 1–4 Uniformity (`10,000` Rounds Each)

Executed via `tests/sim/hiderSimulation.test.ts` (`chooseComputerHidingHand`) with both unbiased Web Crypto (`crypto.getRandomValues` rejection sampling) and a deterministic PRNG (`detRng`):

| Level | Name (fa) | Hands (`K`) | Rounds | Observed Counts (`crypto`) | `crypto` \(\chi^2\) | Deterministic \(\chi^2\) | Status |
|---|---|---:|---:|---|---:|---:|---|
| 1 | تازه‌کار | 2 | 10,000 | `[4993, 5007]` | `0.020` | `0.144` | Pass |
| 2 | آماتور | 2 | 10,000 | `[4994, 5006]` | `0.014` | `0.160` | Pass |
| 3 | حرفه‌ای | 2 | 10,000 | `[4981, 5019]` | `0.144` | `1.960` | Pass |
| 4 | تیمی | 4 | 10,000 | `[2553, 2522, 2443, 2482]` | `2.746` | `5.442` | Pass |
| 4 | تیمی | 6 | 10,000 | `[1642, 1641, 1657, 1751, 1678, 1631]` | `5.924` | `7.521` | Pass |

---

## 3. Simulation Results: Level 5 (`استاد گل‌گیر`) Against Biased Bot Guessers

Executed via `tests/sim/hiderSimulation.test.ts` over **2,000 simulated rounds per bot** for both `K = 2` (uniform baseline: `50.0%` hit prevention) and `K = 4` (uniform baseline: `75.0%` hit prevention):

### `K = 2` Hands (Bounded Max Deviation Limit: `0.180`)
| Bot Guesser | Rounds | Computer Hit-Prevention Rate | Human Hit Rate | Baseline | Max Deviation Observed | Mean Deviation Observed |
|---|---:|---:|---:|---:|---:|---:|
| `sticky` (90% repeat) | 2,000 | **65.20%** | 34.80% | 50.00% | `0.180` | `0.1790` |
| `alternating` (90% switch) | 2,000 | **64.65%** | 35.35% | 50.00% | `0.180` | `0.1795` |
| `frequency` (85% hand 0) | 2,000 | **62.15%** | 37.85% | 50.00% | `0.180` | `0.1792` |
| `wsls` (90% win-stay/lose-shift) | 2,000 | **63.60%** | 36.40% | 50.00% | `0.180` | `0.1791` |
| `uniform` (unbiased 50/50) | 2,000 | **51.15%** | 48.85% | 50.00% | `0.180` | `0.0745` |

### `K = 4` Hands (Bounded Max Deviation Limit: `0.1125`)
| Bot Guesser | Rounds | Computer Hit-Prevention Rate | Human Hit Rate | Baseline | Max Deviation Observed | Mean Deviation Observed |
|---|---:|---:|---:|---:|---:|---:|
| `sticky` | 2,000 | **85.15%** | 14.85% | 75.00% | `0.1125` | `0.1122` |
| `alternating` | 2,000 | **86.85%** | 13.15% | 75.00% | `0.1125` | `0.1122` |
| `frequency` | 2,000 | **85.15%** | 14.85% | 75.00% | `0.1125` | `0.1122` |
| `wsls` | 2,000 | **86.00%** | 14.00% | 75.00% | `0.1125` | `0.1122` |

All Level 5 runs strictly respected the bounded exploitation invariant (`|p_i - 1/K| <= min(0.18, 0.45 / K)`), keeping the computer non-deterministic even against maximally biased bots while lifting hit-prevention by **+12.15% to +15.20%** (`K = 2`) and **+10.15% to +11.85%** (`K = 4`) over uniform random play.

---

## 4. Phase 3 Acceptance Checklist

- **PixiJS v8 animated computer-hiding scenes (`src/render/pixiHidingScene.ts`, `pixi.js@8.23.0`)**: Pass — stylized vector-art closed fists, open palms, glowing gol pearl, 2/4/6-hand responsive layouts, shuffle/feint/uneven-tempo/misdirection choreography, reduced-motion support, and zero external image/font/CDN requests.
- **Cryptographic commit–reveal fairness (`src/game/commitReveal.ts`)**: Pass — 16-byte `crypto.getRandomValues` salt, Web Crypto `SHA-256(choice + ":" + salt)`, pre-guess fingerprint display, post-guess recomputation & `✓ تأیید شد` badge, Persian "این چطور کار می‌کند؟" explanation, and unit-tested tamper detection.
- **`traditional-team.json` & `custom.json` rule sets (`src/game/rules/`)**: Pass — schema-validated JSON rule sets; supports K=4/6 hands, `+2` direct gol, `+0` gol after pooch, `+1` hider on pooch-gol/miss/out-of-guesses, `maxGuessesPerRound: 3`, and full turn-ownership rules.
- **Levels 1–5, progression, HUD, and Level 3 contact enforcement (`src/game/levels.ts`, `src/game/fsm.ts`, `src/storage/progressionStore.ts`)**: Pass — all 5 levels implemented, Level 3 enforces `>= 1` hand contact (`NO_HAND_CONTACT` void rule), Level 5 runs Best-of-3 across `classic-1v1` and `traditional-team`, progression & numeric stats persist in IndexedDB, and FSM handles full `HUMAN_HIDES` <-> `COMPUTER_HIDES` turn alternation.
