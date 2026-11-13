import { OverallFrequencyExpert, RecentFrequencyExpert } from './frequency';
import { RepeatAfterSuccessExpert, SwitchAfterCaughtExpert } from './habituation';
import { MarkovWinStayLoseShiftExpert } from './markov';
import { NGramExpert } from './ngram';
import type { BehaviorExpert } from './types';

export * from './types';
export * from './frequency';
export * from './ngram';
export * from './markov';
export * from './habituation';

export function createDefaultBehaviorExperts(): BehaviorExpert[] {
  return [
    new OverallFrequencyExpert(),
    new RecentFrequencyExpert(6),
    new NGramExpert(1),
    new NGramExpert(2),
    new NGramExpert(3),
    new MarkovWinStayLoseShiftExpert(),
    new RepeatAfterSuccessExpert(),
    new SwitchAfterCaughtExpert(),
  ];
}
