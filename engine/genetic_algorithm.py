"""
Genetic Algorithm optimizer for lottery number combination optimization.

Treats each set of numbers as a Chromosome and evolves populations
through selection, crossover, and mutation to maximize fitness.

Fitness Function considers:
- Number probability scores
- Odd/Even structure balance
- Sum range合理性
- Span distribution
- AC value
- Consecutive probability
- Hot/Cold number balance
- Historical repeat penalty
- Diversity score
- Structural stability
"""
import math
import random
from collections import Counter
from typing import List, Tuple, Dict, Optional, Any

import numpy as np


class GAOptimizer:
    """
    Genetic Algorithm for lottery number combination optimization.

    Parameters
    ----------
    cfg : LotteryConfig
    population_size : int (default=200)
    generations : int (default=100)
    elite_ratio : float (default=0.1) - top fraction to keep unchanged
    crossover_rate : float (default=0.8)
    mutation_rate : float (default=0.15)
    tournament_size : int (default=5)
    """

    def __init__(
        self,
        cfg,
        population_size: int = 200,
        generations: int = 100,
        elite_ratio: float = 0.1,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.15,
        tournament_size: int = 5,
    ):
        self.cfg = cfg
        self.population_size = population_size
        self.generations = generations
        self.elite_ratio = elite_ratio
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = min(tournament_size, population_size)

        self.main_range = list(range(cfg.main_min, cfg.main_max + 1))
        self.sub_range = list(range(cfg.sub_min, cfg.sub_max + 1))

        # History for fitness tracking
        self.fitness_history = []  # [best_fitness, avg_fitness, diversity] per generation
        self._rng = random.Random(42)

    # ====================================================================
    # CHROMOSOME REPRESENTATION
    # ====================================================================

    def _random_chromosome(self) -> Dict:
        """Generate a random valid chromosome (number combination)."""
        return {
            "main": sorted(self._rng.sample(self.main_range, self.cfg.main_count)),
            "sub": sorted(self._rng.sample(self.sub_range, self.cfg.sub_count)),
        }

    # ====================================================================
    # FITNESS FUNCTION
    # ====================================================================

    def fitness(self, chromosome: Dict, prob_scores: Optional[Dict[str, np.ndarray]] = None) -> float:
        """
        Compute fitness score for a chromosome.

        Higher = better combination. Combines multiple criteria.

        Parameters
        ----------
        chromosome : dict with 'main', 'sub' lists
        prob_scores : optional dict with 'main_probs' and 'sub_probs' arrays

        Returns
        -------
        fitness : float
        """
        main = chromosome["main"]
        sub = chromosome["sub"]
        score = 0.0

        # 1. Probability score (20%) - prefer numbers with high model probability
        if prob_scores and "main_probs" in prob_scores:
            mp = prob_scores["main_probs"]
            sp = prob_scores["sub_probs"]
            main_idx = [n - self.cfg.main_min for n in main]
            sub_idx = [n - self.cfg.sub_min for n in sub]
            prob_score = sum(mp[i] for i in main_idx) + sum(sp[i] for i in sub_idx)
            # Normalize: ~0.5 is average, ~1.0 is excellent
            expected = (self.cfg.main_count / len(mp)) + (self.cfg.sub_count / len(sp))
            score += 20.0 * min(prob_score / max(expected, 0.001), 2.0)

        # 2. Odd/Even balance (15%)
        main_odds = sum(1 for n in main if n % 2 == 1)
        sub_odds = sum(1 for n in sub if n % 2 == 1)

        # Ideal: roughly 50/50 split
        main_oe_ideal = self.cfg.main_count / 2
        sub_oe_ideal = self.cfg.sub_count / 2
        oe_penalty = abs(main_odds - main_oe_ideal) * 3 + abs(sub_odds - sub_oe_ideal) * 3
        score += 15.0 * math.exp(-oe_penalty / self.cfg.main_count)

        # 3. Sum range合理性 (15%)
        main_sum = sum(main)
        sub_sum = sum(sub)

        # Expected sum = average number * count
        main_avg = (self.cfg.main_min + self.cfg.main_max) / 2
        sub_avg = (self.cfg.sub_min + self.cfg.sub_max) / 2
        expected_main_sum = main_avg * self.cfg.main_count
        expected_sub_sum = sub_avg * self.cfg.sub_count

        # Allow 20% deviation
        main_sum_dev = abs(main_sum - expected_main_sum) / expected_main_sum
        sub_sum_dev = abs(sub_sum - expected_sub_sum) / max(expected_sub_sum, 1)
        sum_penalty = main_sum_dev + sub_sum_dev * 0.5
        score += 15.0 * math.exp(-sum_penalty * 8)

        # 4. Span distribution (10%)
        main_span = max(main) - min(main)
        expected_span = self.cfg.main_max - self.cfg.main_min
        span_ratio = main_span / max(expected_span, 1)
        # Ideal span: cover ~60-80% of range
        ideal_span = 0.7
        span_penalty = abs(span_ratio - ideal_span)
        score += 10.0 * math.exp(-span_penalty * 10)

        # 5. AC value (10%)
        ac = self._ac_value(main)
        expected_ac = self._expected_ac()
        ac_penalty = abs(ac - expected_ac) / max(expected_ac, 1)
        score += 10.0 * math.exp(-ac_penalty * 3)

        # 6. Consecutive number penalty (10%)
        consec = self._count_consecutive(main)
        # Allow 1 consecutive pair, penalize more
        consec_penalty = max(0, consec - 1)
        score += 10.0 * math.exp(-consec_penalty)

        # 7. Hot/Cold balance (5%)
        # Mix of hot (frequent), warm, and cold numbers
        if prob_scores and "main_probs" in prob_scores:
            mp = prob_scores["main_probs"]
            n_main = len(mp)
            sorted_idx = np.argsort(mp)[::-1]
            hot_thresh = int(n_main * 0.2)
            cold_thresh = int(n_main * 0.8)

            main_idx_set = [n - self.cfg.main_min for n in main]
            hot_count = sum(1 for i in main_idx_set if i <= hot_thresh)
            cold_count = sum(1 for i in main_idx_set if i >= cold_thresh)

            # Prefer mix: 2-3 hot, 1-2 warm, 1-2 cold
            hot_ideal = max(1, self.cfg.main_count // 3)
            cold_ideal = max(1, self.cfg.main_count // 4)
            hc_penalty = abs(hot_count - hot_ideal) * 2 + abs(cold_count - cold_ideal) * 2
            score += 5.0 * math.exp(-hc_penalty / self.cfg.main_count)

        # 8. Historical repeat penalty (5%)
        # Penalize combinations that are too similar to recent draws
        # (handled externally via diversity factor)

        # 9. Diversity within combination (5%)
        # Numbers should be spread out, not clustered
        main_sorted = sorted(main)
        gaps = [main_sorted[i + 1] - main_sorted[i] for i in range(len(main_sorted) - 1)]
        gap_mean = np.mean(gaps) if gaps else 1
        gap_std = np.std(gaps) if gaps else 0
        # Good spread: mean gap reasonable, not too variable
        cv = gap_std / max(gap_mean, 0.1)
        score += 5.0 * math.exp(-cv * 2)

        # 10. Structural stability bonus (5%)
        # Reward combinations that match typical structural patterns
        # e.g., numbers from each decade (1-9, 10-19, 20-29, etc.)
        decades = Counter(n // 10 for n in main)
        decade_count = len(decades)
        ideal_decades = min(4, self.cfg.main_count)
        decade_score = 1.0 - abs(decade_count - ideal_decades) / max(ideal_decades, 1)
        score += 5.0 * decade_score

        return score

    def _ac_value(self, nums: list) -> int:
        n = len(nums)
        if n <= 1:
            return 0
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(abs(nums[i] - nums[j]))
        return len(diffs) - (n - 1)

    def _expected_ac(self) -> float:
        """Expected AC value for random combinations of this type."""
        # Heuristic: AC ~ main_count * (main_count - 1) / 2 - (main_count - 1) * 0.5
        n = self.cfg.main_count
        max_possible = n * (n - 1) // 2 - (n - 1)  # max AC
        return max_possible * 0.65  # typical ~65% of max

    def _count_consecutive(self, nums: list) -> int:
        s = sorted(nums)
        count = 0
        for i in range(len(s) - 1):
            if s[i + 1] - s[i] == 1:
                count += 1
        return count

    # ====================================================================
    # SELECTION (Tournament)
    # ====================================================================

    def _select(self, population: List, fitnesses: List[float]) -> Dict:
        """Tournament selection: pick random subset, return best."""
        best = None
        best_f = -float("inf")
        for _ in range(self.tournament_size):
            idx = self._rng.randint(0, len(population) - 1)
            if fitnesses[idx] > best_f:
                best_f = fitnesses[idx]
                best = population[idx]
        return best

    # ====================================================================
    # CROSSOVER
    # ====================================================================

    def _crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """
        Two-point crossover for main numbers, single-point for sub numbers.
        Ensures validity (no duplicates, correct count).
        """
        if self._rng.random() > self.crossover_rate:
            return dict(parent1), dict(parent2)

        # Main numbers crossover (two-point)
        m1, m2 = parent1["main"], parent2["main"]
        n = len(m1)
        if n >= 3:
            p1 = self._rng.randint(1, n - 2)
            p2 = self._rng.randint(p1 + 1, n - 1)

            # Child 1: first part from p1, middle from p2, last from p1
            child1_main = set()
            for x in m1[:p1]: child1_main.add(x)
            for x in m2[p1:p2]: child1_main.add(x)
            for x in m1[p2:]: child1_main.add(x)

            # Child 2: swap
            child2_main = set()
            for x in m2[:p1]: child2_main.add(x)
            for x in m1[p1:p2]: child2_main.add(x)
            for x in m2[p2:]: child2_main.add(x)

            # Ensure validity: must have exactly main_count numbers
            child1_main = self._ensure_valid_set(child1_main, self.main_range, n)
            child2_main = self._ensure_valid_set(child2_main, self.main_range, n)
        else:
            child1_main = list(m1)
            child2_main = list(m2)

        # Sub numbers crossover (single-point)
        s1, s2 = parent1["sub"], parent2["sub"]
        ns = len(s1)
        if ns >= 2:
            split = self._rng.randint(1, ns - 1)
            child1_sub = set(s1[:split] + s2[split:])
            child2_sub = set(s2[:split] + s1[split:])
            child1_sub = self._ensure_valid_set(child1_sub, self.sub_range, ns)
            child2_sub = self._ensure_valid_set(child2_sub, self.sub_range, ns)
        else:
            child1_sub = list(s1)
            child2_sub = list(s2)

        return (
            {"main": sorted(child1_main), "sub": sorted(child1_sub)},
            {"main": sorted(child2_main), "sub": sorted(child2_sub)},
        )

    def _ensure_valid_set(self, comb_set: set, full_range: list, target_count: int) -> list:
        """Ensure a set has exactly target_count valid numbers in range."""
        valid = sorted(n for n in comb_set if n in full_range)
        if len(valid) >= target_count:
            return sorted(valid[:target_count])
        # Fill missing slots
        pool = [n for n in full_range if n not in valid]
        self._rng.shuffle(pool)
        return sorted(valid + pool[:target_count - len(valid)])

    # ====================================================================
    # MUTATION
    # ====================================================================

    def _mutate(self, chromosome: Dict) -> Dict:
        """Randomly replace numbers with mutation_rate probability."""
        main = list(chromosome["main"])
        sub = list(chromosome["sub"])

        # Mutate main numbers
        for i in range(len(main)):
            if self._rng.random() < self.mutation_rate:
                pool = [n for n in self.main_range if n not in main]
                if pool:
                    new_n = self._rng.choice(pool)
                    old_n = main[i]
                    main[i] = new_n
                    # Ensure no duplicate
                    if main.count(new_n) > 1:
                        main[i] = old_n

        # Mutate sub numbers
        for i in range(len(sub)):
            if self._rng.random() < self.mutation_rate:
                pool = [n for n in self.sub_range if n not in sub]
                if pool:
                    new_n = self._rng.choice(pool)
                    old_n = sub[i]
                    sub[i] = new_n
                    if sub.count(new_n) > 1:
                        sub[i] = old_n

        return {"main": sorted(main), "sub": sorted(sub)}

    # ====================================================================
    # MAIN EVOLUTION LOOP
    # ====================================================================

    def evolve(
        self,
        prob_scores: Optional[Dict[str, np.ndarray]] = None,
        initial_population: Optional[List[Dict]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the full genetic algorithm evolution.

        Parameters
        ----------
        prob_scores : optional dict with 'main_probs' and 'sub_probs'
        initial_population : optional list of seed chromosomes
        verbose : bool

        Returns
        -------
        dict with:
            best_chromosome : best combination found
            best_fitness : its fitness score
            top_n : top N combinations
            evolution_history : [best, avg, diversity] per generation
            fitness_history : list of (gen, best_f, avg_f, diversity)
        """
        # 1. Initialize population
        population = []
        if initial_population:
            # Seed with known good combinations
            for c in initial_population[:self.population_size // 2]:
                population.append(c)
        while len(population) < self.population_size:
            population.append(self._random_chromosome())

        # 2. Evaluate initial fitness
        fitnesses = [self.fitness(c, prob_scores) for c in population]

        best_overall = None
        best_f_overall = -float("inf")
        evolution_history = []

        # 3. Evolution loop
        for gen in range(self.generations):
            new_population = []
            new_fitnesses = []

            # Elite preservation
            elite_count = max(1, int(self.population_size * self.elite_ratio))
            elite_indices = np.argsort(fitnesses)[-elite_count:]
            for idx in elite_indices:
                new_population.append(population[idx])
                new_fitnesses.append(fitnesses[idx])

            # Fill rest with crossover + mutation
            while len(new_population) < self.population_size:
                parent1 = self._select(population, fitnesses)
                parent2 = self._select(population, fitnesses)
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)

                f1 = self.fitness(child1, prob_scores)
                f2 = self.fitness(child2, prob_scores)

                new_population.append(child1)
                new_fitnesses.append(f1)
                if len(new_population) < self.population_size and f2 > f1 * 0.8:
                    new_population.append(child2)
                    new_fitnesses.append(f2)

            # Trim to population size
            if len(new_population) > self.population_size:
                # Keep only the best
                combined = list(zip(new_population, new_fitnesses))
                combined.sort(key=lambda x: -x[1])
                new_population = [c for c, _ in combined[:self.population_size]]
                new_fitnesses = [f for _, f in combined[:self.population_size]]

            population = new_population
            fitnesses = new_fitnesses

            # Track progress
            best_f = max(fitnesses)
            avg_f = sum(fitnesses) / len(fitnesses)
            diversity = len(set(tuple(c["main"]) for c in population))

            evolution_history.append((gen, best_f, avg_f, diversity))

            if best_f > best_f_overall:
                best_f_overall = best_f
                best_overall = population[fitnesses.index(best_f)]

            # Convergence check
            if gen > 20 and len(evolution_history) >= 10:
                recent = [h[1] for h in evolution_history[-10:]]
                if max(recent) - min(recent) < 0.5:
                    if verbose:
                        print(f"  GA converged at gen {gen}")
                    break

            if verbose and (gen % 20 == 0 or gen == self.generations - 1):
                print(f"  Gen {gen:3d}: best={best_f:.2f}, avg={avg_f:.2f}, diversity={diversity}")

        # Compile results
        combined = list(zip(population, fitnesses))
        combined.sort(key=lambda x: -x[1])
        top_n = [
            {"main": c["main"], "sub": c["sub"], "fitness": round(f, 2)}
            for c, f in combined[:min(20, len(combined))]
        ]

        self.fitness_history = evolution_history

        return {
            "best_chromosome": best_overall,
            "best_fitness": round(best_f_overall, 2),
            "top_n": top_n,
            "evolution_history": evolution_history,
            "final_diversity": len(set(tuple(c["main"]) for c in population)),
            "population_size": self.population_size,
            "generations_run": gen + 1,
        }
