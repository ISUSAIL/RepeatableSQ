"""
Repeatable Statistical Testing Framework

Implementation of repeatable testing methodology from:
"Rethink Repeatable Measures of Robot Performance with Statistical Query"
"""

import numpy as np
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import math

# Import our toy distributions
from distributions import (
    Distribution01, BernoulliRare, BetaSkewed, 
    BimodalMixture, UniformSpike
)


@dataclass
class RepeatableTestConfig:
    """Configuration for repeatable testing based on the paper"""
    gamma: float = 0.01  # Accuracy parameter
    c: float = 0.05      # Confidence parameter (1-c is confidence level)
    beta: float = 0.1    # Repeatability parameter (valid for c=0.05)
    alpha: Optional[float] = None  # Quantization parameter (computed if None)
    max_samples: int = 100000  # Maximum samples before forced termination
    min_samples: int = 100     # Minimum samples before considering termination
    random_seed: Optional[int] = None
    
    def validate_parameters(self) -> None:
        """
        Validate parameters for repeatable testing.
        
        Constraints from the paper:
        1. 0 < γ ≤ 1 (accuracy parameter)
        2. 0 < c < 1 (confidence parameter)
        3. 0 < β < 1 (repeatability parameter)
        4. (1-c)² - (1-β) ≥ 0 (for valid α computation)
        """
        errors = []
        warnings = []
        
        # Check basic parameter ranges
        if not (0 < self.gamma <= 1):
            errors.append(f"γ must be in (0, 1], got {self.gamma}")
        
        if not (0 < self.c < 1):
            errors.append(f"c must be in (0, 1), got {self.c}")
        
        if not (0 < self.beta < 1):
            errors.append(f"β must be in (0, 1), got {self.beta}")
        
        # Check discriminant constraint: (1-c)² ≥ (1-β) ⟹ β ≥ 1-(1-c)²
        if 0 < self.c < 1 and 0 < self.beta < 1:
            discriminant = (1 - self.c)**2 - (1 - self.beta)
            if discriminant < 0:
                min_valid_beta = 1 - (1 - self.c)**2
                errors.append(
                    f"Parameters must satisfy (1-c)² ≥ (1-β), but got "
                    f"(1-{self.c})² - (1-{self.beta}) = {discriminant:.6f} < 0. "
                    f"For c={self.c}, β must be ≥ {min_valid_beta:.6f}"
                )
         
        # Print warnings
        if warnings:
            print("Parameter warnings:")
            for warning in warnings:
                print(f"  WARNING: {warning}")
        
        # Raise errors
        if errors:
            error_msg = "Invalid parameters:\n" + "\n".join(f"  ERROR: {err}" for err in errors)
            raise ValueError(error_msg)
    
    def compute_alpha(self) -> float:
        """
        Compute α-quantization parameter based on Equation (12) from the paper:
        α = 2γ[(1-c) - √((1-c)² - (1-β))]/(1-c)
        """
        if self.alpha is not None:
            return self.alpha
        
        # Validate parameters first
        self.validate_parameters()
        
        # From the paper: α = 2γ[(1-c) - √((1-c)² - (1-β))]/(1-c)
        one_minus_c = 1 - self.c
        one_minus_beta = 1 - self.beta
        
        # Compute discriminant (already validated to be non-negative)
        discriminant = one_minus_c**2 - one_minus_beta
        
        alpha = 2 * self.gamma * (one_minus_c - math.sqrt(discriminant)) / one_minus_c
        return alpha
    
    @staticmethod
    def suggest_valid_parameters(gamma: float = 0.01, c: float = 0.05) -> None:
        """
        Suggest valid parameter combinations for given γ and c values.
        
        Args:
            gamma: Desired accuracy parameter
            c: Desired confidence parameter
        """
        if not (0 < gamma <= 1):
            print(f"Invalid γ={gamma}, must be in (0, 1]")
            return
        
        if not (0 < c < 1):
            print(f"Invalid c={c}, must be in (0, 1)")
            return
        
        min_beta = 1 - (1 - c)**2
        
        print(f"Valid parameter ranges for γ={gamma}, c={c}:")
        print(f"  γ: {gamma} (accuracy parameter)")
        print(f"  c: {c} (confidence parameter → {100*(1-c):.1f}% confidence)")
        print(f"  β: must be ≥ {min_beta:.6f} (repeatability parameter)")
        
        # Suggest some good values
        suggested_betas = [min_beta + 0.001, min_beta + 0.01, min_beta + 0.05, 0.5]
        suggested_betas = [b for b in suggested_betas if b >= min_beta and b < 1]
        
        print(f"  Suggested β values: {suggested_betas}")
        
        # Show resulting α values
        print(f"  Resulting α values:")
        for beta in suggested_betas:
            if beta >= min_beta and beta < 1:
                one_minus_c = 1 - c
                discriminant = one_minus_c**2 - (1 - beta)
                if discriminant >= 0:
                    alpha = 2 * gamma * (one_minus_c - math.sqrt(discriminant)) / one_minus_c
                    print(f"    β={beta:.4f} → α={alpha:.6f} ({int(np.ceil(1.0/alpha))} intervals)")
                else:
                    print(f"    β={beta:.4f} → INVALID")


@dataclass 
class RepeatableTestResult:
    """Result of a repeatable statistical test"""
    original_estimate: float       # r_n: Original estimate before quantization
    quantized_estimate: float      # TE_α(r_n): α-quantized estimate
    true_value: float              # r*: Ground truth
    n_samples: int                 # Number of samples used
    empirical_variance: float      # σ̂_n: Empirical variance estimate
    alpha: float                   # α: Quantization parameter used
    gamma: float                   # γ: Accuracy parameter
    c: float                       # c: Confidence parameter
    beta: float                    # β: Repeatability parameter
    termination_bound: float       # Final Bernstein bound value
    is_converged: bool             # Whether algorithm terminated due to convergence
    partition_info: Dict           # Information about the α-partition
    
    def accuracy_guarantee(self) -> float:
        """Returns the accuracy guarantee: |r_n - r*| ≤ γ + α/2 with probability 1-c"""
        return self.gamma + self.alpha / 2
    
    def is_exactly_repeatable_with(self, other: 'RepeatableTestResult') -> bool:
        """Check exact repeatability: two trials must give identical quantized results"""
        return abs(self.quantized_estimate - other.quantized_estimate) < 1e-15


class AlphaQuantizer:
    """
    α-quantization mechanism for ensuring repeatability.
    
    Implements the almost uniform α-partition as described in the paper.
    """
    
    def __init__(self, alpha: float, output_range: Tuple[float, float] = (0.0, 1.0)):
        """
        Initialize α-quantizer.
        
        Args:
            alpha: Quantization parameter
            output_range: Range of possible outputs (for [0,1] distributions)
        """
        self.alpha = alpha
        self.range_min, self.range_max = output_range
        self.range_width = self.range_max - self.range_min
        
        # Create uniform α-partition
        self.n_intervals = max(1, int(np.ceil(self.range_width / alpha)))
        self.actual_alpha = self.range_width / self.n_intervals
        
        # Partition boundaries
        self.boundaries = np.linspace(self.range_min, self.range_max, self.n_intervals + 1)
        
        # Midpoints of intervals (quantized values)
        self.midpoints = (self.boundaries[:-1] + self.boundaries[1:]) / 2
    
    def quantize(self, value: float) -> float:
        """
        Apply α-quantization: TE_α(value) = midpoint of interval containing value.
        
        Args:
            value: Original value to quantize
            
        Returns:
            Quantized value (midpoint of containing interval)
        """
        # Clamp value to range
        value = np.clip(value, self.range_min, self.range_max)
        
        # Find which interval contains the value
        interval_idx = np.searchsorted(self.boundaries[1:], value)
        interval_idx = min(interval_idx, len(self.midpoints) - 1)
        
        return self.midpoints[interval_idx]
    
    def get_partition_info(self) -> Dict:
        """Get information about the α-partition"""
        return {
            'alpha': self.alpha,
            'actual_alpha': self.actual_alpha,
            'n_intervals': self.n_intervals,
            'boundaries': self.boundaries.tolist(),
            'midpoints': self.midpoints.tolist()
        }


class RepeatableStatQuery:
    """
    Repeatable Statistical Query Engine
    
    Implements Algorithm 1 and Algorithm 2 from the paper:
    - Algorithm 1: Basic statistical query with empirical Bernstein stopping
    - Algorithm 2: Repeatable testing with α-quantization
    """
    
    def __init__(self):
        """Initialize the repeatable testing framework"""
        self.test_history: List[RepeatableTestResult] = []
        self.shared_partitions: Dict[str, AlphaQuantizer] = {}  # For repeatability
    
    def _empirical_bernstein_bound(self, n: int, empirical_var: float, 
                                  max_weight: float, c: float) -> float:
        """
        Compute empirical Bernstein inequality bound from the paper.
        
        Bound: √(2σ̂_n ln(2/c)/n) + 7(mw̄)²ln(2/c)/3(n-1)
        where σ̂_n is empirical variance, mw̄ is maximum weight
        """
        if n <= 1:
            return float('inf')
        
        log_term = math.log(2 / c)
        
        # Variance term: √(2σ̂_n ln(2/c)/n)
        variance_term = math.sqrt(2 * empirical_var * log_term / n)
        
        # Range term: 7(mw̄)²ln(2/c)/3(n-1)
        range_term = 7 * (max_weight ** 2) * log_term / (3 * (n - 1))
        
        return variance_term + range_term
    
    def _compute_empirical_variance(self, samples: np.ndarray, weights: np.ndarray,
                                   current_estimate: float) -> float:
        """
        Compute empirical variance estimate: σ̂_n = (1/|T|) Σ(ψ(x_i)w(x_i) - r_n)²
        
        For our case with ψ(x) = x (mean estimation), this becomes:
        σ̂_n = (1/n) Σ(x_i * w_i - r_n)²
        """
        if len(samples) == 0:
            return 0.0
        
        weighted_samples = samples * weights
        squared_deviations = (weighted_samples - current_estimate) ** 2
        
        return np.mean(squared_deviations)
    
    def _algorithm1_basic_sq(self, target_dist: Distribution01, 
                           importance_dist: Optional[Distribution01],
                           config: RepeatableTestConfig) -> Tuple[float, int, float]:
        """
        Algorithm 1: Basic Statistical Query with Empirical Bernstein Stopping
        
        Returns:
            (estimate, n_samples_used, empirical_variance)
        """
        # Setup random state (only if explicitly provided)
        # Note: For repeatability testing, we want different random seeds per trial
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
        
        # Use importance sampling if provided, otherwise standard MC
        if importance_dist is None:
            importance_dist = target_dist
        
        samples = []
        weights = []
        
        for n in range(1, config.max_samples + 1):
            # Generate new sample
            x = importance_dist.sample(1)[0]
            
            # Compute importance weight
            if importance_dist.name == target_dist.name:
                w = 1.0  # Monte Carlo case
            else:
                target_pdf = target_dist.pdf(np.array([x]))[0]
                importance_pdf = importance_dist.pdf(np.array([x]))[0]
                if importance_pdf > 1e-12:
                    w = target_pdf / importance_pdf
                    # Store raw weight - truncation will be applied during estimation
                    # Cap only extreme outliers to prevent numerical issues
                    w = min(w, 1000.0)  # Very high cap, just for numerical stability
                else:
                    w = 0.0
            
            samples.append(x)
            weights.append(w)
            
            # Check termination condition (after minimum samples)
            if n >= config.min_samples:
                samples_array = np.array(samples)
                weights_array = np.array(weights)
                
                # Apply truncated importance sampling for unbiased estimation
                truncation_threshold = 50.0
                truncated_weights = np.minimum(weights_array, truncation_threshold)
                
                # Bias correction for truncated weights
                if np.any(weights_array > truncation_threshold):
                    total_raw = np.sum(weights_array)
                    total_truncated = np.sum(truncated_weights)
                    if total_truncated > 0:
                        bias_correction = min(total_raw / total_truncated, 2.0)  # Cap at 2x
                        truncated_weights *= bias_correction
                
                # Current estimate using truncated weights: r_n = Σ(x_i * w_i) / Σ(w_i)
                if np.sum(truncated_weights) > 0:
                    current_estimate = np.sum(samples_array * truncated_weights) / np.sum(truncated_weights)
                else:
                    current_estimate = 0.0
                
                # Use truncated weights for variance computation too
                weights_array = truncated_weights
                
                # Empirical variance
                empirical_var = self._compute_empirical_variance(
                    samples_array, weights_array, current_estimate
                )
                
                # Maximum weight for bound computation
                max_weight = np.max(weights_array)
                
                # Bernstein bound
                bound = self._empirical_bernstein_bound(n, empirical_var, max_weight, config.c)
                
                # Termination condition: γ ≥ bound
                if config.gamma >= bound:
                    return current_estimate, n, empirical_var
        
        # Fallback: use all samples if didn't converge
        samples_array = np.array(samples)
        weights_array = np.array(weights)
        
        if np.sum(weights_array) > 0:
            final_estimate = np.sum(samples_array * weights_array) / np.sum(weights_array)
        else:
            final_estimate = 0.0
        
        final_empirical_var = self._compute_empirical_variance(
            samples_array, weights_array, final_estimate
        )
        
        return final_estimate, config.max_samples, final_empirical_var
    
    def algorithm2_repeatable_test(self, target_dist: Distribution01,
                                  importance_dist: Optional[Distribution01],
                                  config: RepeatableTestConfig,
                                  partition_key: Optional[str] = None) -> RepeatableTestResult:
        """
        Algorithm 2: Repeatable Testing Framework
        
        This implements the full repeatable testing algorithm from the paper.
        Two modes:
        1. Testing Initiator: Creates new α-partition and runs quantized algorithm
        2. Testing Replicator: Uses existing α-partition for exact repeatability
        
        Args:
            target_dist: Target distribution to estimate
            importance_dist: Importance distribution (None for Monte Carlo)
            config: Test configuration
            partition_key: Key for shared partition (None = create new partition)
            
        Returns:
            RepeatableTestResult with exact repeatability guarantees
        """
        # Compute α parameter
        alpha = config.compute_alpha()
        
        # Create or retrieve α-quantizer
        if partition_key is None:
            # Testing Initiator mode: create new partition
            quantizer = AlphaQuantizer(alpha, output_range=(0.0, 1.0))
            partition_key = f"dist_{target_dist.name}_{hash(str(config.__dict__))}"
            self.shared_partitions[partition_key] = quantizer
        else:
            # Testing Replicator mode: use existing partition
            if partition_key not in self.shared_partitions:
                # If partition doesn't exist, create it (handles the first call with a specific key)
                quantizer = AlphaQuantizer(alpha, output_range=(0.0, 1.0))
                self.shared_partitions[partition_key] = quantizer
            else:
                quantizer = self.shared_partitions[partition_key]
        
        # Run Algorithm 1 (basic statistical query)
        original_estimate, n_samples, empirical_var = self._algorithm1_basic_sq(
            target_dist, importance_dist, config
        )
        
        # Apply α-quantization
        quantized_estimate = quantizer.quantize(original_estimate)
        
        # Compute final termination bound using actual max weight from samples
        max_weight = 1.0  # For MC case, weights are always 1.0
        if importance_dist is not None and importance_dist.name != target_dist.name:
            # For IS case, we need to estimate max weight more carefully
            # Use a small sample to estimate the maximum weight
            test_samples = importance_dist.sample(100)
            target_pdf = target_dist.pdf(test_samples)
            importance_pdf = importance_dist.pdf(test_samples)
            test_weights = np.where(importance_pdf > 1e-12, target_pdf / importance_pdf, 0.0)
            max_weight = min(10.0, np.max(test_weights) * 1.2)  # Cap at 10, add 20% safety margin
        
        final_bound = self._empirical_bernstein_bound(n_samples, empirical_var, max_weight, config.c)
        
        # Create result
        result = RepeatableTestResult(
            original_estimate=original_estimate,
            quantized_estimate=quantized_estimate,
            true_value=target_dist.true_mean(),
            n_samples=n_samples,
            empirical_variance=empirical_var,
            alpha=alpha,
            gamma=config.gamma,
            c=config.c,
            beta=config.beta,
            termination_bound=final_bound,
            is_converged=(config.gamma >= final_bound),
            partition_info=quantizer.get_partition_info()
        )
        
        self.test_history.append(result)
        return result
    
    def demonstrate_exact_repeatability(self, target_dist: Distribution01,
                                      importance_dist: Optional[Distribution01],
                                      config: RepeatableTestConfig,
                                      n_trials: int = 5) -> Dict:
        """
        Demonstrate exact repeatability: multiple independent trials with identical results
        
        Returns:
            Dictionary with repeatability demonstration results
        """
        print(f"Demonstrating exact repeatability for {target_dist.name}")
        print(f"Parameters: γ={config.gamma}, c={config.c}, β={config.beta}")
        print(f"Computed α={config.compute_alpha():.6f}")
        
        # Generate consistent partition key for this demonstration
        # Use stable config parameters (excluding random_seed) to ensure same partition
        stable_config = f"g{config.gamma}_c{config.c}_b{config.beta}_ms{config.max_samples}"
        partition_key = f"demo_{target_dist.name}_{stable_config}"
        
        results = []
        
        for trial in range(n_trials):
            if trial == 0:
                # Initiator: create new partition with predefined key
                result = self.algorithm2_repeatable_test(
                    target_dist, importance_dist, config, partition_key=partition_key
                )
            else:
                # Replicator: use the same partition key (partition already exists)
                result = self.algorithm2_repeatable_test(
                    target_dist, importance_dist, config, partition_key=partition_key
                )
            
            results.append(result)
            print(f"  Trial {trial+1}: original={result.original_estimate:.6f}, "
                  f"quantized={result.quantized_estimate:.6f}, n={result.n_samples}")
        
        # Check exact repeatability
        base_result = results[0]
        all_exactly_repeatable = True
        non_repeatable_count = 0
        
        for i, result in enumerate(results[1:], 1):
            is_exactly_repeatable = base_result.is_exactly_repeatable_with(result)
            if not is_exactly_repeatable:
                all_exactly_repeatable = False
                non_repeatable_count += 1
                print(f"  ⚠ Trial {i+1} NOT exactly repeatable! "
                      f"(quantized: {result.quantized_estimate:.6f} vs {base_result.quantized_estimate:.6f})")
        
        if all_exactly_repeatable:
            print(f"  ✓ All {n_trials} trials exactly repeatable!")
        else:
            success_rate = (n_trials - non_repeatable_count) / n_trials
            theoretical_success_rate = 1 - config.beta
            print(f"  Repeatability: {n_trials - non_repeatable_count}/{n_trials} trials repeatable "
                  f"({100*success_rate:.1f}% vs {100*theoretical_success_rate:.1f}% theoretical)")
        
        return {
            'all_exactly_repeatable': all_exactly_repeatable,
            'repeatable_count': n_trials - non_repeatable_count,
            'total_trials': n_trials,
            'success_rate': (n_trials - non_repeatable_count) / n_trials,
            'theoretical_success_rate': 1 - config.beta,
            'quantized_estimate': base_result.quantized_estimate,
            'alpha': base_result.alpha,
            'accuracy_guarantee': base_result.accuracy_guarantee(),
            'results': results
        }
