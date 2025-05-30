"""
Ultimate Value at Risk (VaR) and Expected Shortfall (ES) Calculator
==================================================================

This advanced calculator provides comprehensive functionality for analyzing financial risk 
metrics, combining educational value with production-ready performance.

Key Features:
- Efficient calculation with result caching
- Multiple distribution options (Normal and Student's t)
- Comprehensive visualization tools
- Detailed educational explanations
- Time horizon sensitivity analysis
- Robust error handling and input validation

Definitions:
- VaR (Value at Risk): Maximum potential loss at a given confidence level over a specified time period
- ES (Expected Shortfall): Expected loss when exceeding the VaR threshold (also called Conditional VaR)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from typing import Dict, Tuple, Union, List, Optional, Any, Callable
import seaborn as sns
from dataclasses import dataclass
from functools import wraps

# Set plot style for consistent visuals
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 7)
plt.rcParams["figure.dpi"] = 100

@dataclass(frozen=True)
class ReturnStats:
    """Immutable container for return statistics to prevent accidental modification."""
    mean: float
    std: float
    n: int
    distribution: str = "normal"
    degrees_of_freedom: Optional[float] = None
    scale: Optional[float] = None


class RiskMetricsCalculator:
    """
    Comprehensive calculator for Value at Risk (VaR) and Expected Shortfall (ES).
    
    This class provides methods to calculate and visualize risk metrics using both
    parametric and non-parametric (historical) approaches. It supports multiple
    distribution assumptions and includes detailed educational features.
    """

    def __init__(self, returns: Union[pd.Series, np.ndarray, List[float]], *, name: str = None):
        """
        Initialize the calculator with financial returns data.
        
        Parameters:
        -----------
        returns : Union[pd.Series, np.ndarray, List[float]]
            Financial returns data (e.g., daily returns) as fractional values (e.g., 0.01 for 1%)
        name : str, optional
            Name for this data series (e.g., "Portfolio A" or "S&P 500")
        """
        # Convert input to pandas Series and handle NaN values
        if isinstance(returns, (np.ndarray, list)):
            self.returns = pd.Series(returns, dtype=float)
        elif isinstance(returns, pd.Series):
            self.returns = returns.astype(float)
        else:
            raise TypeError("Returns must be a pandas Series, numpy array, or list of floats.")
        
        # Drop NaN values if any
        if self.returns.isnull().any():
            original_len = len(self.returns)
            self.returns = self.returns.dropna()
            print(f"Warning: Dropped {original_len - len(self.returns)} NaN values from returns.")
        
        # Validate data
        if len(self.returns) < 2:
            raise ValueError("Insufficient data: need at least 2 non-NaN returns for calculations.")
        
        # Store data attributes
        self.name = name or "Financial Series"
        self._cache: Dict[Tuple[str, float, int, str], float] = {}  # For caching results
        
        # Calculate basic statistics for normal distribution
        mean = self.returns.mean()
        std = self.returns.std(ddof=1)  # Using sample standard deviation
        n = len(self.returns)
        
        # Store normal distribution stats
        self._stats_normal = ReturnStats(mean=mean, std=std, n=n, distribution="normal")
        
        # Fit t-distribution if enough data points
        if n >= 4:  # Need at least 4 points for reliable t-distribution fitting
            try:
                df, loc, scale = scipy.stats.t.fit(self.returns)
                # Adjust standard deviation since t-distribution has different variance properties
                adjusted_std = scale * np.sqrt(df / (df - 2)) if df > 2 else std
                self._stats_t = ReturnStats(
                    mean=loc, std=adjusted_std, n=n, 
                    distribution="t", degrees_of_freedom=df, scale=scale
                )
                self._t_distribution_available = True
            except Exception as e:
                print(f"Warning: Could not fit t-distribution: {e}. Using normal distribution only.")
                self._t_distribution_available = False
        else:
            print("Note: Too few data points for reliable t-distribution fitting. Using normal distribution only.")
            self._t_distribution_available = False
        
        # Print initialization summary
        print(f"Initialized {self.name} with {n} data points")
        print(f"Mean return: {mean:.6f}")
        print(f"Standard deviation: {std:.6f}")
        if self._t_distribution_available:
            print(f"Fitted t-distribution with {self._stats_t.degrees_of_freedom:.2f} degrees of freedom")

    # ---------- Factory Methods ---------- #
    
    @classmethod
    def from_prices(cls, prices: Union[pd.Series, np.ndarray, List[float]], 
                   log_returns: bool = True, name: str = None) -> 'RiskMetricsCalculator':
        """
        Create a RiskMetricsCalculator from a series of prices.
        
        Parameters:
        -----------
        prices : Union[pd.Series, np.ndarray, List[float]]
            Time series of prices
        log_returns : bool, default=True
            If True, calculate logarithmic returns; if False, calculate simple returns
        name : str, optional
            Name for this data series
            
        Returns:
        --------
        RiskMetricsCalculator
            A calculator initialized with the derived returns
        """
        # Convert to Series if needed
        if isinstance(prices, (np.ndarray, list)):
            prices_series = pd.Series(prices, dtype=float)
        elif isinstance(prices, pd.Series):
            prices_series = prices.astype(float)
        else:
            raise TypeError("Prices must be a pandas Series, numpy array, or list of floats.")
        
        # Handle NaN values
        if prices_series.isnull().any():
            original_len = len(prices_series)
            prices_series = prices_series.dropna()
            print(f"Warning: Dropped {original_len - len(prices_series)} NaN values from prices.")
        
        # Validate data
        if len(prices_series) < 2:
            raise ValueError("Insufficient price data: need at least 2 non-NaN prices to calculate returns.")
        
        # Calculate returns
        if log_returns:
            returns = np.log(prices_series / prices_series.shift(1)).dropna()
            method = "logarithmic"
        else:
            returns = (prices_series / prices_series.shift(1) - 1).dropna()
            method = "simple"
            
        # Create name if none provided
        if name is None:
            name = f"Series ({method} returns)"
            
        print(f"Calculated {method} returns from {len(prices_series)} prices.")
        return cls(returns, name=name)
    
    @classmethod
    def from_csv(cls, file_path: str, price_column: str = 'Close',
                date_column: Optional[str] = 'Date', log_returns: bool = True,
                name: Optional[str] = None) -> 'RiskMetricsCalculator':
        """
        Create a RiskMetricsCalculator from a CSV file containing price data.
        
        Parameters:
        -----------
        file_path : str
            Path to the CSV file
        price_column : str, default='Close'
            Name of the column containing price data
        date_column : str, optional, default='Date'
            Name of the column containing dates (for sorting)
        log_returns : bool, default=True
            If True, calculate logarithmic returns; if False, calculate simple returns
        name : str, optional
            Name for this data series; if None, uses the file path
            
        Returns:
        --------
        RiskMetricsCalculator
            A calculator initialized with the derived returns
        """
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Validate columns
            if price_column not in df.columns:
                raise ValueError(f"Price column '{price_column}' not found in CSV file.")
            
            # Handle date column and sorting if provided
            if date_column is not None:
                if date_column not in df.columns:
                    print(f"Warning: Date column '{date_column}' not found. Using data in existing order.")
                else:
                    try:
                        df[date_column] = pd.to_datetime(df[date_column])
                        df = df.sort_values(date_column)
                        print(f"Data sorted by date from {df[date_column].min().date()} to {df[date_column].max().date()}")
                    except Exception as e:
                        print(f"Warning: Could not parse dates in column '{date_column}': {e}. Using data in existing order.")
            
            # Use file path as name if none provided
            if name is None:
                name = file_path
                
            # Extract prices and create calculator
            return cls.from_prices(df[price_column], log_returns=log_returns, name=name)
            
        except Exception as e:
            raise ValueError(f"Error loading CSV file '{file_path}': {e}")

    # ---------- Helper Methods ---------- #
    
    def _get_stats(self, dist: str = "normal") -> ReturnStats:
        """Get the appropriate statistics object based on distribution choice."""
        if dist == "normal":
            return self._stats_normal
        elif dist == "t":
            if not self._t_distribution_available:
                print("Warning: t-distribution not available. Using normal distribution instead.")
                return self._stats_normal
            return self._stats_t
        else:
            raise ValueError(f"Unsupported distribution: {dist}. Use 'normal' or 't'.")
    
    def _cached_calculation(func):
        """Decorator to cache calculation results."""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Create a unique key for this calculation
            confidence_level = kwargs.get('confidence_level', 0.95)
            time_horizon = kwargs.get('time_horizon', 1)
            dist = kwargs.get('dist', "normal")
            func_name = func.__name__
            
            # Create a cache key
            key = (func_name, confidence_level, time_horizon, dist)
            
            # Return cached result if available
            if key in self._cache:
                return self._cache[key]
            
            # Calculate and cache result
            result = func(self, *args, **kwargs)
            self._cache[key] = result
            return result
        
        return wrapper

    # ---------- Risk Calculation Methods ---------- #
    
    @_cached_calculation
    def calculate_parametric_var(self, confidence_level: float = 0.95, 
                                time_horizon: int = 1, dist: str = "normal") -> float:
        """
        Calculate parametric Value at Risk (VaR).
        
        Parameters:
        -----------
        confidence_level : float, default=0.95
            Confidence level for VaR (e.g., 0.95 for 95%)
        time_horizon : int, default=1
            Time horizon in days (or periods matching the returns frequency)
        dist : str, default="normal"
            Distribution assumption: "normal" or "t"
            
        Returns:
        --------
        float
            Parametric VaR as a positive number (absolute loss)
        """
        # Input validation
        if not 0 < confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1.")
        if time_horizon <= 0:
            raise ValueError("Time horizon must be positive.")
            
        # Get appropriate statistics
        return_stats = self._get_stats(dist)
        mean = return_stats.mean
        std = return_stats.std
        
        # Calculate VaR based on distribution
        if dist == "normal":
            # For normal distribution: VaR = -(μ*h + zα*σ*√h)
            z_score = scipy.stats.norm.ppf(1 - confidence_level)
            var = -(mean * time_horizon + z_score * std * np.sqrt(time_horizon))
        else:  # t-distribution
            # For t-distribution: VaR = -(μ*h + tα,df*σ*√h)
            df = return_stats.degrees_of_freedom
            t_score = scipy.stats.t.ppf(1 - confidence_level, df)
            var = -(mean * time_horizon + t_score * std * np.sqrt(time_horizon))
            
        return var
    
    def calculate_historical_var(self, confidence_level: float = 0.95) -> float:
        """
        Calculate historical (non-parametric) Value at Risk (VaR).
        
        Parameters:
        -----------
        confidence_level : float, default=0.95
            Confidence level for VaR (e.g., 0.95 for 95%)
            
        Returns:
        --------
        float
            Historical VaR as a positive number (absolute loss)
        """
        # Input validation
        if not 0 < confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1.")
            
        # Calculate historical VaR as the negative of the return at the (1-confidence_level) quantile
        return -np.percentile(self.returns, 100 * (1 - confidence_level))
    
    @_cached_calculation
    def calculate_parametric_es(self, confidence_level: float = 0.95, 
                               time_horizon: int = 1, dist: str = "normal") -> float:
        """
        Calculate parametric Expected Shortfall (ES).
        
        Parameters:
        -----------
        confidence_level : float, default=0.95
            Confidence level for ES (e.g., 0.95 for 95%)
        time_horizon : int, default=1
            Time horizon in days (or periods matching the returns frequency)
        dist : str, default="normal"
            Distribution assumption: "normal" or "t"
            
        Returns:
        --------
        float
            Parametric ES as a positive number (absolute loss)
        """
        # Input validation
        if not 0 < confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1.")
        if time_horizon <= 0:
            raise ValueError("Time horizon must be positive.")
            
        # Get appropriate statistics
        return_stats = self._get_stats(dist)
        mean = return_stats.mean
        std = return_stats.std
        
        # Calculate ES based on distribution
        if dist == "normal":
            # For normal distribution: ES = -(μ*h - σ*√h * φ(zα)/(1-α))
            # where φ is the PDF of standard normal and zα is the quantile
            z_alpha = scipy.stats.norm.ppf(1 - confidence_level)
            norm_pdf_z = scipy.stats.norm.pdf(z_alpha)
            es = -(mean * time_horizon - std * np.sqrt(time_horizon) * norm_pdf_z / (1 - confidence_level))
        else:  # t-distribution
            # For t-distribution: ES calculation is more complex
            df = return_stats.degrees_of_freedom
            t_alpha = scipy.stats.t.ppf(1 - confidence_level, df)
            
            # t-distribution ES formula
            # ES = -(μ*h - σ*√h * (t_pdf(t_α,df)*(df+t_α²)/((1-α)*(df-1))))
            # But we need to be careful about df ≤ 1 which makes ES infinite
            if df <= 1:
                print(f"Warning: t-distribution with df={df} has infinite ES. Using VaR as approximation.")
                return self.calculate_parametric_var(confidence_level, time_horizon, dist)
                
            t_pdf_t = scipy.stats.t.pdf(t_alpha, df)
            es_factor = t_pdf_t * (df + t_alpha**2) / ((1 - confidence_level) * (df - 1))
            es = -(mean * time_horizon - std * np.sqrt(time_horizon) * es_factor)
            
        return es
    
    def calculate_historical_es(self, confidence_level: float = 0.95) -> Tuple[float, int]:
        """
        Calculate historical (non-parametric) Expected Shortfall (ES).
        
        Parameters:
        -----------
        confidence_level : float, default=0.95
            Confidence level for ES (e.g., 0.95 for 95%)
            
        Returns:
        --------
        Tuple[float, int]
            Historical ES as a positive number (absolute loss) and the number of observations used
        """
        # Input validation
        if not 0 < confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1.")
            
        # Calculate VaR threshold
        var_threshold = np.percentile(self.returns, 100 * (1 - confidence_level))
        
        # Find returns that are at or below the VaR threshold
        extreme_returns = self.returns[self.returns <= var_threshold]
        n_extreme = len(extreme_returns)
        
        # Handle case with no observations (rare with sufficient data)
        if n_extreme == 0:
            print(f"Warning: No observations beyond VaR threshold for {confidence_level*100}% confidence. Using VaR.")
            return -var_threshold, 0
            
        # Return the negative mean of extreme returns (to represent loss as positive)
        return -extreme_returns.mean(), n_extreme

    # ---------- Summary Methods ---------- #
    
    def generate_risk_report(self, confidence_levels=[0.95], time_horizon=1, dist="normal"):
        data = []
        for cl in confidence_levels:
            var_hist = self.calculate_historical_var(cl)
            es_hist, _ = self.calculate_historical_es(cl)
            var_param = self.calculate_parametric_var(cl, time_horizon=time_horizon, dist=dist)
            es_param = self.calculate_parametric_es(cl, time_horizon=time_horizon, dist=dist)
            data.append({
                "Confiança": f"{cl*100:.1f}%",
                "VaR Histórico": var_hist,
                "ES Histórico": es_hist,
                f"VaR Paramétrico ({dist})": var_param,
                f"ES Paramétrico ({dist})": es_param,
            })
        return pd.DataFrame(data)

    # ---------- Visualization Methods ---------- #
    
    def plot_return_distribution(self, confidence_levels: List[float] = [0.95, 0.99], 
                                time_horizon: int = 1, dist: str = "normal", 
                                show_theoretical: bool = True, figsize=(12, 8)):
        """
        Plot the return distribution with VaR and ES visualizations.
        
        Parameters:
        -----------
        confidence_levels : List[float], default=[0.95, 0.99]
            Confidence levels to visualize
        time_horizon : int, default=1
            Time horizon for parametric calculations
        dist : str, default="normal"
            Distribution assumption for parametric calculations
        show_theoretical : bool, default=True
            Whether to show the theoretical distribution curve
        figsize : tuple, default=(12, 8)
            Figure size in inches
            
        Returns:
        --------
        matplotlib.figure.Figure
            The distribution plot figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot return distribution
        sns.histplot(self.returns, bins=50, kde=True, stat="density", ax=ax, 
                    label="Historical Returns", color="steelblue", alpha=0.6)
        
        # Get axis limits for vertical lines
        y_min, y_max = ax.get_ylim()
        
        # Generate a distinct color for each confidence level
        colors = sns.color_palette("viridis", len(confidence_levels))
        
        # Add VaR and ES lines for each confidence level
        for i, cl in enumerate(confidence_levels):
            # Calculate metrics
            p_var = self.calculate_parametric_var(cl, time_horizon, dist)
            p_es = self.calculate_parametric_es(cl, time_horizon, dist)
            h_var = self.calculate_historical_var(cl)
            h_es, _ = self.calculate_historical_es(cl)
            
            # Plot parametric metrics
            ax.axvline(-p_var, color=colors[i], linestyle="--", 
                      label=f"Parametric VaR ({cl*100:.0f}%, {time_horizon}d): {p_var:.4f}")
            ax.axvline(-p_es, color=colors[i], linestyle="-", 
                      label=f"Parametric ES ({cl*100:.0f}%, {time_horizon}d): {p_es:.4f}")
            
            # Plot historical metrics with slight alpha reduction
            ax.axvline(-h_var, color=colors[i], linestyle=":", alpha=0.7, 
                      label=f"Historical VaR ({cl*100:.0f}%): {h_var:.4f}")
            ax.axvline(-h_es, color=colors[i], linestyle="-.", alpha=0.7, 
                      label=f"Historical ES ({cl*100:.0f}%): {h_es:.4f}")
        
        # Add theoretical distribution if requested
        if show_theoretical:
            return_stats = self._get_stats(dist)
            x = np.linspace(min(self.returns) * 1.5, max(self.returns) * 1.5, 1000)
            
            if dist == "normal":
                # Normal distribution PDF
                y = scipy.stats.norm.pdf(x, return_stats.mean * time_horizon, return_stats.std * np.sqrt(time_horizon))
                ax.plot(x, y, "r--", alpha=0.8, 
                       label=f"Normal Distribution ({time_horizon}d)")
            else:
                # t-distribution PDF
                df = return_stats.degrees_of_freedom
                scale = return_stats.scale * np.sqrt(time_horizon)
                y = scipy.stats.t.pdf(x, df, loc=return_stats.mean * time_horizon, scale=scale)
                ax.plot(x, y, "r--", alpha=0.8, 
                       label=f"t-Distribution (df={df:.1f}, {time_horizon}d)")
        
        # Add plot labels and styling
        ax.set_title(f"Return Distribution with VaR and ES ({time_horizon}-day horizon)", fontsize=14)
        ax.set_xlabel("Return", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        
        # Add a shaded area for the tail region of the worst confidence level
        worst_cl = min(confidence_levels)
        worst_var = self.calculate_parametric_var(worst_cl, time_horizon, dist)
        ax.fill_between(x=np.linspace(ax.get_xlim()[0], -worst_var, 100),
                        y1=0, y2=y_max * 0.1, color='red', alpha=0.1,
                        label=f"Tail risk region ({(1-worst_cl)*100:.1f}%)")
        
        # Finalize plot
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout(rect=[0, 0, 0.75, 1])  # Make room for legend
        
        return fig
    
    def plot_confidence_sensitivity(self, confidence_range: Tuple[float, float] = (0.90, 0.99), 
                                   time_horizon: int = 1, dist: str = "normal", 
                                   num_points: int = 50, figsize=(14, 6)):
        """
        Plot VaR and ES sensitivity to changing confidence levels.
        
        Parameters:
        -----------
        confidence_range : Tuple[float, float], default=(0.90, 0.99)
            Range of confidence levels to analyze
        time_horizon : int, default=1
            Time horizon for parametric calculations
        dist : str, default="normal"
            Distribution assumption for parametric calculations
        num_points : int, default=50
            Number of points to evaluate within the confidence range
        figsize : tuple, default=(14, 6)
            Figure size in inches
            
        Returns:
        --------
        matplotlib.figure.Figure
            The sensitivity plot figure
        """
        # Input validation
        if not (0 < confidence_range[0] < confidence_range[1] < 1):
            raise ValueError("Confidence range must be within (0, 1) with start < end.")
        
        # Generate confidence levels
        conf_levels = np.linspace(confidence_range[0], confidence_range[1], num_points)
        
        # Calculate metrics across confidence levels
        p_vars = [self.calculate_parametric_var(cl, time_horizon, dist) for cl in conf_levels]
        h_vars = [self.calculate_historical_var(cl) for cl in conf_levels]
        p_es = [self.calculate_parametric_es(cl, time_horizon, dist) for cl in conf_levels]
        h_es = [self.calculate_historical_es(cl)[0] for cl in conf_levels]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)
        
        # Plot VaR sensitivity
        ax1.plot(conf_levels * 100, p_vars, "r-", linewidth=2, 
                label=f"Parametric VaR ({dist}, {time_horizon}d)")
        ax1.plot(conf_levels * 100, h_vars, "b--", linewidth=2, 
                label="Historical VaR")
        ax1.set_title("VaR Sensitivity to Confidence Level", fontsize=14)
        ax1.set_xlabel("Confidence Level (%)", fontsize=12)
        ax1.set_ylabel("Value at Risk (absolute loss)", fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot ES sensitivity
        ax2.plot(conf_levels * 100, p_es, "r-", linewidth=2, 
                label=f"Parametric ES ({dist}, {time_horizon}d)")
        ax2.plot(conf_levels * 100, h_es, "b--", linewidth=2, 
                label="Historical ES")
        ax2.set_title("Expected Shortfall Sensitivity to Confidence Level", fontsize=14)
        ax2.set_xlabel("Confidence Level (%)", fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Set shared y-label and overall title
        fig.suptitle(f"Risk Metrics Sensitivity Analysis", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for title
        
        return fig
    
    def plot_time_horizon_sensitivity(self, time_horizons: List[int], 
                                    confidence_level: float = 0.95, 
                                    dist: str = "normal", figsize=(12, 6)):
        """
        Plot VaR and ES sensitivity to changing time horizons.
        
        Parameters:
        -----------
        time_horizons : List[int]
            List of time horizons to analyze
        confidence_level : float, default=0.95
            Confidence level for calculations
        dist : str, default="normal"
            Distribution assumption for parametric calculations
        figsize : tuple, default=(12, 6)
            Figure size in inches
            
        Returns:
        --------
        matplotlib.figure.Figure
            The time horizon sensitivity plot figure
        """
        # Input validation
        if not all(isinstance(h, int) and h > 0 for h in time_horizons):
            raise ValueError("Time horizons must be positive integers.")
        if not 0 < confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1.")
        
        # Calculate metrics across time horizons
        p_vars = [self.calculate_parametric_var(confidence_level, h, dist) for h in time_horizons]
        p_es = [self.calculate_parametric_es(confidence_level, h, dist) for h in time_horizons]
        
        # Create square-root-of-time scaling reference
        h_var = self.calculate_historical_var(confidence_level)
        h_es, _ = self.calculate_historical_es(confidence_level)
        sqrt_scaling_var = [h_var * np.sqrt(h) for h in time_horizons]
        sqrt_scaling_es = [h_es * np.sqrt(h) for h in time_horizons]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot metrics
        ax.plot(time_horizons, p_vars, "r-", linewidth=2, marker="o", 
               label=f"Parametric VaR ({dist}, {confidence_level*100:.0f}%)")
        ax.plot(time_horizons, p_es, "b-", linewidth=2, marker="s", 
               label=f"Parametric ES ({dist}, {confidence_level*100:.0f}%)")
        
        # Plot square-root scaling reference
        ax.plot(time_horizons, sqrt_scaling_var, "r:", linewidth=1, alpha=0.7,
               label="√t Scaling (VaR)")
        ax.plot(time_horizons, sqrt_scaling_es, "b:", linewidth=1, alpha=0.7,
               label="√t Scaling (ES)")
        
        # Add labels and styling
        ax.set_title(f"Risk Metrics vs. Time Horizon (Confidence Level: {confidence_level*100:.0f}%)", 
                    fontsize=14)
        ax.set_xlabel("Time Horizon (days)", fontsize=12)
        ax.set_ylabel("Risk Metric (absolute loss)", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add text explanation
        ax.text(0.02, 0.98, "Note: Historical metrics shown with √t scaling for reference.", 
                transform=ax.transAxes, fontsize=10, va="top", alpha=0.7)
        
        plt.tight_layout()
        
        return fig
    
    def plot_comparative_distributions(self, comparison_returns: Optional[Union[pd.Series, np.ndarray, List[float]]] = None, 
                                      comparison_name: str = "Comparison Series", 
                                      confidence_level: float = 0.95, 
                                      time_horizon: int = 1, 
                                      dist: str = "normal", 
                                      figsize=(14, 6)):
        """
        Compare return distributions and risk metrics with another series.
        
        Parameters:
        -----------
        comparison_returns : Union[pd.Series, np.ndarray, List[float]], optional
            Returns of the comparison series; if None, only the primary series is shown
        comparison_name : str, default="Comparison Series"
            Name for the comparison series
        confidence_level : float, default=0.95
            Confidence level for risk calculations
        time_horizon : int, default=1
            Time horizon for parametric calculations
        dist : str, default="normal"
            Distribution assumption for parametric calculations
        figsize : tuple, default=(14, 6)
            Figure size in inches
            
        Returns:
        --------
        matplotlib.figure.Figure
            The comparative distribution plot figure
        """
        # Create calculator for comparison series if provided
        if comparison_returns is not None:
            comp_calc = RiskMetricsCalculator(comparison_returns, name=comparison_name)
        else:
            comp_calc = None
        
        # Create figure with 1 or 2 subplots
        if comp_calc is None:
            fig, ax = plt.subplots(figsize=figsize)
            axes = [ax]
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)
            axes = [ax1, ax2]
        
        # Function to plot a single distribution
        def plot_distribution(ax, calculator, color="steelblue"):
            # Plot return distribution
            sns.histplot(calculator.returns, bins=50, kde=True, stat="density", 
                       ax=ax, color=color, alpha=0.6, label="Returns")
            
            # Calculate and plot VaR and ES
            p_var = calculator.calculate_parametric_var(confidence_level, time_horizon, dist)
            p_es = calculator.calculate_parametric_es(confidence_level, time_horizon, dist)
            h_var = calculator.calculate_historical_var(confidence_level)
            h_es, _ = calculator.calculate_historical_es(confidence_level)
            
            ax.axvline(-p_var, color="red", linestyle="--", 
                     label=f"Param VaR: {p_var:.4f}")
            ax.axvline(-p_es, color="red", linestyle="-", 
                     label=f"Param ES: {p_es:.4f}")
            ax.axvline(-h_var, color="blue", linestyle="--", 
                     label=f"Hist VaR: {h_var:.4f}")
            ax.axvline(-h_es, color="blue", linestyle="-", 
                     label=f"Hist ES: {h_es:.4f}")
            
            # Add theoretical distribution
            x = np.linspace(min(calculator.returns) * 1.5, max(calculator.returns) * 1.5, 1000)
            if dist == "normal":
                stats = calculator._get_stats(dist)
                y = scipy.stats.norm.pdf(x, stats.mean * time_horizon, stats.std * np.sqrt(time_horizon))
                ax.plot(x, y, "g--", alpha=0.5, label=f"Normal Dist")
            else:  # t-distribution
                stats = calculator._get_stats(dist)
                df = stats.degrees_of_freedom
                scale = stats.scale * np.sqrt(time_horizon)
                y = scipy.stats.t.pdf(x, df, loc=stats.mean * time_horizon, scale=scale)
                ax.plot(x, y, "g--", alpha=0.5, label=f"t-Dist (df={df:.1f})")
            
            # Add styling
            ax.set_title(f"{calculator.name}", fontsize=14)
            ax.set_xlabel("Return", fontsize=12)
            ax.set_ylabel("Density", fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        # Plot primary series
        plot_distribution(axes[0], self, color="steelblue")
        
        # Plot comparison series if provided
        if comp_calc is not None:
            plot_distribution(axes[1], comp_calc, color="darkorange")
            
            # Add overall title comparing the two series
            fig.suptitle(f"Distribution Comparison ({confidence_level*100:.0f}% CL, {time_horizon}-day)", 
                        fontsize=16)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95] if comp_calc else [0, 0, 1, 1])
        
        return fig

    # ---------- Educational Methods ---------- #
    
    @staticmethod
    def explain_var_es_concepts() -> str:
        """
        Provide educational explanation of VaR and ES concepts.
        
        Returns:
        --------
        str
            Detailed explanation of VaR and ES methodologies and interpretations
        """
        return """
        Value at Risk (VaR) and Expected Shortfall (ES) Explained
        ========================================================
        
        Value at Risk (VaR)
        -------------------
        VaR measures the potential loss in value of a portfolio over a defined period 
        for a given confidence interval. For example, a one-day 95% VaR of 1.5% means 
        there is a 5% chance that the portfolio will lose more than 1.5% of its value 
        in a single day, assuming current market conditions.
        
        Two main approaches to calculating VaR:
        
        1. Parametric VaR: Assumes returns follow a specific probability distribution 
           (usually normal or Student's t). It's calculated using the statistical 
           properties of this distribution (mean and standard deviation).
           
           Formula: VaR = -(μ * T + z * σ * sqrt(T))
           where:
           - μ is the mean return
           - σ is the standard deviation of returns
           - z is the Z-score corresponding to the chosen confidence level
           - T is the time horizon in days
        
        2. Historical VaR: Makes no assumptions about the distribution. Instead, it uses 
           actual historical returns to determine the loss threshold. It's simply the 
           relevant percentile of the empirical distribution.
        
        Expected Shortfall (ES)
        ----------------------
        ES addresses a key limitation of VaR - it doesn't tell you how severe losses 
        could be when they exceed the VaR threshold. ES measures the expected loss 
        given that the loss exceeds the VaR level.
        
        For example, if the 95% VaR is 1.5%, the 95% ES would answer: "When we encounter 
        a loss worse than the 95% VaR, what's the average loss we can expect?"
        
        Calculation approaches:
        
        1. Parametric ES: For normally distributed returns:
           ES = -(μ*T - σ*√T * φ(Φ^(-1)(α)) / (1-α))
           where:
           - φ is the PDF of standard normal
           - Φ^(-1) is the quantile function of standard normal
           - α is the confidence level
        
        2. Historical ES: Average of all returns that are worse than the VaR threshold.
        
        Key Differences Between VaR and ES
        ---------------------------------
        1. VaR tells you "how bad things can get" at a specific confidence level.
        2. ES tells you "how bad things are on average when they exceed your VaR threshold."
        3. ES is more sensitive to extreme events ("tail risk").
        4. ES is a coherent risk measure, while VaR is not (it doesn't always satisfy the 
           subadditivity property).
        
        Distribution Assumptions
        -----------------------
        - Normal distribution is simpler but often underestimates tail risk in financial markets.
        - Student's t-distribution typically provides a better fit for financial returns, capturing
          the "fat tails" observed in real market data.
        - The square-root-of-time rule for scaling risk metrics assumes returns are i.i.d.
          (independent and identically distributed), which may not hold over longer horizons.
        
        Limitations and Considerations
        ----------------------------
        - VaR and ES are backward-looking and rely on historical data; they may fail to
          anticipate unprecedented market conditions.
        - Parametric approaches assume a specific distribution, which may not reflect reality.
        - Historical approaches assume the future will resemble the past.
        - Both metrics should be complemented with stress testing and scenario analysis.
        """
    
    @staticmethod
    def explain_risk_management_applications() -> str:
        """
        Provide educational explanation of risk management applications.
        
        Returns:
        --------
        str
            Detailed explanation of how VaR and ES are applied in risk management
        """
        return """
        Applications of VaR and ES in Risk Management
        ===========================================
        
        Regulatory Requirements
        ---------------------
        - Basel III/IV: Banks use VaR and increasingly ES for market risk capital requirements.
        - Solvency II: Insurance companies use similar metrics for risk-based capital.
        - UCITS/AIFMD: Investment funds use VaR for risk monitoring and disclosure.
        
        Internal Risk Management
        ----------------------
        - Setting Risk Limits: Firms establish VaR/ES limits for desks or portfolios.
        - Risk-Adjusted Performance: Metrics like RAROC (Risk-Adjusted Return on Capital)
          incorporate VaR/ES in performance evaluation.
        - Capital Allocation: VaR/ES help determine how much capital to allocate to different
          business units based on their risk profiles.
        
        Portfolio Management
        ------------------
        - Asset Allocation: Optimizing portfolios based on risk metrics rather than just volatility.
        - Diversification Analysis: Comparing standalone vs. portfolio VaR to assess diversification benefits.
        - Stress Testing: Using historical worst-case scenarios to complement VaR/ES estimates.
        
        Best Practices
        ------------
        - Use Multiple Metrics: Combine VaR, ES, stress tests, and scenario analysis.
        - Backtesting: Regularly test VaR estimates against actual outcomes to validate models.
        - Consider Multiple Time Horizons: Analyze both short-term and long-term risk exposures.
        - Transparency: Clearly communicate assumptions and limitations of risk metrics.
        - Supplementary Analysis: Consider other factors like liquidity risk that may not be
          fully captured by VaR/ES.
        
        Challenges and Limitations
        ------------------------
        - Model Risk: All models are simplifications of reality and subject to error.
        - Tail Dependencies: Standard approaches may underestimate joint extreme events.
        - Dynamic Market Conditions: Risk characteristics change over time, particularly in crises.
        - Liquidity Considerations: Standard VaR/ES don't account for liquidity risk in stressed markets.
        - Procyclicality: Risk measures tend to be low during calm periods and spike during stress.
        """

# --- Helper Functions for Interactive Mode ---

def display_risk_metrics(calculator: RiskMetricsCalculator, confidence_level: float, 
                        time_horizon: int, dist: str = "normal"):
    """Display all calculated risk metrics in a clear format."""
    print("\n" + "=" * 50)
    print(f"Risk Metrics for {calculator.name}")
    print("=" * 50)
    print(f"Confidence Level: {confidence_level*100:.1f}%")
    print(f"Time Horizon: {time_horizon} day(s)")
    print(f"Distribution Model: {dist}")
    print("-" * 50)
    
    # Calculate and display metrics
    p_var = calculator.calculate_parametric_var(confidence_level, time_horizon, dist)
    h_var = calculator.calculate_historical_var(confidence_level)
    p_es = calculator.calculate_parametric_es(confidence_level, time_horizon, dist)
    h_es, h_es_n = calculator.calculate_historical_es(confidence_level)
    
    # Format and display values
    print(f"{'Parametric VaR:':<25} {p_var:>8.4f} ({p_var*100:>6.2f}%)")
    print(f"{'Historical VaR:':<25} {h_var:>8.4f} ({h_var*100:>6.2f}%)")
    print(f"{'Parametric ES:':<25} {p_es:>8.4f} ({p_es*100:>6.2f}%)")
    print(f"{'Historical ES:':<25} {h_es:>8.4f} ({h_es*100:>6.2f}%)")
    
    # Add observation count for historical ES
    print(f"(Historical ES based on {h_es_n} observations, {h_es_n/calculator._stats_normal.n*100:.1f}% of data)")
    print("=" * 50)

def parse_text_to_list(text: str, separator: Optional[str] = None) -> List[float]:
    """Parse numerical data from text input to a list of floats."""
    try:
        if separator:
            values = [float(x.strip()) for x in text.split(separator) if x.strip()]
        else:
            values = [float(x.strip()) for x in text.splitlines() if x.strip()]
        
        if not values:
            raise ValueError("No valid numbers found in input.")
        
        return values
    except ValueError as e:
        print(f"Error parsing numbers: {e}")
        raise

def generate_example_data(n_days: int = 252, seed: int = 42) -> pd.Series:
    """Generate example return data with realistic financial market characteristics."""
    np.random.seed(seed)
    
    # Base returns with slight positive drift
    mean_return = 0.0005  # ~12% annualized
    std_dev = 0.01  # ~16% annualized volatility
    base_returns = np.random.normal(mean_return, std_dev, n_days)
    
    # Add negative skew (more extreme negative returns than positive)
    neg_skew = np.random.exponential(0.02, n_days) * np.random.choice([-1, 0], n_days, p=[0.05, 0.95])
    
    # Add occasional positive jumps (less frequent but possible)
    pos_jumps = np.random.exponential(0.015, n_days) * np.random.choice([1, 0], n_days, p=[0.03, 0.97])
    
    # Combine components
    returns = base_returns + neg_skew + pos_jumps
    
    # Convert to pandas Series
    return pd.Series(returns, name="Example Returns")

def run_example():
    """Run a demonstration with sample data."""
    print("\n" + "=" * 60)
    print("Running Example with Simulated Financial Returns")
    print("=" * 60)
    
    # Generate example returns (approximately 1 year of daily returns)
    returns = generate_example_data(252)
    print(f"Generated {len(returns)} days of simulated returns")
    
    # Create calculator
    calculator = RiskMetricsCalculator(returns, name="Sample Portfolio")
    
    # Display metrics for different confidence levels and time horizons
    for conf_level in [0.95, 0.99]:
        for time_horizon in [1, 10]:
            display_risk_metrics(calculator, conf_level, time_horizon, "normal")
    
    # Generate and display summary report
    print("\nSummary Report (Normal Distribution):")
    summary_normal = calculator.generate_risk_report(confidence_levels=[0.9, 0.95, 0.975, 0.99], time_horizon=1, dist="normal")
    print(summary_normal.to_string())
    
    if calculator._t_distribution_available:
        print("\nSummary Report (t-Distribution):")
        summary_t = calculator.generate_risk_report(confidence_levels=[0.9, 0.95, 0.975, 0.99], time_horizon=1, dist="t")
        print(summary_t.to_string())
    
    # Generate and display plots
    print("\nGenerating plots... (close each plot window to continue)")
    
    # Distribution plot
    dist_fig = calculator.plot_return_distribution(confidence_levels=[0.95, 0.99], time_horizon=1)
    plt.show()
    
    # Confidence sensitivity plot
    conf_fig = calculator.plot_confidence_sensitivity(time_horizon=1)
    plt.show()
    
    # Time horizon sensitivity plot
    time_fig = calculator.plot_time_horizon_sensitivity(time_horizons=list(range(1, 11)))
    plt.show()
    
    # Display educational information
    print("\nBasic Concepts of VaR and ES:")
    print(calculator.explain_var_es_concepts())
    
    print("\nPractical Applications in Risk Management:")
    print(calculator.explain_risk_management_applications())

def interactive_mode():
    """Interactive mode guiding the user through risk metric calculation and visualization."""
    print("\n" + "=" * 60)
    print("VaR and Expected Shortfall Calculator - Interactive Mode")
    print("=" * 60)
    print("This tool helps you calculate and visualize risk metrics for financial data.")
    
    calculator = None
    
    while calculator is None:
        print("\nHow would you like to input data?")
        print("1. Enter returns directly (e.g., 0.01 for 1% return)")
        print("2. Enter prices (to be converted to returns)")
        print("3. Load from CSV file")
        print("4. Use example data")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        try:
            if choice == "1":
                print("\nEnter returns, one per line. Enter a blank line when done.")
                print("Example: 0.01 for 1% return, -0.02 for -2% return")
                
                lines = []
                while True:
                    line = input().strip()
                    if not line:
                        break
                    lines.append(line)
                
                returns = parse_text_to_list("\n".join(lines))
                name = input("Enter a name for this series (optional): ").strip() or None
                calculator = RiskMetricsCalculator(returns, name=name)
            
            elif choice == "2":
                print("\nEnter prices, one per line. Enter a blank line when done.")
                
                lines = []
                while True:
                    line = input().strip()
                    if not line:
                        break
                    lines.append(line)
                
                prices = parse_text_to_list("\n".join(lines))
                log_returns = input("Use logarithmic returns? (y/n, default=y): ").strip().lower() != "n"
                name = input("Enter a name for this series (optional): ").strip() or None
                calculator = RiskMetricsCalculator.from_prices(prices, log_returns=log_returns, name=name)
            
            elif choice == "3":
                file_path = input("Enter path to CSV file: ").strip()
                price_col = input("Enter name of price column (default='Close'): ").strip() or "Close"
                date_col = input("Enter name of date column (optional, default='Date'): ").strip() or "Date"
                log_returns = input("Use logarithmic returns? (y/n, default=y): ").strip().lower() != "n"
                name = input("Enter a name for this series (optional): ").strip() or None
                
                calculator = RiskMetricsCalculator.from_csv(
                    file_path, price_column=price_col, date_column=date_col, 
                    log_returns=log_returns, name=name
                )
            
            elif choice == "4":
                n_days = int(input("Enter number of days to simulate (default=252): ").strip() or "252")
                seed = int(input("Enter random seed (default=42): ").strip() or "42")
                
                returns = generate_example_data(n_days, seed)
                calculator = RiskMetricsCalculator(returns, name="Example Data")
            
            elif choice == "5":
                print("Exiting. Goodbye!")
                return
            
            else:
                print("Invalid choice. Please enter a number from 1 to 5.")
        
        except Exception as e:
            print(f"Error: {e}")
            calculator = None
    
    # Get calculation parameters
    print("\nEnter parameters for calculation:")
    conf_level_str = input("Confidence level (e.g., 0.95 for 95%, default=0.95): ").strip() or "0.95"
    time_horizon_str = input("Time horizon in days (default=1): ").strip() or "1"
    dist_choice = input("Distribution model (normal/t, default=normal): ").strip().lower() or "normal"
    
    try:
        conf_level = float(conf_level_str)
        time_horizon = int(time_horizon_str)
        dist = "normal" if dist_choice not in ["normal", "t"] else dist_choice
        
        if not 0 < conf_level < 1:
            print("Warning: Invalid confidence level. Using default 0.95.")
            conf_level = 0.95
        
        if time_horizon <= 0:
            print("Warning: Invalid time horizon. Using default 1.")
            time_horizon = 1
        
        if dist == "t" and not calculator._t_distribution_available:
            print("Warning: t-distribution not available. Using normal distribution.")
            dist = "normal"
    
    except ValueError:
        print("Error in parameters. Using defaults: 95% confidence, 1-day horizon, normal distribution.")
        conf_level = 0.95
        time_horizon = 1
        dist = "normal"
    
    # Display risk metrics
    display_risk_metrics(calculator, conf_level, time_horizon, dist)
    
    # Generate and display summary report
    report = calculator.generate_risk_report(
        confidence_levels=[0.9, conf_level, 0.99], 
        time_horizon=time_horizon, 
        dist=dist
    )
    print("\nSummary Report:")
    print(report.to_string())
    
    # Offer visualization options
    while True:
        print("\nVisualization Options:")
        print("1. Return distribution with VaR and ES")
        print("2. Sensitivity to confidence level")
        print("3. Sensitivity to time horizon")
        print("4. Educational explanation of VaR and ES")
        print("5. Applications in risk management")
        print("6. Exit")
        
        viz_choice = input("\nEnter choice (1-6): ").strip()
        
        if viz_choice == "1":
            conf_levels_str = input("Enter confidence levels to show (comma-separated, default=0.95,0.99): ").strip() or "0.95,0.99"
            try:
                conf_levels = [float(cl) for cl in conf_levels_str.split(",")]
                calculator.plot_return_distribution(confidence_levels=conf_levels, time_horizon=time_horizon, dist=dist)
                plt.show()
            except Exception as e:
                print(f"Error generating plot: {e}")
        
        elif viz_choice == "2":
            calculator.plot_confidence_sensitivity(time_horizon=time_horizon, dist=dist)
            plt.show()
        
        elif viz_choice == "3":
            max_horizon = int(input("Enter maximum time horizon to analyze (default=10): ").strip() or "10")
            horizons = list(range(1, max_horizon + 1))
            calculator.plot_time_horizon_sensitivity(time_horizons=horizons, confidence_level=conf_level, dist=dist)
            plt.show()
        
        elif viz_choice == "4":
            print("\n" + calculator.explain_var_es_concepts())
        
        elif viz_choice == "5":
            print("\n" + calculator.explain_risk_management_applications())
        
        elif viz_choice == "6":
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please enter a number from 1 to 6.")

if __name__ == "__main__":
    # Uncomment the desired mode:
    
    # For a quick example with simulated data:
    # run_example()
    
    # For interactive user interface:
    interactive_mode()