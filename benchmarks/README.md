# Benchmarks and Performance Analysis

This directory contains all benchmarking, profiling, and performance analysis scripts and their results.

## 📁 Directory Structure

```
benchmarks/
├── scripts/                    # All benchmark and profiling scripts
│   ├── profiling/             # Performance profiling scripts
│   │   ├── profile_phase_analysis.py
│   │   └── profile_phase_detector.py
│   ├── benchmarking/          # Benchmarking scripts
│   │   ├── benchmark_dask_performance.py
│   │   └── benchmark_phase_detection.py
│   ├── analysis/              # Analysis and accuracy testing scripts
│   │   └── analyze_cordic_accuracy.py
│   └── benchmark.bat          # Batch script for running benchmarks
└── results/                   # All benchmark and profiling results
    ├── *.prof                 # Profiling output files
    ├── *.json                 # Benchmark results (JSON format)
    ├── *.png                  # Generated plots and visualizations
    └── benchmark_plots/       # Plot output directory
        ├── method_comparison.png
        └── performance_comparison.png
```

## 🚀 Usage

### Running Profiling Scripts
```bash
# Profile phase analysis performance
python benchmarks/scripts/profiling/profile_phase_analysis.py

# Profile phase detector performance
python benchmarks/scripts/profiling/profile_phase_detector.py
```

### Running Benchmark Scripts
```bash
# Benchmark Dask performance
python benchmarks/scripts/benchmarking/benchmark_dask_performance.py

# Benchmark phase detection algorithms
python benchmarks/scripts/benchmarking/benchmark_phase_detection.py
```

### Running Analysis Scripts
```bash
# Analyze CORDIC accuracy
python benchmarks/scripts/analysis/analyze_cordic_accuracy.py
```

### Running All Benchmarks
```bash
# Windows batch script
benchmarks/scripts/benchmark.bat
```

## 📊 Results

- **Profiling Results**: `.prof` files contain detailed performance profiling data
- **Benchmark Results**: `.json` files contain structured benchmark data
- **Visualizations**: `.png` files contain performance comparison charts
- **Plot Directory**: `benchmark_plots/` contains organized plot outputs

## 🔧 Script Categories

### Profiling Scripts
- **Purpose**: Identify performance bottlenecks in core algorithms
- **Output**: `.prof` files for detailed analysis
- **Tools**: cProfile, snakeviz

### Benchmarking Scripts
- **Purpose**: Compare performance across different methods and configurations
- **Output**: `.json` files with structured results
- **Metrics**: Execution time, memory usage, accuracy

### Analysis Scripts
- **Purpose**: Validate algorithm accuracy and correctness
- **Output**: `.png` visualizations and accuracy reports
- **Focus**: CORDIC algorithm accuracy, signal processing validation

## 📈 Performance Insights

Based on recent benchmarks:
- **SignalStacker**: 71% improvement with NumPy vectorization
- **PhaseChangeDetector**: 65% improvement with optimized CORDIC
- **CORDIC Algorithm**: 98% reduction in computation time for large arrays
- **Dask Integration**: Efficient parallel processing for large datasets

## 🛠️ Maintenance

- **Regular Updates**: Run benchmarks after major algorithm changes
- **Result Cleanup**: Archive old results periodically
- **Script Updates**: Update paths when moving files
- **Documentation**: Keep this README updated with new scripts and results
