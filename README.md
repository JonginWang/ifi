<div align="center">
    <img src="./ifi/images/IFI_icon01.png" width="300">
</div>

# IFI (Interferometer Data Analysis) - MVP

This project provides a graphical user interface (GUI) to automate the process of transferring waveform data from a Tektronix MDO3000 series oscilloscope's USB stick to a computer.

This project provides a comprehensive data analysis platform for interferometer measurements, featuring automated data acquisition, processing, and visualization capabilities.

## 🚀 MVP Features

### 1. **Data Acquisition & Management**
- **Tektronix Oscilloscope Integration:** Automated data transfer from MDO3000 series oscilloscopes
- **GUI Interface:** Easy-to-use interface to start and stop the automation process.
- **State Machine Logic:** Robustly handles the process of connecting, listing files, transferring 
data, and deleting files from the USB drive.
- **NAS Database Integration:** Seamless connection to network-attached storage systems
- **VEST Database Support:** Integration with VEST (Vessel Experiment Shot Tracking) database
- **HDF5 Caching System:** Efficient data storage and retrieval with metadata preservation

### 2. **Advanced Signal Processing**
- **Complex Demodulation (CDM):** Phase-to-density conversion for interferometer data
- **Spectral Analysis:** FFT, STFT, and CWT (Continuous Wavelet Transform) capabilities
- **Ridge Extraction:** Time-frequency ridge detection using ssqueezepy
- **Filter Design:** Custom FIR filter design with remezord optimization

### 3. **Interactive Visualization**
- **Multi-format Plotting:** Waveforms, spectrograms, density evolution, and response plots
- **Interactive Mode:** Real-time plot manipulation and analysis
- **Metadata Integration:** Automatic display of shot information and analysis parameters
- **Export Capabilities:** High-resolution plot export in multiple formats

### 4. **Robust Architecture**
- **Modular Design:** Separated concerns for GUI, analysis, and data management
- **Error Handling:** Comprehensive exception handling and logging
- **Performance Optimization:** Numba JIT compilation and parallel processing
- **Cross-platform Support:** Windows, macOS, and Linux compatibility

## 🔧 Technical Capabilities

### **Data Processing Pipeline:**
1. **Raw Data Acquisition** → Tektronix scope data transfer
2. **Data Validation** → Format checking and quality assessment  
3. **Signal Processing** → CDM analysis, filtering, and spectral decomposition
4. **Density Calculation** → Phase-to-density conversion with baseline correction
5. **Visualization** → Interactive plotting with metadata display
6. **Results Storage** → HDF5 caching with automatic directory management

### **Supported Data Formats:**
- **Input:** CSV, HDF5, Tektronix native formats
- **Output:** PNG, PDF, HDF5 with metadata preservation
- **Caching:** HDF5 with JSON-serialized metadata

## 🚀 Quick Start

### **Prerequisites:**
- Python 3.8+ (recommended: Python 3.10)
- Git (for cloning the repository)

### **Installation:**

1. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd ifi
   ```

2. **Create Virtual Environment (Recommended):**
   ```bash
   # Create virtual environment
   python -m venv .venv
   
   # Activate (Windows)
   .venv\Scripts\activate
   
   # Activate (macOS/Linux)
   source .venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application:**
   ```bash
   # GUI Mode
   python -m ifi.main
   
   # Command Line Analysis
   python -m ifi.analysis.main_analysis
   
   # Interactive Analysis
   python -m ifi.analysis.interactive_analysis
   ```

### **Configuration:**
- Copy `ifi/config.ini.template` to `ifi/config.ini`
- Configure database connections and analysis parameters
- Set up NAS credentials for remote data access

## 📁 Project Structure

```
ifi/                                    # Project root
├── 📁 cache/                          # HDF5 cached data storage
│   ├── 40245/                         # Shot-specific cache directories
│   ├── 45821/
│   └── numba_cache/                   # Numba JIT compilation cache
├── 📁 benchmarks/                     # Performance testing and profiling
│   ├── scripts/
│   │   ├── analysis/                  # Analysis benchmarking
│   │   ├── benchmarking/              # Performance benchmarks
│   │   └── profiling/                 # Code profiling scripts
│   └── results/                       # Benchmark results and plots
├── 📁 docs/                          # Documentation
│   ├── build/                        # Sphinx documentation build
│   ├── source/                       # Documentation source files
│   ├── guide_*.md                    # User guides
│   └── report_*.md                   # Analysis reports
├── 📁 ifi/                           # Main package
│   ├── 📁 analysis/                  # Core analysis modules
│   │   ├── functions/                # Analysis utility functions
│   │   ├── params/                   # Analysis parameters
│   │   ├── main_analysis.py          # Main analysis pipeline
│   │   ├── phase_analysis.py         # Phase analysis algorithms
│   │   ├── phi2ne.py                 # Phase-to-density conversion
│   │   ├── plots.py                  # Visualization and plotting
│   │   ├── processing.py             # Signal processing utilities
│   │   └── spectrum.py               # Spectral analysis
│   ├── 📁 db_controller/             # Database management
│   │   ├── nas_db.py                 # NAS database interface
│   │   └── vest_db.py                # VEST database interface
│   ├── 📁 gui/                       # Graphical user interface
│   │   └── main_window.py            # Main GUI application
│   ├── 📁 tek_controller/            # Tektronix oscilloscope control
│   │   └── scope.py                  # Scope communication interface
│   ├── 📁 utils/                     # Utility modules
│   │   ├── cache_setup.py            # Cache configuration
│   │   ├── common.py                 # Common utilities
│   │   ├── file_io.py                # File I/O operations
│   │   └── validation.py             # Data validation
│   ├── 📁 images/                    # Application icons and images
│   ├── 📁 olds/                      # Legacy code archive
│   ├── 📁 under_dev/                 # Development code
│   ├── config.ini                    # Main configuration file
│   └── main.py                       # Application entry point
├── 📁 tests/                         # Test suite
│   ├── analysis/                     # Analysis module tests
│   ├── db_controller/                # Database tests
│   └── utils/                        # Utility tests
├── 📁 results/                       # Analysis results output
├── 📁 dummy/                         # Development and testing scripts
├── 📁 logs/                          # Application logs
├── run.py                            # Alternative entry point
├── ifi.spec                          # PyInstaller specification
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## 🔧 Development & Testing

### **Running Tests:**
```bash
# Run all tests
pytest tests/

# Run specific test modules
pytest tests/analysis/
pytest tests/db_controller/
pytest tests/utils/

# Run with coverage
pytest --cov=ifi tests/
```

### **Code Quality:**
```bash
# Lint checking
ruff check ifi/

# Format code
ruff format ifi/

# Type checking
mypy ifi/
```

### **Benchmarking:**
```bash
# Run performance benchmarks
python benchmarks/scripts/benchmarking/benchmark_phase_detection.py

# Profile analysis performance
python benchmarks/scripts/profiling/profile_phase_analysis.py
```

## 📦 Building an Executable

Package the application into a single executable using PyInstaller:

### **Prerequisites:**
- Clean virtual environment (recommended)
- All dependencies installed

### **Build Process:**

1. **Prepare Environment:**
   ```bash
   # Create and activate virtual environment
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # macOS/Linux
   
   # Install dependencies
   pip install -r requirements.txt
   pip install pyinstaller
   ```

2. **Build Executable:**
   ```bash
   # Windows
   pyinstaller --onefile --windowed --icon=ifi/images/IFI_icon01.ico --add-data "ifi/config.ini;ifi" run.py --name IFI_Analyzer
   
   # macOS/Linux
   pyinstaller --onefile --windowed --icon=ifi/images/IFI_icon01.ico --add-data "ifi/config.ini:ifi" run.py --name IFI_Analyzer
   ```

3. **Output:**
   - Executable created in `dist/` folder
   - Includes all dependencies and configuration files
   - Standalone application (no Python installation required)

## 📚 Documentation

- **API Documentation:** Available in `docs/build/` (generated with Sphinx)
- **User Guides:** Located in `docs/guide_*.md`
- **Analysis Reports:** Available in `docs/report_*.md`
- **Code Documentation:** Inline docstrings throughout the codebase

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and ensure they pass
5. Submit a pull request

## 📄 License

This project is licensed under the terms specified in `LICENSE.md`. 
