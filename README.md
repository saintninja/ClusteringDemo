# Clustering - C++23 clustering toolkit with quality metrics

> 🇷🇺 Русская версия документации находится ниже: [перейти](/ReadMe_RU.md)

**Clustering** is a small C++23 project that clusters tabular data from CSV files and measures how good the result is. It contains four algorithms - **k-means (Lloyd)**, **EM for Gaussian mixtures (EM-GMM)**, **DBSCAN** and **Affinity Propagation** - four quality metrics (**ARI, AMI, silhouette coefficient, Davies-Bouldin index**) and a Python script that draws the result. The code is written as a teaching example: the points of variation of k-means (initialization and distance) are interchangeable strategies, and every algorithm hides behind one interface.

The command line program reads a CSV file, clusters the selected numeric columns, writes a **new** CSV file with a `cluster` column (the input is never modified) and prints a report with the metrics.

---

## Table of contents

- [Clustering - C++23 clustering toolkit with quality metrics](#clustering---c23-clustering-toolkit-with-quality-metrics)
  - [Table of contents](#table-of-contents)
  - [Features](#features)
  - [Requirements](#requirements)
  - [Project structure](#project-structure)
  - [Building](#building)
    - [Windows / Visual Studio](#windows--visual-studio)
    - [Windows / MinGW (MSYS2)](#windows--mingw-msys2)
    - [Linux](#linux)
    - [macOS](#macos)
    - [CMake options](#cmake-options)
    - [Running the tests](#running-the-tests)
  - [Quick start](#quick-start)
  - [Main program (clustering\_cli)](#main-program-clustering_cli)
    - [Synopsis](#synopsis)
    - [Required options](#required-options)
    - [Optional options](#optional-options)
    - [Which options apply to which algorithm](#which-options-apply-to-which-algorithm)
    - [Exit codes](#exit-codes)
  - [Input data format](#input-data-format)
  - [Output file and console report](#output-file-and-console-report)
  - [Quality metrics](#quality-metrics)
  - [Semi-supervised mode (SSL)](#semi-supervised-mode-ssl)
  - [Visualizer (scripts/visualize.py)](#visualizer-scriptsvisualizepy)
    - [Visualizer synopsis](#visualizer-synopsis)
    - [Visualizer required options](#visualizer-required-options)
    - [Visualizer optional options](#visualizer-optional-options)
    - [What the picture shows](#what-the-picture-shows)
  - [Usage scenarios](#usage-scenarios)
  - [Test data](#test-data)
  - [Architecture and design patterns](#architecture-and-design-patterns)
  - [API cheat-sheet](#api-cheat-sheet)
  - [Troubleshooting](#troubleshooting)

---

## Features

- **Four clustering algorithms** behind one interface (`Clusterer`):
  - **`kmeans`** - Lloyd's k-means. Two points of variation are strategies: the **initializer** (`forgy`, `kmeans++`, `seeded`/`ssl`) and the **distance** (`euclidean`, `manhattan`, `mahalanobis`).
  - **`gmm`** - EM algorithm for a Gaussian mixture with full covariance matrices. The initial centers come from the same initializers as in k-means.
  - **`dbscan`** - density-based clustering with the radius `epsilon` and the neighbor threshold `minPts`; outliers get the label `-1`. Works with any of the three distances.
  - **`affinity`** - Affinity Propagation: the number of clusters is not given, it follows from the `preference` parameter.
- **Semi-supervised k-means (SSL)**: a labelled training set defines the initial centers and its labels stay fixed during clustering; the training set is merged into the output file. Also available for `gmm`.
- **Four quality metrics**: ARI and AMI (compare with ground truth), silhouette coefficient and Davies-Bouldin index (use only the data and the found clusters). The implementation was cross-checked against scikit-learn.
- **Robust CSV input**: UTF-8 BOM, Windows line endings, text class labels, missing values (`NA`, empty cells) - such rows are skipped and counted.
- **Optional preprocessing**: z-score standardization (`--standardize`) and random sub-sampling (`--sample`).
- **Visualizer** in Python: scatter plot coloured by cluster, noise as grey crosses, training objects with a black outline, PCA projection for more than two features.
- **Modern C++23**: `std::expected`, `std::span`, `std::format`, designated initializers; no third-party dependencies in the C++ part.
- **Unit tests** (54 tests, CTest) including end-to-end runs on the sample data.

## Requirements

| Component            | Version / notes                                                          |
|----------------------|--------------------------------------------------------------------------|
| CMake                | ≥ 3.20                                                                   |
| C++ compiler         | C++23 with `std::expected` and `std::format`: GCC 13+, Clang 16+ (libstdc++ 13+ or libc++ 17+). Tested with GCC 13.3 (Linux) and GCC 15.2 (MSYS2 MinGW-w64) |
| Python (visualizer)  | 3.9+ with `numpy`, `pandas`, `matplotlib`                                |

## Project structure

```
clustering/
├── CMakeLists.txt
├── data/                           # sample data: boxes.csv, blob.csv, penguins_sex.csv
├── scripts/visualize.py            # visualizer
├── include/clustering/
│   ├── core/                       # Matrix, Cholesky factorization, linear algebra, Result<T>, FixedAssignments
│   ├── distance/                   # Strategy: DistanceMetric -> Euclidean, Manhattan, Mahalanobis + factory
│   ├── initialization/             # Strategy: CentroidInitializer -> Forgy, KMeansPlusPlus, Seeded + TrainingLabels + factory
│   ├── algorithms/                 # Clusterer -> KMeans, GaussianMixture, Dbscan, AffinityPropagation
│   ├── evaluation/                 # ARI, AMI, silhouette, Davies-Bouldin, QualityEvaluator
│   ├── io/                         # CSV reader/writer, dataset loader, label encoder
│   ├── preprocessing/              # standardization
│   └── app/                        # command line, configuration, algorithm factory, pipeline, report
├── src/                            # implementations (mirror include/) and main.cpp
└── tests/                          # unit and end-to-end tests (CTest)
```

## Building

The build is the same on all platforms; only the toolchain differs. The default build type is `Release`.

### Windows / Visual Studio

```bat
cmake -S . -B build
cmake --build build --config Release --parallel
build\Release\clustering_cli.exe --help
```

### Windows / MinGW (MSYS2)

```bat
cmake -S . -B build -G "MinGW Makefiles"
cmake --build build --parallel
build\clustering_cli.exe --help
```

### Linux

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/clustering_cli --help
```

### macOS

```bash
brew install cmake llvm       # Apple Clang may be too old for std::expected / std::format
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=$(brew --prefix llvm)/bin/clang++
cmake --build build --parallel
./build/clustering_cli --help
```

### CMake options

| Option                           | Default   | Description                                                      |
|----------------------------------|-----------|------------------------------------------------------------------|
| `CLUSTERING_CXX_STANDARD`        | `23`      | C++ standard: `23` or `26` (26 needs CMake ≥ 3.25 and a compiler that supports it) |
| `CLUSTERING_BUILD_TESTS`         | `ON`      | build the test executable `clustering_tests`                     |
| `CLUSTERING_WARNINGS_AS_ERRORS`  | `OFF`     | treat compiler warnings as errors (`-Werror` / `/WX`)            |

Example: `cmake -S . -B build -DCLUSTERING_CXX_STANDARD=26 -DCLUSTERING_WARNINGS_AS_ERRORS=ON`.
The program can be installed with `cmake --install build --prefix <dir>` (the executable goes to `<dir>/bin`).

### Running the tests

```bash
ctest --test-dir build --output-on-failure      # add -C Release for Visual Studio
```

The tests read the sample files from `data/`; the path is fixed when CMake configures the project, so re-run CMake if you move the project.

## Quick start

```bash
# k-means with k-means++ initialization on the labelled blob data (ARI/AMI need --labels)
clustering_cli --input data/blob.csv --output blob_kmeans.csv --columns x,y --labels color --algorithm kmeans --k 4 --init kmeans++

# draw the result
python scripts/visualize.py blob_kmeans.csv -c x y
```

Example report (numbers can differ slightly between compilers and standard libraries):

```
Algorithm:       k-means (k=4, initializer: kmeans++, metric: euclidean)
Objects:         4086 (rows skipped because of missing values: 0)
Clusters found:  4, noise objects: 0
Iterations:      10 (converged)
Time:            0.82 ms
Quality:
  ARI:             0.8779
  AMI:             0.8457
  Silhouette:      0.4720
  Davies-Bouldin:  0.6871
Result written to blob_kmeans.csv
```

---

## Main program (clustering_cli)

### Synopsis

```
clustering_cli --input <file.csv> --output <result.csv> --columns <c1,c2,...> [options]
clustering_cli --help
```

Options are written as `--name value` (one value each); the only flag without a value is `--standardize`. The order of options does not matter. Running the program without arguments or with `--help` prints the usage text.

### Required options

| Option                  | Description                                                                     |
|-------------------------|---------------------------------------------------------------------------------|
| `--input <file>`        | input CSV file (with a header row)                                              |
| `--output <file>`       | output CSV file; **must differ from the input files** (the program refuses to overwrite them) |
| `--columns <c1,c2,...>` | comma-separated names of the numeric feature columns (no spaces); two or more columns are normal, one is allowed |

Depending on the algorithm there are additional required options:

| Situation                                | Required option                                                |
|------------------------------------------|----------------------------------------------------------------|
| `--algorithm kmeans` or `gmm`            | `--k <clusters>` - number of clusters / mixture components     |
| `--algorithm dbscan`                     | `--eps <radius>` - neighborhood radius                         |
| `--init seeded` (alias `ssl`)            | a training set: `--training <file>` **or** `--training-fraction <f>` |
| `--training <file>`                      | `--training-labels <column>` (or `--labels`, which is then used for both files) |
| `--training-fraction <f>`                | `--labels <column>` - ground-truth classes in the input file   |

### Optional options

**Data and general**

| Option                    | Default     | Description                                                                 |
|---------------------------|-------------|-----------------------------------------------------------------------------|
| `--labels <column>`       | -           | column of the input file with ground-truth classes (any text or numbers). Enables **ARI and AMI** |
| `--standardize`           | off         | z-score the features (mean 0, standard deviation 1) before clustering. The output file keeps the original values |
| `--sample <n>`            | all rows    | cluster a random sample of `n` objects (the output file then contains only those rows) |
| `--seed <n>`              | `42`        | seed of the random generator (initialization, sampling, training split). The same seed gives the same result on the same compiler/library |

**Algorithm and distances**

| Option                    | Default     | Description                                                                 |
|---------------------------|-------------|-----------------------------------------------------------------------------|
| `--algorithm <name>`      | `kmeans`    | `kmeans`, `gmm`, `dbscan` or `affinity`                                     |
| `--metric <name>`         | `euclidean` | distance used by the algorithm: `euclidean`, `manhattan`, `mahalanobis` (the covariance matrix is estimated from all objects) |
| `--eval-metric <name>`    | `euclidean` | distance used by the silhouette coefficient and Davies-Bouldin index. It is independent of `--metric`, so that different runs stay comparable |

**k-means and EM-GMM**

| Option                    | Default     | Description                                                                 |
|---------------------------|-------------|-----------------------------------------------------------------------------|
| `--k <n>`                 | - (required)| number of clusters (components)                                             |
| `--init <name>`           | `forgy`     | initializer: `forgy` (random objects), `kmeans++` (spread-out objects), `seeded` / `ssl` (class means of a training set) |
| `--max-iter <n>`          | `300`       | iteration limit; if it is reached, the report says "iteration limit reached" |

**DBSCAN**

| Option                    | Default     | Description                                                                 |
|---------------------------|-------------|-----------------------------------------------------------------------------|
| `--eps <r>`               | - (required)| neighborhood radius in the units of the chosen distance (depends on the scale of your data; use `--standardize` for features of different scale) |
| `--min-pts <n>`           | `5`         | minimal number of objects in the neighborhood **including the object itself** for it to be a core object |

**Affinity Propagation**

| Option                    | Default     | Description                                                                 |
|---------------------------|-------------|-----------------------------------------------------------------------------|
| `--damping <d>`           | `0.7`       | damping factor, `0.5 ≤ d < 1`; larger values are more stable but slower     |
| `--preference <p>`        | median of similarities | self-similarity of the objects. **The smaller (more negative), the fewer clusters.** Similarity is `-distance^2` |
| `--max-iter <n>`          | `300`       | iteration limit                                                             |

The method needs O(n²) memory, so at most **6000 objects** are accepted; use `--sample` for bigger data.

**Semi-supervised start (only `kmeans` and `gmm`, see [Semi-supervised mode (SSL)](#semi-supervised-mode-ssl))**

| Option                         | Default | Description                                                              |
|--------------------------------|---------|--------------------------------------------------------------------------|
| `--training <file>`            | -       | CSV file with the labelled training set (the same feature column names as in the input) |
| `--training-labels <column>`   | value of `--labels` | column of the training file with the classes                  |
| `--training-fraction <f>`      | -       | `0 < f < 1`: take a stratified part of the labelled input as the training set and cluster the rest. Mutually exclusive with `--training` |

### Which options apply to which algorithm

| Option                          | `kmeans`   | `gmm`      | `dbscan`   | `affinity` |
|---------------------------------|:----------:|:----------:|:----------:|:----------:|
| `--k`                           | required   | required   | -          | -          |
| `--init`                        | yes        | yes        | -          | -          |
| `--max-iter`                    | yes        | yes        | -          | yes        |
| `--eps`, `--min-pts`            | -          | -          | yes / yes  | -          |
| `--damping`, `--preference`     | -          | -          | -          | yes        |
| `--training*`                   | yes        | yes        | -          | -          |
| `--metric`                      | yes        | initialization only | yes | yes        |
| `--labels`, `--standardize`, `--sample`, `--seed`, `--eval-metric` | yes | yes | yes | yes |

Notes: in `gmm` the distance is used for the initial assignment and for k-means++ weights only - the densities themselves use the estimated covariance matrices. In `dbscan` the radius is measured with `--metric`; in `affinity` the similarities are `-d²` for `--metric`.

### Exit codes

| Code | Meaning                                                              |
|------|----------------------------------------------------------------------|
| `0`  | success (or `--help`)                                                |
| `1`  | runtime error: file not found, unknown column, non-numeric value, invalid parameter for an algorithm, ... (the message is printed to stderr) |
| `2`  | wrong command line: unknown option, missing value or missing required option (the usage text is printed) |

---

## Input data format

- A text file with a **header row**; fields are separated by commas. Quoted fields with commas inside are not supported.
- UTF-8 BOM and Windows (`CRLF`) line endings are accepted; blank lines are ignored.
- Every row must have as many fields as the header.
- Feature columns (`--columns`) must be numeric. A non-numeric value stops the run with a message that names the column and the data row.
- **Missing values** - an empty cell, `NA`, `N/A`, `NaN`, `nan`, `null`, `NULL` - in a feature column (or in the label column, if it is requested) make the program **skip the whole row**. The number of skipped rows is printed in the report. Skipped rows are not written to the output file.
- Class labels (`--labels`, `--training-labels`) can be arbitrary text or numbers; they are only compared for equality, so the names of the classes do not matter.

## Output file and console report

**Output file.** All columns of the input file are copied, then the following columns are added:

| Column    | When         | Meaning                                                                       |
|-----------|--------------|-------------------------------------------------------------------------------|
| `cluster` | always       | cluster number `0, 1, 2, …`; **`-1` means noise** (DBSCAN) or "no exemplars found" (Affinity Propagation). If the name is taken, `_` is appended |
| `origin`  | with a training set | `training` for the training objects, `clustered` for the others           |

**Console report:**

| Line             | Meaning                                                                       |
|------------------|-------------------------------------------------------------------------------|
| `Algorithm`      | the algorithm with the parameters actually used                                |
| `Objects`        | number of clustered objects (and the split into training / clustered, number of skipped rows) |
| `Clusters found` | number of clusters and number of noise objects                                 |
| `Iterations`     | number of iterations and `converged` / `iteration limit reached`               |
| `Time`           | time of the clustering itself (without reading and writing files)              |
| `Quality`        | ARI, AMI, silhouette, Davies-Bouldin (`n/a` - not defined, see below)          |

## Quality metrics

| Metric                    | Needs ground truth | Good value | Notes |
|---------------------------|:------------------:|------------|-------|
| **ARI** (adjusted Rand index) | yes (`--labels`) | close to 1 (0 = random, may be negative) | invariant to the numbering of clusters |
| **AMI** (adjusted mutual information) | yes (`--labels`) | close to 1 | arithmetic-mean normalization, as the default of scikit-learn |
| **Silhouette coefficient** | no               | close to 1 (range -1…1)  | needs at least 2 clusters |
| **Davies-Bouldin index**  | no                 | **smaller is better** (≥ 0) | needs at least 2 clusters |

- `n/a` for ARI/AMI means that `--labels` was not given. `n/a` for silhouette / Davies-Bouldin means fewer than two clusters were found.
- DBSCAN noise (`-1`) is treated as one more class by ARI and AMI (as in scikit-learn) and is **excluded** from the silhouette and Davies-Bouldin computation.
- Silhouette is O(n²), so it is noticeably slower on tens of thousands of objects.
- The distance for the two internal metrics is chosen by `--eval-metric` (Euclidean by default).

## Semi-supervised mode (SSL)

SSL is k-means (and, in the same way, EM-GMM) with a **labelled training set**:

1. The initial centers are the **class means of the training set**. The class with the number `i` in the sorted order of class labels becomes cluster `i`.
2. During clustering the labels of the training objects **never change**, but their coordinates take part in the recalculation of the centers. That is why the clusters can overlap near the borders.
3. The training set is **merged** with the clustered objects: the output file contains the training rows first (`origin = training`), then the other rows (`origin = clustered`).
4. If labels of the clustered objects are known (`--labels`), the report prints the metrics twice: for all objects and for the clustered objects only (the training objects make the first line optimistic).

The training set comes from a separate file or from the labelled input:

```bash
# separate files: unlabelled.csv has no classes, train.csv has the column "color"
clustering_cli --input unlabelled.csv --output result.csv --columns x,y --algorithm kmeans --k 4 --init ssl --training train.csv --training-labels color

# 2% of a labelled file serve as the training set, the rest is clustered
clustering_cli --input data/boxes.csv --output result.csv --columns x,y --labels color --algorithm kmeans --k 30 --init ssl --training-fraction 0.02
```

Rules: the number of training classes must not exceed `--k` (if `--k` is larger, the missing centers are taken from unlabelled objects); `--init ssl` and a training set must be used together; DBSCAN and Affinity Propagation do not support a training set.

---

## Visualizer (scripts/visualize.py)

Draws a CSV file produced by `clustering_cli`: every object is a point, the colour is the cluster.

### Visualizer synopsis

```
python scripts/visualize.py <result.csv> -c <col1> <col2> [more columns] [options]
```

### Visualizer required options

| Option / argument          | Description                                                                |
|----------------------------|----------------------------------------------------------------------------|
| `path` (positional)        | CSV file with the `cluster` column (the output of `clustering_cli`)        |
| `-c`, `--columns <names…>` | **at least two** feature columns, separated by spaces (not commas!). With exactly two columns they are the axes; with more, the objects are projected to the plane with PCA (axes `PC1`, `PC2`) |

### Visualizer optional options

| Option                       | Default   | Description                                                             |
|------------------------------|-----------|-------------------------------------------------------------------------|
| `--cluster-column <name>`    | `cluster` | name of the column with the cluster numbers                              |
| `--origin-column <name>`     | `origin`  | name of the column with `training` / `clustered` (if it exists, training objects are highlighted) |
| `--no-centers`               | off       | do not draw the cluster means. Use it for DBSCAN and other non-convex clusters, where the mean has no meaning |
| `--save <file>`              | -         | save the picture (`.png`, `.pdf`, `.svg`, …) instead of opening a window; needed on a server without a display |

### What the picture shows

- Each cluster has its own colour (10 colours for up to 10 clusters, 20 for more).
- **Grey crosses** - noise (label `-1`).
- **Large `X` markers** - the mean of the objects of the cluster (switch off with `--no-centers`).
- **Black outline** - training objects of the SSL mode.
- The legend is hidden if there are more than 12 groups or a training set is present.
- Rows with missing values in the chosen columns are not drawn.

Examples:

```bash
python scripts/visualize.py result.csv -c x y
python scripts/visualize.py dbscan_result.csv -c x y --no-centers --save dbscan.png
python scripts/visualize.py penguins_result.csv -c culmen_length_mm culmen_depth_mm flipper_length_mm body_mass_g --save penguins_pca.png
```

---

## Usage scenarios

| Goal                                          | Command (shortened)                                                              |
|-----------------------------------------------|----------------------------------------------------------------------------------|
| Compare initializers                          | the same `--k`, `--init forgy` / `kmeans++` / `ssl`, several `--seed` values; compare ARI and silhouette |
| Compare distances                             | `--metric euclidean` / `manhattan` / `mahalanobis`                               |
| Clusters of arbitrary shape and outliers      | `--algorithm dbscan --eps 6 --min-pts 5 --labels color` (try `data/boxes.csv`)   |
| The number of clusters is unknown             | `--algorithm affinity --preference -4000 --sample 2000`; decrease `--preference` to get fewer clusters |
| Overlapping elliptical clusters               | `--algorithm gmm --k 3 --init kmeans++`                                          |
| Part of the objects is labelled               | `--init ssl --training-fraction 0.02 --labels color`                             |
| Multi-dimensional data with different scales  | `--standardize` or `--metric mahalanobis`; see `data/penguins_sex.csv`           |

Example for `penguins_sex.csv` (2 rows with missing values are skipped automatically):

```bash
clustering_cli --input data/penguins_sex.csv --output penguins_gmm.csv --columns culmen_length_mm,culmen_depth_mm,flipper_length_mm,body_mass_g --standardize --algorithm gmm --k 3 --init kmeans++
```

## Test data

| File                    | Content                                                                      |
|-------------------------|------------------------------------------------------------------------------|
| `data/blob.csv`         | 2D, 4086 objects, columns `x,y,color`; 4 overlapping classes                  |
| `data/boxes.csv`        | 2D, 8901 objects, columns `x,y,color`; 30 compact "boxes" (classes)           |
| `data/penguins_sex.csv` | 4 numeric features + `sex`; BOM, 2 rows with missing measurements, and an obvious outlier in `flipper_length_mm` (5000), so standardization or Mahalanobis distance is recommended |

Reference results on these files (seed 42, results may differ slightly between platforms):

| Run                                                           | Clusters | ARI    |
|---------------------------------------------------------------|---------:|-------:|
| `blob.csv`, k-means, `kmeans++`, `--k 4`                      | 4        | 0.878  |
| `boxes.csv`, DBSCAN, `--eps 6 --min-pts 5`                    | 30 (+25 noise) | 0.997 |
| `boxes.csv`, k-means, `ssl`, `--training-fraction 0.02`, `--k 30` | 30   | 0.9995 |

## Architecture and design patterns

| Pattern / technique | Where                                                                                       |
|---------------------|---------------------------------------------------------------------------------------------|
| **Strategy**        | `DistanceMetric` (Euclidean, Manhattan, Mahalanobis) and `CentroidInitializer` (Forgy, k-means++, Seeded) - the two points of variation of k-means |
| **Dependency injection** | `KMeans` and `GaussianMixture` receive the initializer, the metric and the fixed assignments through the constructor; they know only the interfaces |
| **Factory**         | `MakeMetric`, `MakeInitializer`, `MakeClusterer` build objects from the command line settings |
| **Facade**          | `RunPipeline` hides reading, preprocessing, clustering, evaluation and writing behind one call |
| **Value objects, `std::expected`** | `Matrix`, `TrainingLabels`, `Result<T>`: errors of the outer layers (files, command line) are values, invalid parameters of algorithms are exceptions |
| **Open/closed principle** | a new algorithm, metric or initializer is a new class plus one line in a factory       |

How to add something new:

- **A new distance**: derive from `DistanceMetric` (`Distance`, `Name`), add the value to `MetricKind` and a line to `ParseMetricKind` / `MakeMetric`.
- **A new initializer**: derive from `CentroidInitializer` (`Initialize`, `Name`), add it to `InitializerKind` and to `MakeInitializer`.
- **A new algorithm**: derive from `Clusterer` (`Fit`, `Describe`), add it to `AlgorithmKind`, `ParseAlgorithmKind` and `MakeClusterer`.

## API cheat-sheet

| Type                          | Purpose                                                                    |
|-------------------------------|----------------------------------------------------------------------------|
| `Matrix`, `Point`             | row-major matrix of objects; a row is a `std::span<const double>`          |
| `Clusterer`                   | `ClusteringResult Fit(const Matrix&)`, `std::string Describe()`            |
| `ClusteringResult`            | `labels` (`-1` = noise), `centers`, `iterations`, `converged`, `ClusterCount()`, `NoiseCount()` |
| `DistanceMetric`              | `double Distance(Point, Point)`; `MetricPtr` = `std::shared_ptr<const DistanceMetric>` |
| `CentroidInitializer`         | `Matrix Initialize(data, k, metric)`                                       |
| `TrainingLabels`              | classes of the leading training objects; `ToFixedAssignments(n)`           |
| `FixedAssignments`            | `std::vector<std::optional<int>>` - objects whose cluster must not change  |
| `QualityEvaluator`            | `QualityReport Evaluate(data, predicted, truth = {})`                      |
| `RunConfiguration`            | all settings of a run; `ParseCommandLine` fills it, `RunPipeline` executes it |

## Troubleshooting

- **ARI / AMI are `n/a`.** The program does not know the ground truth: add `--labels <column>`. Silhouette and Davies-Bouldin do not need it.
- **`Error: column 'z' not found in the input file`.** The name in `--columns` must match the header exactly (case-sensitive, no spaces around commas). In PowerShell quote the value if it contains special characters.
- **`non-numeric value ... in data row N`.** A feature column contains text. Choose numeric columns or clean the data; empty cells and `NA` are fine (such rows are skipped).
- **`output file must differ from the input files`.** Choose another `--output`.
- **k-means gives a poor result with `forgy`.** Random initial centers can fall into one group. Use `--init kmeans++`, try several `--seed` values, or use the SSL mode.
- **DBSCAN: everything is one cluster or everything is noise.** `--eps` depends on the scale of the data: decrease it for one big cluster, increase it for all-noise. Standardize features of different scale (`--standardize`).
- **Affinity Propagation: too many clusters / `iteration limit reached`.** Decrease `--preference` (e.g. `-4000`, `-60000`), increase `--damping` (0.8–0.9) and `--max-iter`.
- **`affinity propagation needs O(n^2) memory ...`.** More than 6000 objects: add `--sample 2000`.
- **`covariance matrix of the data is not positive definite`** (`--metric mahalanobis`). Two columns are linearly dependent; remove one of them.
- **`training set has more classes than requested clusters`.** Increase `--k` or leave only the classes you need in the training file.
- **`--init seeded needs a training set` / `a training set is used only together with --init seeded`.** `--init ssl` and `--training` / `--training-fraction` always come together.
- **Compiler errors about `std::expected` or `std::format`.** The compiler or its standard library is too old; use GCC 13+ or Clang 16+ (with libc++ 17+ or libstdc++ 13+). On Windows install MSYS2 `mingw-w64-x86_64-gcc`.
- **Visualizer: `ModuleNotFoundError`.** `pip install numpy pandas matplotlib`.
- **Visualizer: no window opens (server, SSH, CI).** Use `--save picture.png`.


