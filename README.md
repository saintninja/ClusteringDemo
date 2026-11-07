# clustering

Учебный проект по кластеризации на C++23: K-means (Ллойд), EM-GMM, DBSCAN, Affinity Propagation и оценка качества (ARI, AMI, силуэт, индекс Дэвиса-Болдина). K-means и GMM поддерживают полу-контролируемый режим (SSL) с обучающей выборкой.

## Структура

    include/clustering/   публичные заголовки
      core/               Matrix, Cholesky, линейная алгебра, Result<T>
      distance/           DistanceMetric: Euclidean, Manhattan, Mahalanobis + фабрика
      initialization/     CentroidInitializer: Forgy, k-means++, Seeded (SSL), TrainingLabels + фабрика
      algorithms/         Clusterer: KMeans, GaussianMixture, Dbscan, AffinityPropagation
      evaluation/         ARI, AMI, Silhouette, Davies-Bouldin, QualityEvaluator
      io/                 CSV, загрузка выборки, запись результата
      preprocessing/      стандартизация признаков
      app/                разбор командной строки, фабрика алгоритмов, конвейер, отчёт
    src/                  реализации (зеркалят include) и main.cpp
    tests/                модульные тесты (CTest)
    data/                 boxes.csv, blob.csv, penguins_sex.csv
    scripts/visualize.py  рисунок результата

## Сборка

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build
    ctest --test-dir build --output-on-failure

Компилятор: GCC 13+ или Clang 16+ (libstdc++ 13+ либо libc++ 17+). Стандарт (по умолчанию 23) выбирается так: `-DCLUSTERING_CXX_STANDARD=26`. Для `-Werror`: `-DCLUSTERING_WARNINGS_AS_ERRORS=ON`.

## Примеры

    clustering_cli --input data/blob.csv --output out.csv --columns x,y --labels color \
        --algorithm kmeans --k 4 --init kmeans++ --metric euclidean
    clustering_cli --input unlabelled.csv --output out.csv --columns x,y \
        --algorithm kmeans --k 4 --init ssl --training train.csv --training-labels color
    clustering_cli --input data/boxes.csv --output out.csv --columns x,y --labels color \
        --algorithm kmeans --k 30 --init ssl --training-fraction 0.02
    clustering_cli --input data/boxes.csv --output out.csv --columns x,y --labels color \
        --algorithm dbscan --eps 6 --min-pts 5
    clustering_cli --input data/boxes.csv --output out.csv --columns x,y --labels color \
        --algorithm affinity --sample 2000
    clustering_cli --input data/penguins_sex.csv --output out.csv --standardize \
        --columns culmen_length_mm,culmen_depth_mm,flipper_length_mm,body_mass_g \
        --algorithm kmeans --k 3 --init kmeans++ --metric mahalanobis

    python scripts/visualize.py out.csv -c x y --no-centers

## SSL (k-means с обучающей выборкой)

`--init ssl` (или `seeded`) требует обучающую выборку: файл `--training` с колонкой классов `--training-labels` либо часть размеченного входа `--training-fraction`. Начальные центры равны средним по классам обучающей выборки; класс с номером `i` в порядке сортировки меток становится кластером `i`. Метки обучающих объектов не меняются, но их координаты участвуют в пересчёте центров. Обучающая выборка добавляется в выходной файл (первыми строками, колонка `origin` = `training`), поэтому у границ кластеры могут накладываться. Метрики качества печатаются для всех объектов и отдельно только для кластеризованных.
