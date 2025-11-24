--tp=1 --n-layers=1
Took 59.04 seconds to run

--tp=2 --n-layers=1
Took 62.85 seconds to run

--tp=4 --n-layers=1
Took 55.35 seconds to run

--tp=1 --n-layers=2
Took 66.20 seconds to run

--tp=1 --n-layers=4
Took 85.50 seconds to run

--tp=1 --n-layers=8
Took 120.52 seconds to run

--tp=2 --n-layers=8
Took 120.47 seconds to run

--tp=4 --n-layers=8
Took 110.06 seconds to run

- Effect of model size (--n-layers)
Increasing the number of layers increases the model’s compute cost.
Runtime grew from 59s (1 layer) to 120s (8 layers).

- Effect of Tensor Parallel Degree
For small models (1 layer), TP introduced more communication than compute speedup, so:
TP=1 was stable and efficient
TP=2 was slightly slower
TP=4 gave a modest improvement but gains were small

For the larger 8-layer model:
TP=1 and TP=2 were similar
TP=4 was significantly faster (110s vs 120s)

- Conclusion
Tensor parallelism becomes beneficial only when the model is large enough that compute dominates communication.
For small models, communication overhead makes TP unnecessary or slower.
For larger models (8 layers), TP=4 provided the best performance.
