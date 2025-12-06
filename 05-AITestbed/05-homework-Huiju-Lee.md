### Cerebras Homework : Run the Llama-7B example for different batch sizes and compare the performance.

batch_size: 128
<details>
 ```
 2025-12-06 01:53:15,087 INFO:   ===========================================================================
2025-12-06 01:53:15,087 INFO:   Trainer Fit Summary
2025-12-06 01:53:15,087 INFO:   ---------------------------------------------------------------------------
2025-12-06 01:53:15,087 INFO:   Trainer will run 1 train loop.
2025-12-06 01:53:15,087 INFO:   
2025-12-06 01:53:15,087 INFO:   Train steps per train loop:
2025-12-06 01:53:15,087 INFO:   * 1 loop of 200 steps
2025-12-06 01:53:15,087 INFO:   for a total of 200 train steps.
2025-12-06 01:53:15,087 INFO:   
2025-12-06 01:53:15,087 INFO:   Checkpoints will be taken every 200 steps, for a total of 1 checkpoint.
2025-12-06 01:53:15,087 INFO:   
2025-12-06 01:53:15,087 INFO:   Progress will be logged every 50 steps.
2025-12-06 01:53:15,087 INFO:   ===========================================================================
2025-12-06 01:53:15,087 INFO:   ---------------------------------------------------------------------------
2025-12-06 01:53:15,087 INFO:   Starting train loop 1 of 1, from global step 1 to 200 (200 steps)
2025-12-06 01:53:15,087 INFO:   ---------------------------------------------------------------------------
2025-12-06 01:53:15,657 INFO:   Saving checkpoint at step 0
2025-12-06 01:53:24,088 INFO:   Saved checkpoint model_dir_llama2_7b/checkpoint_0.mdl
2025-12-06 01:53:32,832 INFO:   Compiling the model. This may take a few minutes.
2025-12-06 01:53:32,848 INFO:   Appliance client semantic version: 1.1.0, cluster server semantic version: 1.1.2, job operator semantic version: 1.1.2
2025-12-06 01:53:33,830 INFO:   Initiating a new image build job against the cluster server.
2025-12-06 01:53:33,835 INFO:   User sidecar image build is disabled from server. Falling back to venv mounting.
2025-12-06 01:53:33,858 INFO:   Appliance client semantic version: 1.1.0, cluster server semantic version: 1.1.2, job operator semantic version: 1.1.2
2025-12-06 01:53:33,860 INFO:   Initiating a new compile wsjob against the cluster server.
2025-12-06 01:53:33,880 INFO:   Job id: wsjob-5nga78gwbdvznmjwqbfbhn, workflow id: wflow-idsoplncoewzuziwpvsgy4, namespace: job-operator, remote log path: /n1/wsjob/workdir/job-operator/wsjob-5nga78gwbdvznmjwqbfbhn
2025-12-06 01:53:53,881 INFO:   Poll ingress status: Waiting for all Coordinator pods to be running, current running: 0/1.
2025-12-06 01:53:53,883 INFO:   Recording the timestamp when jobs is scheduled.
2025-12-06 01:53:53,900 WARNING:   Event 2025-12-06 01:53:34 +0000 UTC reason=InconsistentVersion wsjob=wsjob-5nga78gwbdvznmjwqbfbhn message='Warning: client semantic version 1.1.0 is inconsistent with cluster server semantic version 1.1.2, there's a risk job could fail due to inconsistent setup.'
2025-12-06 01:54:03,900 INFO:   Poll ingress status: Job ingress ready, dashboard: https://grafana.anl0.cerebras.internal/d/WebHNShVz/wsjob-dashboard?orgId=1&var-wsjob=wsjob-5nga78gwbdvznmjwqbfbhn&from=1764985422000&to=now
2025-12-06 01:54:03,911 INFO:   Poll ingress success: Job ingress ready, dashboard: https://grafana.anl0.cerebras.internal/d/WebHNShVz/wsjob-dashboard?orgId=1&var-wsjob=wsjob-5nga78gwbdvznmjwqbfbhn&from=1764985422000&to=now
2025-12-06 01:54:08,323 INFO:   Pre-optimization transforms...
2025-12-06 01:54:22,858 INFO:   Optimizing layouts and memory usage...
2025-12-06 01:54:23,006 INFO:   Gradient accumulation enabled
2025-12-06 01:54:23,017 INFO:   Gradient accumulation trying micro batch size 4...
2025-12-06 01:54:40,617 INFO:   Exploring floorplans
2025-12-06 01:54:59,518 INFO:   Exploring data layouts
2025-12-06 01:59:23,785 INFO:   Optimizing memory usage
2025-12-06 01:59:40,680 INFO:   Gradient accumulation trying micro batch size 32...
2025-12-06 01:59:58,552 INFO:   Exploring floorplans
2025-12-06 02:00:27,210 INFO:   Exploring data layouts
2025-12-06 02:02:50,750 INFO:   Optimizing memory usage
2025-12-06 02:03:36,640 INFO:   Gradient accumulation trying micro batch size 8...
2025-12-06 02:03:54,587 INFO:   Exploring floorplans
2025-12-06 02:04:11,077 INFO:   Exploring data layouts
2025-12-06 02:04:55,853 INFO:   Optimizing memory usage
2025-12-06 02:05:15,692 INFO:   Gradient accumulation trying micro batch size 64...
2025-12-06 02:05:34,071 INFO:   Exploring floorplans
2025-12-06 02:06:09,269 INFO:   Exploring data layouts
 2025-12-06 02:09:02,447 INFO:   Optimizing memory usage
2025-12-06 02:10:06,486 INFO:   Gradient accumulation trying micro batch size 16...
2025-12-06 02:10:24,286 INFO:   Exploring floorplans
2025-12-06 02:10:45,428 INFO:   Exploring data layouts
2025-12-06 02:11:33,308 INFO:   Optimizing memory usage
2025-12-06 02:11:53,330 INFO:   Gradient accumulation trying full batch size 128...
2025-12-06 02:12:29,212 INFO:   Exploring floorplans
2025-12-06 02:14:30,899 INFO:   Exploring data layouts
2025-12-06 02:39:46,978 INFO:   Optimizing memory usage
2025-12-06 02:41:56,993 INFO:   Gradient accumulation showed a benefit
2025-12-06 02:41:58,662 INFO:   Post-layout optimizations for <pbox=0, vbox=0, microbatch=32, lanes=1>...
2025-12-06 02:41:58,681 INFO:   Post-layout optimizations for <pbox=0, vbox=0, microbatch=32, lanes=3>...
2025-12-06 02:41:58,682 INFO:   Post-layout optimizations for <pbox=0, vbox=0, microbatch=32, lanes=2>...
2025-12-06 02:41:58,687 INFO:   Post-layout optimizations for <pbox=0, vbox=0, microbatch=64, lanes=5>...
2025-12-06 02:41:58,694 INFO:   Post-layout optimizations for <pbox=0, vbox=0, microbatch=64, lanes=1>...
2025-12-06 02:41:58,699 INFO:   Post-layout optimizations for <pbox=0, vbox=0, microbatch=64, lanes=3>...
2025-12-06 02:41:58,702 INFO:   Post-layout optimizations for <pbox=0, vbox=0, microbatch=64, lanes=4>...
2025-12-06 02:41:58,723 INFO:   Post-layout optimizations for <pbox=0, vbox=0, microbatch=64, lanes=2>...
2025-12-06 02:41:59,440 INFO:   Post-layout optimizations for <pbox=0, vbox=0, batch=128, lanes=4>...
2025-12-06 02:41:59,446 INFO:   Post-layout optimizations for <pbox=0, vbox=0, batch=128, lanes=3>...
2025-12-06 02:41:59,446 INFO:   Post-layout optimizations for <pbox=0, vbox=0, batch=128, lanes=2>...
2025-12-06 02:41:59,449 INFO:   Post-layout optimizations for <pbox=0, vbox=0, batch=128, lanes=6>...
2025-12-06 02:41:59,533 INFO:   Post-layout optimizations for <pbox=0, vbox=0, batch=128, lanes=5>...
2025-12-06 02:42:06,756 INFO:   Allocating buffers for <pbox=0, vbox=0, microbatch=32, lanes=1>...
2025-12-06 02:42:07,368 INFO:   Allocating buffers for <pbox=0, vbox=0, microbatch=32, lanes=2>...
2025-12-06 02:42:08,501 INFO:   Allocating buffers for <pbox=0, vbox=0, microbatch=32, lanes=3>...
2025-12-06 02:42:08,555 INFO:   Allocating buffers for <pbox=0, vbox=0, microbatch=64, lanes=5>...
2025-12-06 02:42:08,561 INFO:   Allocating buffers for <pbox=0, vbox=0, microbatch=64, lanes=3>...
2025-12-06 02:42:09,024 INFO:   Allocating buffers for <pbox=0, vbox=0, microbatch=64, lanes=4>...
2025-12-06 02:42:09,352 INFO:   Allocating buffers for <pbox=0, vbox=0, microbatch=64, lanes=2>...
2025-12-06 02:42:09,355 INFO:   Allocating buffers for <pbox=0, vbox=0, microbatch=64, lanes=1>...
2025-12-06 02:42:13,170 INFO:   Code generation for <pbox=0, vbox=0, microbatch=32, lanes=1>...
2025-12-06 02:42:13,174 INFO:   Allocating buffers for <pbox=0, vbox=0, batch=128, lanes=4>...
2025-12-06 02:42:13,299 INFO:   Code generation for <pbox=0, vbox=0, microbatch=32, lanes=2>...
2025-12-06 02:42:14,050 INFO:   Code generation for <pbox=0, vbox=0, microbatch=32, lanes=3>...
2025-12-06 02:42:15,060 INFO:   Code generation for <pbox=0, vbox=0, microbatch=64, lanes=3>...
2025-12-06 02:42:15,413 INFO:   Code generation for <pbox=0, vbox=0, microbatch=64, lanes=5>...
2025-12-06 02:42:16,145 INFO:   Code generation for <pbox=0, vbox=0, microbatch=64, lanes=4>...
2025-12-06 02:42:16,757 INFO:   Code generation for <pbox=0, vbox=0, microbatch=64, lanes=2>...
2025-12-06 02:42:16,817 INFO:   Allocating buffers for <pbox=0, vbox=0, batch=128, lanes=2>...
2025-12-06 02:42:16,897 INFO:   Code generation for <pbox=0, vbox=0, microbatch=64, lanes=1>...
2025-12-06 02:42:20,315 INFO:   Allocating buffers for <pbox=0, vbox=0, batch=128, lanes=3>...
2025-12-06 02:42:21,013 INFO:   Allocating buffers for <pbox=0, vbox=0, batch=128, lanes=6>...
2025-12-06 02:42:21,914 INFO:   Allocating buffers for <pbox=0, vbox=0, batch=128, lanes=5>...
2025-12-06 02:42:35,500 INFO:   Code generation for <pbox=0, vbox=0, batch=128, lanes=4>...
2025-12-06 02:42:41,857 INFO:   Code generation for <pbox=0, vbox=0, batch=128, lanes=2>...
2025-12-06 02:42:45,699 INFO:   Code generation for <pbox=0, vbox=0, batch=128, lanes=3>...
2025-12-06 02:42:47,535 INFO:   Code generation for <pbox=0, vbox=0, batch=128, lanes=6>...
2025-12-06 02:42:49,080 INFO:   Code generation for <pbox=0, vbox=0, batch=128, lanes=5>...
2025-12-06 02:45:42,352 INFO:   Compiling at original per-box batch size 128
2025-12-06 02:45:59,387 INFO:   Compile estimated performance: 34.1 samples/sec (34.1 samples/sec/system). Estimate may vary by 10-25% from actual runtime performance.
2025-12-06 02:46:21,839 INFO:   Compiling image...
2025-12-06 02:46:24,304 INFO:   Compiling kernels
2025-12-06 02:48:20,369 INFO:   Compiling final image
2025-12-06 02:50:58,956 INFO:   Compile artifacts successfully written to remote compile directory. Compile hash is: cs_2325291116317321205
2025-12-06 02:51:04,236 INFO:   Compile was successful!
2025-12-06 02:51:04,237 INFO:   Programming Cerebras Wafer Scale Cluster for execution. This may take a few minutes.
2025-12-06 02:51:06,263 INFO:   Appliance client semantic version: 1.1.0, cluster server semantic version: 1.1.2, job operator semantic version: 1.1.2
2025-12-06 02:51:06,267 INFO:   Initiating a new execute wsjob against the cluster server.
2025-12-06 02:51:06,296 INFO:   Job id: wsjob-3zs8qawg5pdbkseukjedlq, workflow id: wflow-idsoplncoewzuziwpvsgy4, namespace: job-operator, remote log path: /n1/wsjob/workdir/job-operator/wsjob-3zs8qawg5pdbkseukjedlq
2025-12-06 02:51:26,296 INFO:   Poll ingress status: Waiting for all Chief pods to be running, current running: 0/1.
2025-12-06 02:51:26,299 INFO:   Recording the timestamp when jobs is scheduled.
2025-12-06 02:51:26,309 WARNING:   Event 2025-12-06 02:51:07 +0000 UTC reason=InconsistentVersion wsjob=wsjob-3zs8qawg5pdbkseukjedlq message='Warning: client semantic version 1.1.0 is inconsistent with cluster server semantic version 1.1.2, there's a risk job could fail due to inconsistent setup.'
2025-12-06 02:51:36,310 INFO:   Poll ingress status: Waiting for all Activation pods to be running, current running: 0/20.
2025-12-06 02:52:26,317 INFO:   Poll ingress status: Waiting for all Chief pods to be running, current running: 0/1.
2025-12-06 02:52:36,324 INFO:   Poll ingress status: Waiting for all Activation pods to be running, current running: 0/20.
2025-12-06 02:52:46,330 INFO:   Poll ingress status: Waiting for all Chief pods to be running, current running: 0/1.
2025-12-06 02:52:56,337 INFO:   Poll ingress status: Waiting for all Activation pods to be running, current running: 0/20.
2025-12-06 02:53:16,344 INFO:   Poll ingress status: Waiting for all Chief pods to be running, current running: 0/1.
2025-12-06 02:53:26,352 INFO:   Poll ingress status: Waiting for all Activation pods to be running, current running: 0/20.
2025-12-06 02:53:46,359 INFO:   Poll ingress status: Waiting for all Activation pods to be running, current running: 19/20.
2025-12-06 02:53:56,367 INFO:   Poll ingress status: Job ingress ready, dashboard: https://grafana.anl0.cerebras.internal/d/WebHNShVz/wsjob-dashboard?orgId=1&var-wsjob=wsjob-3zs8qawg5pdbkseukjedlq&from=1764988887000&to=now
2025-12-06 02:53:56,374 INFO:   Poll ingress success: Job ingress ready, dashboard: https://grafana.anl0.cerebras.internal/d/WebHNShVz/wsjob-dashboard?orgId=1&var-wsjob=wsjob-3zs8qawg5pdbkseukjedlq&from=1764988887000&to=now
2025-12-06 02:53:56,482 INFO:   Preparing to execute using 1 CSX
2025-12-06 02:54:37,793 INFO:   About to send initial weights
2025-12-06 02:54:58,221 INFO:   Finished sending initial weights
2025-12-06 02:54:58,222 INFO:   Finalizing appliance staging for the run
2025-12-06 02:54:58,235 INFO:   Waiting for device programming to complete
2025-12-06 02:58:54,173 INFO:   Device programming is complete
2025-12-06 02:58:55,756 INFO:   Using network type: ROCE
2025-12-06 02:58:55,757 INFO:   Waiting for input workers to prime the data pipeline and begin streaming ...
2025-12-06 02:58:55,769 INFO:   Input workers have begun streaming input data
2025-12-06 02:58:56,972 INFO:   Appliance staging is complete
2025-12-06 02:58:56,972 INFO:   Beginning appliance run
2025-12-06 03:02:15,929 INFO:   | Train Device=CSX, Step=50, Loss=8.10329, Rate=32.52 samples/sec, GlobalRate=32.17 samples/sec, LoopTimeRemaining=0:10:03, TimeRemaining=0:10:03
2025-12-06 03:05:32,591 INFO:   | Train Device=CSX, Step=100, Loss=7.38091, Rate=32.76 samples/sec, GlobalRate=32.36 samples/sec, LoopTimeRemaining=0:06:46, TimeRemaining=0:06:46
2025-12-06 03:08:49,502 INFO:   | Train Device=CSX, Step=150, Loss=6.86648, Rate=32.64 samples/sec, GlobalRate=32.41 samples/sec, LoopTimeRemaining=0:03:29, TimeRemaining=0:03:29
2025-12-06 03:12:06,140 INFO:   | Train Device=CSX, Step=200, Loss=6.44260, Rate=33.24 samples/sec, GlobalRate=32.44 samples/sec, LoopTimeRemaining=0:00:12, TimeRemaining=0:00:12
2025-12-06 03:12:06,147 INFO:   Saving checkpoint at step 200
2025-12-06 03:22:50,236 INFO:   Saved checkpoint model_dir_llama2_7b/checkpoint_200.mdl
2025-12-06 03:23:04,336 INFO:   Training completed successfully!
2025-12-06 03:23:04,343 INFO:   Processed 25600 training sample(s) in 5389.258144948 seconds.
 ```
</details>
<details>
 ```
 2025-12-06 01:53:15,087 INFO:   ===========================================================================
2025-12-06 01:53:15,087 INFO:   Trainer Fit Summary
2025-12-06 01:53:15,087 INFO:   ---------------------------------------------------------------------------
2025-12-06 01:53:15,087 INFO:   Trainer will run 1 train loop.
2025-12-06 01:53:15,087 INFO:   
2025-12-06 01:53:15,087 INFO:   Train steps per train loop:
2025-12-06 01:53:15,087 INFO:   * 1 loop of 200 steps
2025-12-06 01:53:15,087 INFO:   for a total of 200 train steps.
2025-12-06 01:53:15,087 INFO:   
2025-12-06 01:53:15,087 INFO:   Checkpoints will be taken every 200 steps, for a total of 1 checkpoint.
2025-12-06 01:53:15,087 INFO:   
2025-12-06 01:53:15,087 INFO:   Progress will be logged every 50 steps.
2025-12-06 01:53:15,087 INFO:   ===========================================================================
2025-12-06 01:53:15,087 INFO:   ---------------------------------------------------------------------------
2025-12-06 01:53:15,087 INFO:   Starting train loop 1 of 1, from global step 1 to 200 (200 steps)
2025-12-06 01:53:15,087 INFO:   ---------------------------------------------------------------------------
2025-12-06 01:53:15,657 INFO:   Saving checkpoint at step 0
2025-12-06 01:53:24,088 INFO:   Saved checkpoint model_dir_llama2_7b/checkpoint_0.mdl
2025-12-06 01:53:32,832 INFO:   Compiling the model. This may take a few minutes.
2025-12-06 01:53:32,848 INFO:   Appliance client semantic version: 1.1.0, cluster server semantic version: 1.1.2, job operator semantic version: 1.1.2
2025-12-06 01:53:33,830 INFO:   Initiating a new image build job against the cluster server.
2025-12-06 01:53:33,835 INFO:   User sidecar image build is disabled from server. Falling back to venv mounting.
2025-12-06 01:53:33,858 INFO:   Appliance client semantic version: 1.1.0, cluster server semantic version: 1.1.2, job operator semantic version: 1.1.2
2025-12-06 01:53:33,860 INFO:   Initiating a new compile wsjob against the cluster server.
2025-12-06 01:53:33,880 INFO:   Job id: wsjob-5nga78gwbdvznmjwqbfbhn, workflow id: wflow-idsoplncoewzuziwpvsgy4, namespace: job-operator, remote log path: /n1/wsjob/workdir/job-operator/wsjob-5nga78gwbdvznmjwqbfbhn
2025-12-06 01:53:53,881 INFO:   Poll ingress status: Waiting for all Coordinator pods to be running, current running: 0/1.
2025-12-06 01:53:53,883 INFO:   Recording the timestamp when jobs is scheduled.
2025-12-06 01:53:53,900 WARNING:   Event 2025-12-06 01:53:34 +0000 UTC reason=InconsistentVersion wsjob=wsjob-5nga78gwbdvznmjwqbfbhn message='Warning: client semantic version 1.1.0 is inconsistent with cluster server semantic version 1.1.2, there's a risk job could fail due to inconsistent setup.'
2025-12-06 01:54:03,900 INFO:   Poll ingress status: Job ingress ready, dashboard: https://grafana.anl0.cerebras.internal/d/WebHNShVz/wsjob-dashboard?orgId=1&var-wsjob=wsjob-5nga78gwbdvznmjwqbfbhn&from=1764985422000&to=now
2025-12-06 01:54:03,911 INFO:   Poll ingress success: Job ingress ready, dashboard: https://grafana.anl0.cerebras.internal/d/WebHNShVz/wsjob-dashboard?orgId=1&var-wsjob=wsjob-5nga78gwbdvznmjwqbfbhn&from=1764985422000&to=now
2025-12-06 01:54:08,323 INFO:   Pre-optimization transforms...
2025-12-06 01:54:22,858 INFO:   Optimizing layouts and memory usage...
2025-12-06 01:54:23,006 INFO:   Gradient accumulation enabled
2025-12-06 01:54:23,017 INFO:   Gradient accumulation trying micro batch size 4...
2025-12-06 01:54:40,617 INFO:   Exploring floorplans
2025-12-06 01:54:59,518 INFO:   Exploring data layouts
2025-12-06 01:59:23,785 INFO:   Optimizing memory usage
2025-12-06 01:59:40,680 INFO:   Gradient accumulation trying micro batch size 32...
2025-12-06 01:59:58,552 INFO:   Exploring floorplans
2025-12-06 02:00:27,210 INFO:   Exploring data layouts
2025-12-06 02:02:50,750 INFO:   Optimizing memory usage
2025-12-06 02:03:36,640 INFO:   Gradient accumulation trying micro batch size 8...
2025-12-06 02:03:54,587 INFO:   Exploring floorplans
2025-12-06 02:04:11,077 INFO:   Exploring data layouts
2025-12-06 02:04:55,853 INFO:   Optimizing memory usage
2025-12-06 02:05:15,692 INFO:   Gradient accumulation trying micro batch size 64...
2025-12-06 02:05:34,071 INFO:   Exploring floorplans
2025-12-06 02:06:09,269 INFO:   Exploring data layouts
 2025-12-06 02:09:02,447 INFO:   Optimizing memory usage
2025-12-06 02:10:06,486 INFO:   Gradient accumulation trying micro batch size 16...
2025-12-06 02:10:24,286 INFO:   Exploring floorplans
2025-12-06 02:10:45,428 INFO:   Exploring data layouts
2025-12-06 02:11:33,308 INFO:   Optimizing memory usage
2025-12-06 02:11:53,330 INFO:   Gradient accumulation trying full batch size 128...
2025-12-06 02:12:29,212 INFO:   Exploring floorplans
2025-12-06 02:14:30,899 INFO:   Exploring data layouts
2025-12-06 02:39:46,978 INFO:   Optimizing memory usage
2025-12-06 02:41:56,993 INFO:   Gradient accumulation showed a benefit
2025-12-06 02:41:58,662 INFO:   Post-layout optimizations for <pbox=0, vbox=0, microbatch=32, lanes=1>...
2025-12-06 02:41:58,681 INFO:   Post-layout optimizations for <pbox=0, vbox=0, microbatch=32, lanes=3>...
2025-12-06 02:41:58,682 INFO:   Post-layout optimizations for <pbox=0, vbox=0, microbatch=32, lanes=2>...
2025-12-06 02:41:58,687 INFO:   Post-layout optimizations for <pbox=0, vbox=0, microbatch=64, lanes=5>...
2025-12-06 02:41:58,694 INFO:   Post-layout optimizations for <pbox=0, vbox=0, microbatch=64, lanes=1>...
2025-12-06 02:41:58,699 INFO:   Post-layout optimizations for <pbox=0, vbox=0, microbatch=64, lanes=3>...
2025-12-06 02:41:58,702 INFO:   Post-layout optimizations for <pbox=0, vbox=0, microbatch=64, lanes=4>...
2025-12-06 02:41:58,723 INFO:   Post-layout optimizations for <pbox=0, vbox=0, microbatch=64, lanes=2>...
2025-12-06 02:41:59,440 INFO:   Post-layout optimizations for <pbox=0, vbox=0, batch=128, lanes=4>...
2025-12-06 02:41:59,446 INFO:   Post-layout optimizations for <pbox=0, vbox=0, batch=128, lanes=3>...
2025-12-06 02:41:59,446 INFO:   Post-layout optimizations for <pbox=0, vbox=0, batch=128, lanes=2>...
2025-12-06 02:41:59,449 INFO:   Post-layout optimizations for <pbox=0, vbox=0, batch=128, lanes=6>...
2025-12-06 02:41:59,533 INFO:   Post-layout optimizations for <pbox=0, vbox=0, batch=128, lanes=5>...
2025-12-06 02:42:06,756 INFO:   Allocating buffers for <pbox=0, vbox=0, microbatch=32, lanes=1>...
2025-12-06 02:42:07,368 INFO:   Allocating buffers for <pbox=0, vbox=0, microbatch=32, lanes=2>...
2025-12-06 02:42:08,501 INFO:   Allocating buffers for <pbox=0, vbox=0, microbatch=32, lanes=3>...
2025-12-06 02:42:08,555 INFO:   Allocating buffers for <pbox=0, vbox=0, microbatch=64, lanes=5>...
2025-12-06 02:42:08,561 INFO:   Allocating buffers for <pbox=0, vbox=0, microbatch=64, lanes=3>...
2025-12-06 02:42:09,024 INFO:   Allocating buffers for <pbox=0, vbox=0, microbatch=64, lanes=4>...
2025-12-06 02:42:09,352 INFO:   Allocating buffers for <pbox=0, vbox=0, microbatch=64, lanes=2>...
2025-12-06 02:42:09,355 INFO:   Allocating buffers for <pbox=0, vbox=0, microbatch=64, lanes=1>...
2025-12-06 02:42:13,170 INFO:   Code generation for <pbox=0, vbox=0, microbatch=32, lanes=1>...
2025-12-06 02:42:13,174 INFO:   Allocating buffers for <pbox=0, vbox=0, batch=128, lanes=4>...
2025-12-06 02:42:13,299 INFO:   Code generation for <pbox=0, vbox=0, microbatch=32, lanes=2>...
2025-12-06 02:42:14,050 INFO:   Code generation for <pbox=0, vbox=0, microbatch=32, lanes=3>...
2025-12-06 02:42:15,060 INFO:   Code generation for <pbox=0, vbox=0, microbatch=64, lanes=3>...
2025-12-06 02:42:15,413 INFO:   Code generation for <pbox=0, vbox=0, microbatch=64, lanes=5>...
2025-12-06 02:42:16,145 INFO:   Code generation for <pbox=0, vbox=0, microbatch=64, lanes=4>...
2025-12-06 02:42:16,757 INFO:   Code generation for <pbox=0, vbox=0, microbatch=64, lanes=2>...
2025-12-06 02:42:16,817 INFO:   Allocating buffers for <pbox=0, vbox=0, batch=128, lanes=2>...
2025-12-06 02:42:16,897 INFO:   Code generation for <pbox=0, vbox=0, microbatch=64, lanes=1>...
2025-12-06 02:42:20,315 INFO:   Allocating buffers for <pbox=0, vbox=0, batch=128, lanes=3>...
2025-12-06 02:42:21,013 INFO:   Allocating buffers for <pbox=0, vbox=0, batch=128, lanes=6>...
2025-12-06 02:42:21,914 INFO:   Allocating buffers for <pbox=0, vbox=0, batch=128, lanes=5>...
2025-12-06 02:42:35,500 INFO:   Code generation for <pbox=0, vbox=0, batch=128, lanes=4>...
2025-12-06 02:42:41,857 INFO:   Code generation for <pbox=0, vbox=0, batch=128, lanes=2>...
2025-12-06 02:42:45,699 INFO:   Code generation for <pbox=0, vbox=0, batch=128, lanes=3>...
2025-12-06 02:42:47,535 INFO:   Code generation for <pbox=0, vbox=0, batch=128, lanes=6>...
2025-12-06 02:42:49,080 INFO:   Code generation for <pbox=0, vbox=0, batch=128, lanes=5>...
2025-12-06 02:45:42,352 INFO:   Compiling at original per-box batch size 128
2025-12-06 02:45:59,387 INFO:   Compile estimated performance: 34.1 samples/sec (34.1 samples/sec/system). Estimate may vary by 10-25% from actual runtime performance.
2025-12-06 02:46:21,839 INFO:   Compiling image...
2025-12-06 02:46:24,304 INFO:   Compiling kernels
2025-12-06 02:48:20,369 INFO:   Compiling final image
2025-12-06 02:50:58,956 INFO:   Compile artifacts successfully written to remote compile directory. Compile hash is: cs_2325291116317321205
2025-12-06 02:51:04,236 INFO:   Compile was successful!
2025-12-06 02:51:04,237 INFO:   Programming Cerebras Wafer Scale Cluster for execution. This may take a few minutes.
2025-12-06 02:51:06,263 INFO:   Appliance client semantic version: 1.1.0, cluster server semantic version: 1.1.2, job operator semantic version: 1.1.2
2025-12-06 02:51:06,267 INFO:   Initiating a new execute wsjob against the cluster server.
2025-12-06 02:51:06,296 INFO:   Job id: wsjob-3zs8qawg5pdbkseukjedlq, workflow id: wflow-idsoplncoewzuziwpvsgy4, namespace: job-operator, remote log path: /n1/wsjob/workdir/job-operator/wsjob-3zs8qawg5pdbkseukjedlq
2025-12-06 02:51:26,296 INFO:   Poll ingress status: Waiting for all Chief pods to be running, current running: 0/1.
2025-12-06 02:51:26,299 INFO:   Recording the timestamp when jobs is scheduled.
2025-12-06 02:51:26,309 WARNING:   Event 2025-12-06 02:51:07 +0000 UTC reason=InconsistentVersion wsjob=wsjob-3zs8qawg5pdbkseukjedlq message='Warning: client semantic version 1.1.0 is inconsistent with cluster server semantic version 1.1.2, there's a risk job could fail due to inconsistent setup.'
2025-12-06 02:51:36,310 INFO:   Poll ingress status: Waiting for all Activation pods to be running, current running: 0/20.
2025-12-06 02:52:26,317 INFO:   Poll ingress status: Waiting for all Chief pods to be running, current running: 0/1.
2025-12-06 02:52:36,324 INFO:   Poll ingress status: Waiting for all Activation pods to be running, current running: 0/20.
2025-12-06 02:52:46,330 INFO:   Poll ingress status: Waiting for all Chief pods to be running, current running: 0/1.
2025-12-06 02:52:56,337 INFO:   Poll ingress status: Waiting for all Activation pods to be running, current running: 0/20.
2025-12-06 02:53:16,344 INFO:   Poll ingress status: Waiting for all Chief pods to be running, current running: 0/1.
2025-12-06 02:53:26,352 INFO:   Poll ingress status: Waiting for all Activation pods to be running, current running: 0/20.
2025-12-06 02:53:46,359 INFO:   Poll ingress status: Waiting for all Activation pods to be running, current running: 19/20.
2025-12-06 02:53:56,367 INFO:   Poll ingress status: Job ingress ready, dashboard: https://grafana.anl0.cerebras.internal/d/WebHNShVz/wsjob-dashboard?orgId=1&var-wsjob=wsjob-3zs8qawg5pdbkseukjedlq&from=1764988887000&to=now
2025-12-06 02:53:56,374 INFO:   Poll ingress success: Job ingress ready, dashboard: https://grafana.anl0.cerebras.internal/d/WebHNShVz/wsjob-dashboard?orgId=1&var-wsjob=wsjob-3zs8qawg5pdbkseukjedlq&from=1764988887000&to=now
2025-12-06 02:53:56,482 INFO:   Preparing to execute using 1 CSX
2025-12-06 02:54:37,793 INFO:   About to send initial weights
2025-12-06 02:54:58,221 INFO:   Finished sending initial weights
2025-12-06 02:54:58,222 INFO:   Finalizing appliance staging for the run
2025-12-06 02:54:58,235 INFO:   Waiting for device programming to complete
2025-12-06 02:58:54,173 INFO:   Device programming is complete
2025-12-06 02:58:55,756 INFO:   Using network type: ROCE
2025-12-06 02:58:55,757 INFO:   Waiting for input workers to prime the data pipeline and begin streaming ...
2025-12-06 02:58:55,769 INFO:   Input workers have begun streaming input data
2025-12-06 02:58:56,972 INFO:   Appliance staging is complete
2025-12-06 02:58:56,972 INFO:   Beginning appliance run
2025-12-06 03:02:15,929 INFO:   | Train Device=CSX, Step=50, Loss=8.10329, Rate=32.52 samples/sec, GlobalRate=32.17 samples/sec, LoopTimeRemaining=0:10:03, TimeRemaining=0:10:03
2025-12-06 03:05:32,591 INFO:   | Train Device=CSX, Step=100, Loss=7.38091, Rate=32.76 samples/sec, GlobalRate=32.36 samples/sec, LoopTimeRemaining=0:06:46, TimeRemaining=0:06:46
2025-12-06 03:08:49,502 INFO:   | Train Device=CSX, Step=150, Loss=6.86648, Rate=32.64 samples/sec, GlobalRate=32.41 samples/sec, LoopTimeRemaining=0:03:29, TimeRemaining=0:03:29
2025-12-06 03:12:06,140 INFO:   | Train Device=CSX, Step=200, Loss=6.44260, Rate=33.24 samples/sec, GlobalRate=32.44 samples/sec, LoopTimeRemaining=0:00:12, TimeRemaining=0:00:12
2025-12-06 03:12:06,147 INFO:   Saving checkpoint at step 200
2025-12-06 03:22:50,236 INFO:   Saved checkpoint model_dir_llama2_7b/checkpoint_200.mdl
2025-12-06 03:23:04,336 INFO:   Training completed successfully!
2025-12-06 03:23:04,343 INFO:   Processed 25600 training sample(s) in 5389.258144948 seconds.
```
<details> 

</details>

<details>
batch_size: 64
```
2025-12-06 03:51:15,329 INFO:   ===========================================================================
2025-12-06 03:51:15,329 INFO:   Trainer Fit Summary
2025-12-06 03:51:15,329 INFO:   ---------------------------------------------------------------------------
2025-12-06 03:51:15,329 INFO:   Trainer will run 1 train loop.
2025-12-06 03:51:15,329 INFO:   
2025-12-06 03:51:15,330 INFO:   Train steps per train loop:
2025-12-06 03:51:15,330 INFO:   * 1 loop of 200 steps
2025-12-06 03:51:15,330 INFO:   for a total of 200 train steps.
2025-12-06 03:51:15,330 INFO:   
2025-12-06 03:51:15,330 INFO:   Checkpoints will be taken every 200 steps, for a total of 1 checkpoint.
2025-12-06 03:51:15,330 INFO:   
2025-12-06 03:51:15,330 INFO:   Progress will be logged every 50 steps.
2025-12-06 03:51:15,330 INFO:   ===========================================================================
2025-12-06 03:51:15,330 INFO:   ---------------------------------------------------------------------------
2025-12-06 03:51:15,330 INFO:   Starting train loop 1 of 1, from global step 1 to 200 (200 steps)
2025-12-06 03:51:15,330 INFO:   ---------------------------------------------------------------------------
2025-12-06 03:51:15,903 INFO:   Saving checkpoint at step 0
2025-12-06 03:51:24,999 INFO:   Saved checkpoint model_dir_llama2_7b/checkpoint_0.mdl
2025-12-06 03:51:33,593 INFO:   Compiling the model. This may take a few minutes.
2025-12-06 03:51:33,610 INFO:   Appliance client semantic version: 1.1.0, cluster server semantic version: 1.1.2, job operator semantic version: 1.1.2
2025-12-06 03:51:34,605 INFO:   Initiating a new image build job against the cluster server.
2025-12-06 03:51:34,610 INFO:   User sidecar image build is disabled from server. Falling back to venv mounting.
2025-12-06 03:51:34,631 INFO:   Appliance client semantic version: 1.1.0, cluster server semantic version: 1.1.2, job operator semantic version: 1.1.2
2025-12-06 03:51:34,634 INFO:   Initiating a new compile wsjob against the cluster server.
2025-12-06 03:51:34,654 INFO:   Job id: wsjob-gtmmek9dnwvuq4dhkcmkgz, workflow id: wflow-b8v3hpwtqru9fzjkkwkxjf, namespace: job-operator, remote log path: /n1/wsjob/workdir/job-operator/wsjob-gtmmek9dnwvuq4dhkcmkgz
2025-12-06 03:51:54,654 INFO:   Poll ingress status: Waiting for all Coordinator pods to be running, current running: 0/1.
2025-12-06 03:51:54,657 INFO:   Recording the timestamp when jobs is scheduled.
2025-12-06 03:51:54,673 WARNING:   Event 2025-12-06 03:51:35 +0000 UTC reason=InconsistentVersion wsjob=wsjob-gtmmek9dnwvuq4dhkcmkgz message='Warning: client semantic version 1.1.0 is inconsistent with cluster server semantic version 1.1.2, there's a risk job could fail due to inconsistent setup.'
2025-12-06 03:52:04,673 INFO:   Poll ingress status: Waiting for job ingress readiness.
2025-12-06 03:52:44,681 INFO:   Poll ingress status: Job ingress ready, dashboard: https://grafana.anl0.cerebras.internal/d/WebHNShVz/wsjob-dashboard?orgId=1&var-wsjob=wsjob-gtmmek9dnwvuq4dhkcmkgz&from=1764992503000&to=now
2025-12-06 03:52:44,693 INFO:   Poll ingress success: Job ingress ready, dashboard: https://grafana.anl0.cerebras.internal/d/WebHNShVz/wsjob-dashboard?orgId=1&var-wsjob=wsjob-gtmmek9dnwvuq4dhkcmkgz&from=1764992503000&to=now
2025-12-06 03:52:45,470 INFO:   Found existing cached compile with hash: "cs_3420986671335802013"
2025-12-06 03:52:49,328 INFO:   Compile artifacts successfully written to remote compile directory. Compile hash is: cs_3420986671335802013
2025-12-06 03:52:54,678 INFO:   Compile was successful!
2025-12-06 03:52:54,679 INFO:   Programming Cerebras Wafer Scale Cluster for execution. This may take a few minutes.
2025-12-06 03:52:56,705 INFO:   Appliance client semantic version: 1.1.0, cluster server semantic version: 1.1.2, job operator semantic version: 1.1.2
2025-12-06 03:52:56,708 INFO:   Initiating a new execute wsjob against the cluster server.
2025-12-06 03:52:56,744 INFO:   Job id: wsjob-gu3uanrjnefdfonsstn2fb, workflow id: wflow-b8v3hpwtqru9fzjkkwkxjf, namespace: job-operator, remote log path: /n1/wsjob/workdir/job-operator/wsjob-gu3uanrjnefdfonsstn2fb
2025-12-06 03:53:16,745 INFO:   Poll ingress status: Waiting for all Worker pods to be running, current running: 0/1.
2025-12-06 03:53:16,747 INFO:   Recording the timestamp when jobs is scheduled.
2025-12-06 03:53:16,763 WARNING:   Event 2025-12-06 03:52:57 +0000 UTC reason=InconsistentVersion wsjob=wsjob-gu3uanrjnefdfonsstn2fb message='Warning: client semantic version 1.1.0 is inconsistent with cluster server semantic version 1.1.2, there's a risk job could fail due to inconsistent setup.'
2025-12-06 03:53:26,763 INFO:   Poll ingress status: Waiting for all Weight pods to be running, current running: 10/20.
2025-12-06 03:53:36,774 INFO:   Poll ingress status: Job ingress ready, dashboard: https://grafana.anl0.cerebras.internal/d/WebHNShVz/wsjob-dashboard?orgId=1&var-wsjob=wsjob-gu3uanrjnefdfonsstn2fb&from=1764992597000&to=now
2025-12-06 03:53:36,785 INFO:   Poll ingress success: Job ingress ready, dashboard: https://grafana.anl0.cerebras.internal/d/WebHNShVz/wsjob-dashboard?orgId=1&var-wsjob=wsjob-gu3uanrjnefdfonsstn2fb&from=1764992597000&to=now
2025-12-06 03:53:36,889 INFO:   Preparing to execute using 1 CSX
2025-12-06 03:54:22,641 INFO:   About to send initial weights
2025-12-06 03:54:41,655 INFO:   Finished sending initial weights
2025-12-06 03:54:41,656 INFO:   Finalizing appliance staging for the run
2025-12-06 03:54:41,669 INFO:   Waiting for device programming to complete
2025-12-06 03:58:22,193 INFO:   Device programming is complete
2025-12-06 03:58:23,289 INFO:   Using network type: ROCE
2025-12-06 03:58:23,289 INFO:   Waiting for input workers to prime the data pipeline and begin streaming ...
2025-12-06 03:58:23,303 INFO:   Input workers have begun streaming input data
2025-12-06 03:58:24,448 INFO:   Appliance staging is complete
2025-12-06 03:58:24,448 INFO:   Beginning appliance run
2025-12-06 04:00:07,397 INFO:   | Train Device=CSX, Step=50, Loss=8.11024, Rate=31.47 samples/sec, GlobalRate=31.10 samples/sec, LoopTimeRemaining=0:05:16, TimeRemaining=0:05:16
2025-12-06 04:01:48,950 INFO:   | Train Device=CSX, Step=100, Loss=7.71769, Rate=31.28 samples/sec, GlobalRate=31.30 samples/sec, LoopTimeRemaining=0:03:34, TimeRemaining=0:03:34
2025-12-06 04:03:30,759 INFO:   | Train Device=CSX, Step=150, Loss=7.06736, Rate=31.61 samples/sec, GlobalRate=31.34 samples/sec, LoopTimeRemaining=0:01:53, TimeRemaining=0:01:53
2025-12-06 04:05:12,830 INFO:   | Train Device=CSX, Step=200, Loss=6.57218, Rate=31.75 samples/sec, GlobalRate=31.35 samples/sec, LoopTimeRemaining=0:00:11, TimeRemaining=0:00:11
2025-12-06 04:05:12,835 INFO:   Saving checkpoint at step 200
2025-12-06 04:14:02,380 INFO:   Saved checkpoint model_dir_llama2_7b/checkpoint_200.mdl
2025-12-06 04:14:31,396 INFO:   Training completed successfully!
2025-12-06 04:14:31,403 INFO:   Processed 12800 training sample(s) in 1396.075379023 seconds.
```
<details>

<details>
batch_size: 32
```
2025-12-06 03:26:28,739 INFO:   ===========================================================================
2025-12-06 03:26:28,740 INFO:   Trainer Fit Summary
2025-12-06 03:26:28,740 INFO:   ---------------------------------------------------------------------------
2025-12-06 03:26:28,740 INFO:   Trainer will run 1 train loop.
2025-12-06 03:26:28,740 INFO:   
2025-12-06 03:26:28,740 INFO:   Train steps per train loop:
2025-12-06 03:26:28,740 INFO:   * 1 loop of 200 steps
2025-12-06 03:26:28,740 INFO:   for a total of 200 train steps.
2025-12-06 03:26:28,740 INFO:   
2025-12-06 03:26:28,740 INFO:   Checkpoints will be taken every 200 steps, for a total of 1 checkpoint.
2025-12-06 03:26:28,740 INFO:   
2025-12-06 03:26:28,740 INFO:   Progress will be logged every 50 steps.
2025-12-06 03:26:28,740 INFO:   ===========================================================================
2025-12-06 03:26:28,740 INFO:   ---------------------------------------------------------------------------
2025-12-06 03:26:28,740 INFO:   Starting train loop 1 of 1, from global step 1 to 200 (200 steps)
2025-12-06 03:26:28,740 INFO:   ---------------------------------------------------------------------------
2025-12-06 03:26:29,307 INFO:   Saving checkpoint at step 0
2025-12-06 03:26:37,919 INFO:   Saved checkpoint model_dir_llama2_7b/checkpoint_0.mdl
2025-12-06 03:26:46,495 INFO:   Compiling the model. This may take a few minutes.
2025-12-06 03:26:46,511 INFO:   Appliance client semantic version: 1.1.0, cluster server semantic version: 1.1.2, job operator semantic version: 1.1.2
2025-12-06 03:26:47,500 INFO:   Initiating a new image build job against the cluster server.
2025-12-06 03:26:47,504 INFO:   User sidecar image build is disabled from server. Falling back to venv mounting.
2025-12-06 03:26:47,525 INFO:   Appliance client semantic version: 1.1.0, cluster server semantic version: 1.1.2, job operator semantic version: 1.1.2
2025-12-06 03:26:47,528 INFO:   Initiating a new compile wsjob against the cluster server.
2025-12-06 03:26:47,549 INFO:   Job id: wsjob-gsnbpvucykpzrp4q43yn2s, workflow id: wflow-9h5recvf5oz7mwapbgrrzb, namespace: job-operator, remote log path: /n1/wsjob/workdir/job-operator/wsjob-gsnbpvucykpzrp4q43yn2s
2025-12-06 03:27:07,549 INFO:   Poll ingress status: Waiting for all Coordinator pods to be running, current running: 0/1.
2025-12-06 03:27:07,552 INFO:   Recording the timestamp when jobs is scheduled.
2025-12-06 03:27:07,566 WARNING:   Event 2025-12-06 03:26:48 +0000 UTC reason=InconsistentVersion wsjob=wsjob-gsnbpvucykpzrp4q43yn2s message='Warning: client semantic version 1.1.0 is inconsistent with cluster server semantic version 1.1.2, there's a risk job could fail due to inconsistent setup.'
2025-12-06 03:27:17,566 INFO:   Poll ingress status: Waiting for job ingress readiness.
2025-12-06 03:27:37,574 INFO:   Poll ingress status: Job ingress ready, dashboard: https://grafana.anl0.cerebras.internal/d/WebHNShVz/wsjob-dashboard?orgId=1&var-wsjob=wsjob-gsnbpvucykpzrp4q43yn2s&from=1764991018000&to=now
2025-12-06 03:27:37,582 INFO:   Poll ingress success: Job ingress ready, dashboard: https://grafana.anl0.cerebras.internal/d/WebHNShVz/wsjob-dashboard?orgId=1&var-wsjob=wsjob-gsnbpvucykpzrp4q43yn2s&from=1764991018000&to=now
2025-12-06 03:27:38,327 INFO:   Found existing cached compile with hash: "cs_356484260818439097"
2025-12-06 03:27:42,044 INFO:   Compile artifacts successfully written to remote compile directory. Compile hash is: cs_356484260818439097
2025-12-06 03:27:47,572 INFO:   Compile was successful!
2025-12-06 03:27:47,573 INFO:   Programming Cerebras Wafer Scale Cluster for execution. This may take a few minutes.
2025-12-06 03:27:49,600 INFO:   Appliance client semantic version: 1.1.0, cluster server semantic version: 1.1.2, job operator semantic version: 1.1.2
2025-12-06 03:27:49,603 INFO:   Initiating a new execute wsjob against the cluster server.
2025-12-06 03:27:49,633 INFO:   Job id: wsjob-duyk568ov8cdqfmbctuxgk, workflow id: wflow-9h5recvf5oz7mwapbgrrzb, namespace: job-operator, remote log path: /n1/wsjob/workdir/job-operator/wsjob-duyk568ov8cdqfmbctuxgk
2025-12-06 03:28:09,633 INFO:   Poll ingress status: Waiting for all Coordinator pods to be running, current running: 0/1.
2025-12-06 03:28:09,636 INFO:   Recording the timestamp when jobs is scheduled.
2025-12-06 03:28:09,652 WARNING:   Event 2025-12-06 03:27:50 +0000 UTC reason=InconsistentVersion wsjob=wsjob-duyk568ov8cdqfmbctuxgk message='Warning: client semantic version 1.1.0 is inconsistent with cluster server semantic version 1.1.2, there's a risk job could fail due to inconsistent setup.'
2025-12-06 03:28:19,652 INFO:   Poll ingress status: Waiting for job ingress readiness.
2025-12-06 03:28:39,663 INFO:   Poll ingress status: Job ingress ready, dashboard: https://grafana.anl0.cerebras.internal/d/WebHNShVz/wsjob-dashboard?orgId=1&var-wsjob=wsjob-duyk568ov8cdqfmbctuxgk&from=1764991077000&to=now
2025-12-06 03:28:39,673 INFO:   Poll ingress success: Job ingress ready, dashboard: https://grafana.anl0.cerebras.internal/d/WebHNShVz/wsjob-dashboard?orgId=1&var-wsjob=wsjob-duyk568ov8cdqfmbctuxgk&from=1764991077000&to=now
2025-12-06 03:28:39,762 INFO:   Preparing to execute using 1 CSX
2025-12-06 03:29:11,363 INFO:   About to send initial weights
2025-12-06 03:29:35,654 INFO:   Finished sending initial weights
2025-12-06 03:29:35,655 INFO:   Finalizing appliance staging for the run
2025-12-06 03:29:35,664 INFO:   Waiting for device programming to complete
2025-12-06 03:33:35,497 INFO:   Device programming is complete
2025-12-06 03:33:36,543 INFO:   Using network type: ROCE
2025-12-06 03:33:36,544 INFO:   Waiting for input workers to prime the data pipeline and begin streaming ...
2025-12-06 03:33:36,552 INFO:   Input workers have begun streaming input data
2025-12-06 03:33:37,669 INFO:   Appliance staging is complete
2025-12-06 03:33:37,669 INFO:   Beginning appliance run
2025-12-06 03:34:34,873 INFO:   | Train Device=CSX, Step=50, Loss=8.31441, Rate=30.37 samples/sec, GlobalRate=27.99 samples/sec, LoopTimeRemaining=0:02:59, TimeRemaining=0:02:59
2025-12-06 03:35:30,613 INFO:   | Train Device=CSX, Step=100, Loss=7.60903, Rate=27.69 samples/sec, GlobalRate=28.34 samples/sec, LoopTimeRemaining=0:02:02, TimeRemaining=0:02:02
2025-12-06 03:36:26,889 INFO:   | Train Device=CSX, Step=150, Loss=7.32566, Rate=27.76 samples/sec, GlobalRate=28.37 samples/sec, LoopTimeRemaining=0:01:06, TimeRemaining=0:01:06
2025-12-06 03:37:22,154 INFO:   | Train Device=CSX, Step=200, Loss=7.34266, Rate=30.97 samples/sec, GlobalRate=28.51 samples/sec, LoopTimeRemaining=0:00:10, TimeRemaining=0:00:10
2025-12-06 03:37:22,159 INFO:   Saving checkpoint at step 200
2025-12-06 03:46:16,901 INFO:   Saved checkpoint model_dir_llama2_7b/checkpoint_200.mdl
2025-12-06 03:46:45,224 INFO:   Training completed successfully!
2025-12-06 03:46:45,231 INFO:   Processed 6400 training sample(s) in 1216.494026363 seconds.
```
<details>
 
The total runtime of each job was quite different because it included compile time and setup time.
However, the real training time (the time spent running the 200 training steps) was almost the same for all batch sizes — around 20 minutes.
When we compare the actual training speed, larger batch sizes (64 and 128) ran a bit faster than batch size 32.
Overall, batch size does improve performance, but the improvement is small, and the total runtime depends heavily on whether the model needed a fresh compile or could reuse a cached one.
