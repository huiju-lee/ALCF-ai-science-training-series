##Homework 1: ml-in-the-loop
```
initial_training_count = 10 
max_training_count = 25  
batch_size = 5 
```
Iteration 1:
	Training on 10/130258 random molecules
	Best predicted molecule: O=C1CC(=O)N2CC12 with ionization energy 14.36 Ha
	Performed 5 new simulations
	Estimate of KNN Model Mean Relative Error (MRE): 0.03 %
	Finished loop iteration in 7.24s

Iteration 2:
	Training on 15/130258 random molecules
	Best predicted molecule: O=CC(=O)N1N=NN=N1 with ionization energy 15.28 Ha
	Performed 5 new simulations
	Estimate of KNN Model Mean Relative Error (MRE): 0.10 %
	Finished loop iteration in 3.45s

Iteration 3:
	Training on 20/130258 random molecules
	Best predicted molecule: O=CC(=O)N1N=NN=N1 with ionization energy 15.28 Ha
	Performed 5 new simulations
	Estimate of KNN Model Mean Relative Error (MRE): 0.06 %
	Finished loop iteration in 2.66s

Training completed in 22.66 seconds

![results](./parsl_ml_in_the_loop.png)
