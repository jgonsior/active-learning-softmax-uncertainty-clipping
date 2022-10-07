> Code for Research Paper "To Softmax, or not to Softmax: that is the question when applying Active Learning for Transformer Models"
<hr>

## Running the experiments
First, make sure to install the needed dependecies as defined in `Pipfile`, preferably using Pipenv.

We have defined the parameter grid of our experiments at the beginning of the file `run_experiment.py`. To run a single workload use:

```bash
python run_experiment.py --workload baselines --n_array_jobs 1 --array_job_id 0
```

By appending the CLI parameter `--dry_run` you can examine first, what the single experimnt runs are. 

If you hav eaccess to SLURM based HPC cluster, you can also use our SLURM files.

For evaluation, the file `evaluate_experiments.py` can be used to generate the plots of the paper (and more), simply uncomment at the bottom of the file the desired plots (note that some of them use a lot of Memory and might take some time).

Upon request, we can also provide you access to the raw experiment results (~20GB of Data) in order to save a lot of computer ressources on your end. Or, if you know a place where we can host freely host our 20GB of experiment results, please also feel free to reach out to us.

## Acknowledgments

The underlying [small-text](https://github.com/webis-de/small-text) framework is a software created by Christopher Schröder ([@chschroeder](https://github.com/chschroeder)) at Leipzig University's [NLP group](http://asv.informatik.uni-leipzig.de/) 
which is a part of the [Webis](https://webis.de/) research network. 
