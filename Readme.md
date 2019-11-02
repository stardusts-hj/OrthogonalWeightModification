Replication of Orthogonal Weights Modification (https://www.nature.com/articles/s42256-019-0080-x)  
Evaluated with a splitMNIST experiment.

### Usage
I've implemented three variants of the algorithm:
* "none": no OWM updates at all  
* "batch": OWM, updates of projection matrix after each training batch
* "task": OWM, update of projection matrix after each task

To run the experiment, call **expt_owm.py** from the command line.
Example:
```bash
python expt_owm.py
```
To run for example a training regime with OWM updates after each task, call
```bash
python expt_owm.py -owm task
```
I've defined several other flags that can be passed as command line arguments. Have a look at **expt_owm.py** for further info
