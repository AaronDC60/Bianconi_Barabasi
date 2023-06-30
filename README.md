# Bianconi-Barabasi model for complex networks

## Project description

The Barabasi-Albert model uses the concepts of growth and preferential attachment (based on the degree of the nodes) to evolve into a complex network that has a scale-free degree distribution. 
However, due to the fact that the preferential attachment is only based on the degree of the nodes, the model cannot take into account a phenomenon where nodes that might emerge later could also have the potential to become a large hub.

In this project, we will work on an extended version of this model, the Bianconi-Barabasi model, where a notion of a node having a certain fitness is introduced.
In this model, every node is assigned a fitness value, which is used together with the degree to determine the preferential growth.

## Content

This repository contains:
* The [plan](https://github.com/AaronDC60/Bianconi_Barabasi/blob/main/Project%20Plan.docx) of the project
* The [slides](https://github.com/AaronDC60/Bianconi_Barabasi/blob/main/BBB-%20presentation.pptx) of a presentation that was given about the project.
* A [src](https://github.com/AaronDC60/Bianconi_Barabasi/tree/main/src) folder with all the source code of the model.
* A [notebooks](https://github.com/AaronDC60/Bianconi_Barabasi/tree/main/notebooks) folder with all `.ipynb` files used to analyse the model/real network.
* A [data](https://github.com/AaronDC60/Bianconi_Barabasi/tree/main/data) folder with model output used for analysis and real world data.
* A folder with all the [figures/gifs](https://github.com/AaronDC60/Bianconi_Barabasi/tree/main/results) made during the project.
* A [requirements file](https://github.com/AaronDC60/Bianconi_Barabasi/blob/main/requirements.txt) containing all the packages that are necessary to run the code.

To install all the required packages, use the following command.
```
pip install -r requirements.txt
```

### src

The source code of the model is spread over three files:
* [`model.py`](https://github.com/AaronDC60/Bianconi_Barabasi/blob/main/src/model.py), which contains the implementation of the Bianconi-Barabasi model.
* [`fitness.py`](https://github.com/AaronDC60/Bianconi_Barabasi/blob/main/src/fitness.py), which contains the implementation of a class that can generate values from a number of different distributions.
* [`utils.py`](https://github.com/AaronDC60/Bianconi_Barabasi/blob/main/src/utils.py), which contains several helper functions (for the analysis of the degree dynamics).

In the src folder, there is a [tests folder](https://github.com/AaronDC60/Bianconi_Barabasi/tree/main/src/tests) to check if all the classes work properly.
These tests can be ran from the terminal using the following command:
```
pytest .
```

### Notebooks

* [`model_analysis_1.ipynb`](https://github.com/AaronDC60/Bianconi_Barabasi/blob/main/notebooks/model_analysis_1.ipynb) : Analysis of the degree distribution, network properties, statistical tests and network visualization of model generated networks with different fitness distributions. The notebook starts with a section to generate networks from scratch but the networks used for the analysis are stored in the [data/model](https://github.com/AaronDC60/Bianconi_Barabasi/tree/main/data/model) folder.
* [`model_analysis_2.ipynb`](https://github.com/AaronDC60/Bianconi_Barabasi/blob/main/notebooks/model_analysis_2.ipynb) : Analysis of the degree dynamics for networks generated with a delta and uniform fitness distribution. Note that depending on the device, the kernel might crash because of high memory usage. 
* [`fitness_estimation.ipynb`](https://github.com/AaronDC60/Bianconi_Barabasi/blob/main/notebooks/fitness_estimation.ipynb) : Estimation of the fitness distribution of model generated networks.
* [`phase_transition.ipynb`](https://github.com/AaronDC60/Bianconi_Barabasi/blob/main/notebooks/phase_transition.ipynb) : Analysis of the phase transition as a function of the shape of the fitness distribution. Note that the calculation takes several days, the stored model output (1.4GB) used to make the plot is located on [Google drive](https://drive.google.com/file/d/1JTjsyl3gCg21QXyxXGg70CB5mi_GVm8u/view?usp=drive_link).
* [`distributions.ipynb`](https://github.com/AaronDC60/Bianconi_Barabasi/blob/main/notebooks/distributions.ipynb) : Visualization of several different distributions from which the model can sample.
* [`real_network.ipynb`](https://github.com/AaronDC60/Bianconi_Barabasi/blob/main/notebooks/real_network.ipynb) : Analysis of a real network for which the input file is located in the [data](https://github.com/AaronDC60/Bianconi_Barabasi/tree/main/data) folder.
