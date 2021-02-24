# E(3) Equivariant Neural Network Tutorial

#### ‼ This tutorial page is for e3nn verisons < 0.2. Please see the [e3nn User Guide](https://docs.e3nn.org/en/latest/guide/guide.html) for more up to date examples.

#### For original tutorial page [click here](/e3nn_tutorial/index_orig).

### ([Recommended Reading](#reading) // [Code](#code) // [Slides](https://docs.google.com/presentation/d/1PznWO7HULKSal_fkPttho735UUmNgXXclIT6EQPaeCU/edit?usp=sharing) // [Citing](#cite) // [Feedback](#feedback))

### Tutorials by [Tess E. Smidt](https://crd.lbl.gov/departments/computational-science/ccmc/staff/alvarez-fellows/tess-smidt/)
(with additional contributions by [Mario Geiger](https://mariogeiger.ch/) and [Josh Rackers](https://cfwebprod.sandia.gov/cfdocs/CompResearch/templates/insert/profile.cfm?jracker))

#### Tess gives a special thanks to [Mario Geiger](https://e3nn.ch/), [Ben Miller](http://mathben.com/), [Kostiantyn Lapchevskyi](https://www.linkedin.com/in/klsky/) for all they do for the `e3nn` repo and them, [Daniel Murnane](https://www.linkedin.com/in/daniel-murnane-01277031/), and [Sean Lubner](https://eta.lbl.gov/people/Sean-Lubner) for many conversations that lead to the generation of the tutorial notebooks.

* * *

## Recommended Reading {#reading}
* [Cormorant: Covariant Molecular Neural Networks](https://arxiv.org/abs/1906.04015)
  * Brandon Anderson, Truong-Son Hy, Risi Kondor

* [3D Steerable CNNs: Learning Rotationally Equivariant Features in Volumetric Data](https://arxiv.org/abs/1807.02547)
  * Maurice Weiler, Mario Geiger, Max Welling, Wouter Boomsma, Taco Cohen

* [Tensor field networks: Rotation- and translation-equivariant neural networks for 3D point clouds](https://arxiv.org/abs/1802.08219)
  * Nathaniel Thomas, Tess Smidt, Steven Kearnes, Lusann Yang, Li Li, Kai Kohlhoff, Patrick Riley

* * *

## Code {#code}
For code examples, we will be using the [`e3nn` repository](https://github.com/e3nn/e3nn). Installation instructions can be found [here](https://github.com/e3nn/e3nn/#installation). To test your installation of `e3nn`, we recommend running the [following code example](https://github.com/e3nn/e3nn/blob/master/examples/point/tetris.py).

To follow along during the tutorial, we recommend you clone the tutorial repository in addition to installing `e3nn`.
```
git clone git@github.com:blondegeek/e3nn_tutorial.git
```

### Tutorial notebooks
* Data types: Going between geometric tensors in Cartesian and spherical harmonic bases and representation lists (`Rs`) in `e3nn`
  * ( [notebook](https://github.com/blondegeek/e3nn_tutorial/blob/master/data_types.ipynb) // [html](https://blondegeek.github.io/e3nn_tutorial/data_types.html) )
* Operations on Spherical Tensors: Visualization of spherical tensor addition and products
  * ( [notebook](https://github.com/blondegeek/e3nn_tutorial/blob/master/operations_on_spherical_tensors.ipynb) // [html](https://blondegeek.github.io/e3nn_tutorial/operations_on_spherical_tensors.html) )
* Simple Tasks and Symmetry: Using equivariant networks can have unintuitive consequences, we use 3 simple tasks to illustrate how network outputs must have equal or higher symmetry than inputs.
  * ( [notebook](https://github.com/blondegeek/e3nn_tutorial/blob/master/simple_tasks_and_symmetry.ipynb) // [html](https://blondegeek.github.io/e3nn_tutorial/simple_tasks_and_symmetry.html) )
* Nuts and Bolts of `e3nn`: A step by step walkthrough of how to set up a convolution and what is going on with all those `partial`s.
  * ( [notebook](https://github.com/blondegeek/e3nn_tutorial/blob/master/nuts_and_bolts_of_e3nn.ipynb) )
* Plot with radial functions: Now you can plot angular and radial Fourier transforms of geometry
  * ( [notebook](https://github.com/blondegeek/e3nn_tutorial/blob/master/example_plot_with_radial.ipynb) // [html](https://blondegeek.github.io/e3nn_tutorial/example_plot_with_radial.html) )
* NEW! Creating neighbor lists for molecules and crystals using `e3nn.point.data_helpers`.
  * ( [notebook](https://github.com/blondegeek/e3nn_tutorial/blob/master/datatypes_for_neighbors.ipynb) )

#### Why notebook AND html?
For the notebooks that use `plotly` the notebooks are distributed without cells executed because the plots are large (because Tess made them too high-resolution... oops.). If you download the HTML verison, you can interact with the plots without needing to execute the code.

### Citing {#cite}
If you find these tutorials helpful for your research, please consider citing us!

The DOI for these tutorials is:
[![DOI](https://zenodo.org/badge/221095368.svg)](https://zenodo.org/badge/latestdoi/221095368)

Cite this tutorial with the following `bibtex`:
```
@software{e3nn_tutorial_2020_3724982,
  author       = {Tess Smidt and
		  Mario Geiger and
		  Josh Rackers},
  title        = {github.com/blondegeek/e3nn_tutorial},
  month        = mar,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {0.1},
  doi          = {10.5281/zenodo.3724982},
  url          = {https://doi.org/10.5281/zenodo.3724982}
}
```

The DOI for `e3nn` is:
[![DOI](https://zenodo.org/badge/237431920.svg)](https://zenodo.org/badge/latestdoi/237431920)

Cite `e3nn` with the following `bibtex`:
```
@software{e3nn_2020_3723557,
  author       = {Mario Geiger and
                  Tess Smidt and
                  Benjamin K. Miller and
                  Wouter Boomsma and
                  Kostiantyn Lapchevskyi and
                  Maurice Weiler and
                  Michał Tyszkiewicz and
                  Jes Frellsen},
  title        = {github.com/e3nn/e3nn},
  month        = mar,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {v0.3-alpha},
  doi          = {10.5281/zenodo.3723557},
  url          = {https://doi.org/10.5281/zenodo.3723557}
}
```

#### Got feedback on the code tutorials? {#feedback}
Tess wants to hear all about it, so please, please, please write Tess an email at `tsmidt@lbl.gov` or `blondegeek@gmail.com`! The goal is to make these notebooks maximally useful to others. 

Is there a tutorial you'd love to see? Is there a tutorial you'd like to contribute? Add an issue or make a pull request!
* * *

