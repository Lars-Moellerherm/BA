# Bachelor thesis
at TU Dortmund, Germany
## Abstract

In this thesis, a method for the optimization of the energy reconstruction for high energy cosmic photons, which are detected by the Cherenkov Telescope Array will be tested.
Currently there is a prediction by supervised machine learning regressors for every single telescope.
Because of the array structure of CTA, the analysis can be optimized by summarizing the results of these telescopes.
There is no performance gain in the simple or the weighted average over each prediction, but a nested model, which predicts every event, causes an improvement
of the relative error and for the energy resolution of CTA.
Because of the estimator´s wide number range, there is a large performance loss for low energies.
This can be managed by a transformation of the output, which ensures a performance improvement.
These methods improve the bias of the relative error and the energy resolution by 75% in a large energy range of CTA.
