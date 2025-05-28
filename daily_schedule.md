## Daily Schedule (Actual &mdash; Kept Retrospectively)

*Regular meeting schedule is Wednesdays and Saturdays, 11:00-12:00*

Back to [Course home page](../index.html)

### Part 1: Scientific Python (using Imad Pasha, *Astronomical Python*)

#### Week 1 &mdash; Shell and Python Quick-Start/Review

* May 16 &mdash; Complete Chapters 1 to 3: Unix (shell) Basics, Installing Python, and the Astronomy/Scientific Data Analysis Stack &mdash; Problem Set 0: Get Anaconda downloaded and installed and use the IPython interface &mdash; Discussed Python language features, syntax, and style (PEP 8), differences between Windows and Unix shells, globbing, and Python's Operating System Insulation Layer (OSIL)
* May 17 &mdash; Complete Chapter 4: Introduction to Python &mdash; Problem Set 1: Use for loops to compute the first 20 Fibonacci numbers (screenshot your solution in IPython) &mdash; Discussed notebook tools and IDEs

#### Week 2 Matplotlib and Numpy

* May 21 &mdash; Complete Chapter 5: Visualization with Matplotlib ([&sect;2 A Simple Plot](./pasha/c05/c05s02.ipynb), [&sect;4 Subplots](./pasha/c05/c05s04.ipynb), [&sect;5 Adjusting Marker Properties](./pasha/c05/c05s05.ipynb), [&sect;6 Adjusting Ticks](./pasha/c05/c05s06.ipynb), [&sect;7 Adjusting Fonts and Font Sizes](./pasha/c05/c05s07.ipynb), [&sect;8 Multiple Subplots](./pasha/c05/c05s08.ipynb), [&sect;9 Subplot Mosaic](./pasha/c05/c05s09.ipynb), [&sect;10 Research Example: Displaying a Best Fit](./pasha/c05/c05s10.ipynb), [&sect;11 Error Bars](./pasha/c05/c05s11.ipynb), [&sect;12 Plotting *n*-Dimensional Data](./pasha/c05/c05s12.ipynb), [&sect;13 Color Bars](./pasha/c05/c05s13.ipynb)) &mdash; Install (if not already part of your Python distribution) and start working in Jupyter Lab &mdash; Problem Set 2: Make some histogram and scatter plots using the [iris dataset](./iris/iris_dataset.csv) (save your plots as a Jupyter Lab notebook) ([PS02](./psets/ps02.ipynb))
* May 25 &mdash; Complete Chapter 6: Numerical Computing with NumPy ([&sect;5 Research Example: An Exoplanet Transit](./pasha/c06/c06s05.ipynb)) &mdash; Create a github account, fork the repo: brianhill/scientific-data-analysis &mdash; Then figure out how to get a local copy onto your machine of your forks (hexijin/scientific-data-analysis or jeremychoy/scienctific-data-analsysis) and this will involve installing git on your machine (which will be different for Mac or Windows) &mdash; Started learning shell access to git, and the add, commit, push cycle (which we will be adding more to once that is routine)

#### Week 3 &mdash; SciPy and AstroPy

* May 28 &mdash; Complete Chapter 7: Scientific Computing with SciPy ([&sect;2 Numerical Integration](./pasha/c07/c07s02.ipynb), [&sect;3 Optimization](./pasha/c07/c07s03.ipynb), [&sect;4 Statistics](./pasha/c07/c07s04.ipynb)) &mdash; Problem Set 3 (in addition to working through all the code in the chapter): Do Exercise 7.1 ([PS03](./psets/ps03.ipynb)), and show Jeremy how to save his first version of `scientific-data-analysis/jeremy/pasha07/c07.ipynb` (which will require some `mkdir` and `cd` commands in his local git repository), and then how to use `git` to add the new notebook, commit it, and push it (and you should both repeat the add-commit-push cycle while building out your notebooks until executing those commands in the shell is completely routine)

* May 31 &mdash; Complete Chapter 8: Astropy and Astronomical Packages

### Part 2: Data Science Foundations (using Joel Grus, *Data Science from Scratch, 2nd Edition*)

#### === BELOW THIS DIVISION IS GOAL/TENTATIVE PLAN &mdash; NOT ACTUAL ===

#### Week 4 &mdash; Another 

* June 4 &mdash; Chapters 1-4: Grus's review of Python and Matplotlib and then Linear Algebra (Grus introduces his Vector and Matrix classes)
* June 7 &mdash; Chapters 5 and 6: Statistics and Probability

#### Week 5 &mdash; Optimization

* June 11 &mdash; Chapters 7 and 8: Hypotheses &amp; Inference and Gradient Descent
* June 14 &mdash; Chapters 9 and 10: Getting Data and Working with Data

#### Week 6 &mdash; Machine Learning and Neural Networks

* June 18 &mdash; Chapters 11 and 13: Machine Learning and Naive Bayes
* June 21 &mdash; Chapters 18 and 19: Neural Networks and Deep Learning

#### Week 7 &mdash; Deep Learning and Natural Language Processing

* June 25 &mdash; A great finale for the Data Science Portion of the IS would be to (1) follow along with this live coding session (and build the code with him, pausing when you need to catch up): [Joel Grus - Building a Deep Learning Library](https://joelgrus.com/2017/12/04/livecoding-madness-building-a-deep-learning-library/) and (2) to do Chapter 21, Natural Language Processing (which does not include the 2017 &ldquo;Attention is All You Need&rdquo; transformers revolution, but which sets you up for a course on LLMs)
