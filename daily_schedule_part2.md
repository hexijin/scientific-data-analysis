## Daily Schedule Part 2 (Actual &mdash; Kept Retrospectively)

*Regular meeting schedule is Wednesdays and Saturdays, 11:00-12:00*

Back to [Course home page](./index.html)

See also [Daily Schedule - Part 1](./daily_schedule_part1.html)

### Part 2: Data Science Foundations (using Joel Grus, *Data Science from Scratch, 2nd Edition*)

Part 2 Uses Grus and lasts for the remaining three-and-a-half weeks of Term 6

#### Week 4 &mdash; Yet Another Review of Python &mdash; Some Vector and Matrix Algebra &mdash; Statistics and Probability

* June 4 &mdash; Chapters 1-3: Another excellent review of Python and Matplotlib which will help systematize your understanding of the language features you were using in Pasha's book &mdash; The assignment is to do the review of the three chapters, but to completely stop using Jupyter or Jupyter lab, and instead get everything working in PyCharm Professional Edition (free for students) or VS Code (but I have zero experience with that) &mdash; When Grus says (at the beginning of Chapter 2) that you should not be tampering with your base Python environment, he is completely correct (so learn how to make a venv that you could call grus or dsfs and then switch to it &mdash; if you didn't already do that for working through Pasha) &mdash; [Ch. 3: Visualizing Data](./grus/ch03/grus_ch03_code.py)
* June 7 &mdash; Chapters 4-6: Linear Algebra (wherein Grus introduces his Vector and Matrix implementations which could have been classes, or could have leveraged numpy, but which he craftily used type aliases, because that was the simplest way to implement from scratch), Statistics, and Probability (due to having taken last fall's Bayesian Statistics class, the math in Chapters 5 and 6 will be review) &mdash; [Ch. 4: Linear Algebra](./grus/ch04/grus_ch04_code.py)

#### === BELOW THIS DIVISION IS GOAL/TENTATIVE PLAN &mdash; NOT ACTUAL ===

#### Week 5 &mdash; Optimization (aka Minimization and Maximization) &mdash; Working with Data

* June 11 &mdash; Chapters 7 and 8: Hypotheses &amp; Inference and Gradient Descent
* June 14 &mdash; Chapters 9 and 10: Getting and Working with Data

#### Week 6 &mdash; Machine Learning &mdash; Neural Networks &mdash; Start Deep Learning

* June 18 &mdash; Chapters 11 and 13: Machine Learning and Naive Bayes
* June 21 &mdash; Chapters 18 and 19: Neural Networks and Deep Learning &mdash; Good prepartion for your coding assessment, would be to do this real-time coding session to see how a real pro codes, including type-hinting, systematic adherence to style choices, and code testing: [Joel Grus - Building a Deep Learning Library](https://joelgrus.com/2017/12/04/livecoding-madness-building-a-deep-learning-library/) (build the code in PyCharm as Grus builds it in VS Code, pausing the video whenever you need to catch up with him) &mdash; This real-time coding session will also give you a blindingly fast overview of Chapters 18 and 19

#### Week 7 &mdash; Continue Deep Learning &mdash; Introduction to Natural Language Processing

* June 25 &mdash; Chapter 21: Natural Language Processing &mdash; Watch the fifth of the 3Blue1Brown videos, [Transformers Explained Visually](https://youtu.be/wjZofJX0v4M), by Grant Sanderson numbered DL1 to DL7 &mdash; "DL" is short for "Deep Learning," and the seven videos were published from 2017 to 2024 &mdash; The fifth video gives you alook into the 2017 transformers revolution, and at what you would study next if you want to keep getting closer to the state of the art of machine learning and LLMs

#### Looking Beyond

Although the chapter we finished with did not attempt to cover the 2017 "[Attention is All You Need](https://arxiv.org/abs/1706.03762)" paper, Grus has almost perfectly set you up for a second-semester course in deep learning and LLMs that almost every computer science department is now going to have on offer. Before taking such a course, you can review and survey beyond where Grus has taken us in several ways:

(1) If at some point, you are interested in a concise and mathematically-sophisticated review of all that we have done (and then some), consider [A high-bias, low-variance introduction to Machine Learning for physicists](./references/MachineLearningForPhysicists.pdf).

(2) Grus considers how LLMs have changed and will continue to change the workflow of a data scientist by watching from the 15:00 mark in this late-2023 video [Doing Data Science in the Time of ChatGPT](https://youtu.be/oyV81rnLSJc?t=900). This is a casual survey that may only serve to cement what you have already discovered you can do with a current-generation LLM like Grok 3 or ChatGPT 4.5.

(3) Recapitulate what we have done and then look inside the mathematics and implementation of LLMs, without actually doing any more implementation, by watching all seven of the 3Blue1Brown videos by Grant Sanderson. Sanderson's mathematics visualizations are unequaled, and I enjoy watching them even when he is presenting something I already understand, but perhaps the first few in the series are not worth your time given how much we have covered in Grus. The last few will certainly be worthwhile. 

(4) How this is going to affect industry after industry is anybody's guess, but a recent and informed guess (from venture capitalist Marc Andreessen) is in this late-2024 [Lex Fridman interview of Marc Andreessen](https://youtu.be/OHWnPOKh_S0?feature=shared&t=11849) (the link deliberately jumps you to a point over three hours into the video).
