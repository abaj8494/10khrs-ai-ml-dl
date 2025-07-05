# Table of Contents

1.  [Context](#org6dbe148)
2.  [Structure of the Repository](#org65603ff)
3.  [Problems](#problems)
	- [Scoreboard](#scoreboard)
4.  [Learning](#learning)
5.  [Projects](#projects)
6.  [papers](#papers)

<a id="org6dbe148"></a>

# Context

This repo is my 10,000 Hours of Machine Learning.

[Peter Norvig's Post](https://norvig.com/21-days.html)<a href="https://norvig.com/21-days.html"><img src="img/rat.png" alt="Auguste Gusteau" style="vertical-align: text-bottom; width: 22px; height: auto;"></a>

*Note: this repository grew in size and scope until the singularity event [abaj.ai](https://abaj.ai) <a href="https://abaj.ai/"><img src="https://abaj.ai/abs_hsv.svg" alt="Aayush Bajaj's Augmenting Infrastructure" style="vertical-align: text-bottom; width: 16px; height: auto;"></a>.*

All of the files in this repository are submodules of that site / [monolithic-repo](https://github.com/abaj8494/site), with much of this code being tangled (in the Emacs sense) to the site's original code blocks.


> "Machine Learning is just lego for adults" - Dr. Kieran Samuel Owens

> "S/he who has a why, can bear almost any how." - Friedrich Nietzche

> "If you can do it without Machine Learning, do it without Machine Learning" - Google ML Handbook, Rule 1

---

To become an expert at anything, there is a common denominator:

<div class="org-center">
<p>
10,000 hours of <b>deliberate practise</b> on the subject.
</p>
</div>


<a id="org65603ff"></a>

# Structure of this Repository

There are 3 main features:

<a name="problems"></a>

<details>
<summary><h2>Problems</h2></summary>

-   These are solutions to **classical** problems, MNIST, Boston Housing, XOR, etc.
-   They let me practise particular ML algorithms such as SVM's, Logistic Regression, Decision Trees, etc.
-   The file structure is:
	- 0.tools
	- 1.supervised learning
	- 2.unsupervised learning
	- 3.transfer learning
	- 4.deep learning
	- 5.reinforcement learning

<a id="#scoreboard"></a>

### Scoreboard

- MNIST, Multiclass Classification: **99.6%**, CNN with Regularisation
- Pima Indians, Binary Classification: **85%**, Decision Trees
- Cifar10, Multiclass Classification: 3-layer CNN: **55%**
- Caltech10, Transfer Learning with ResNet: **93%**
- FMNIST, Multiclass Classification: **93%**. CNN with Batch Norm.
- Ionosphere, MLP, Pytorch implementation: **85.3%**. Binary Classification. Xavier, Kaiming inits.
- Titanic, Binary Classification: **82%** Decision Trees.
- Life Expectancy (WHO). Regression; **0.29** R2 score. LOOCV experiments.
- Wine Multiclass Classif: **0.981481**: RandomForest and Linear SVM. Confusion Matrix, AdaBoost, LogReg, DT.

</details>

<a name="learning"></a>

<details>
<summary><h2>Problems</h2></summary>

-   This folder contains the code I have written whilst following along with textbooks, open courseware and YouTube playlists. Some notable features are:
    - Andrej Karpathy's entire [Zero to Hero series](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) series.
    - Michael Neilson's book ["Neural Networks and Deep Learning"](http://neuralnetworksanddeeplearning.com/)
    - d2l by [fast ai](https://fast.ai)
</details>

<a name="projects"></a>
<details>
<summary><h2>Projects</h2></summary>

-   These are my non-trivial, novel pearls.
-   These projects are unique and were challenging / instructive to solve:
    - Peg Solitaire. recursive backtracking, rotation invariant algorithm. <a href="https://abaj.ai/projects/csp/peg-solitaire"><img src="https://abaj.ai/abs_hsv.svg" alt="Peg Solitaire" style="vertical-align: text-bottom; width: 16px; height: auto;"></a>
    - Hashiwokakero. reimplemented in Go with a test suite, see the spin-off [repo](https://github.com/abaj8494/hashi)
    - Kits19: Kidney Segmentation using 2 and 3D volumetric data from CT scans. 57th on world [leaderboard](https://kits19.grand-challenge.org/evaluation/965bcad2-cbb9-42a8-8b56-a777c9f165e2/)
    - OCR K-means _parallelised_
    - SVM with Monster (group theory dimension); no real benefit other than novelty
    - similarly, `erf-function` as a _loss_ function 😹
    - interview prep..
</details>


<a name="papers"></a>
<details>
<summary><h1>Papers</h1></summary>

> "Read 2 papers a week" - Andrew Ng

<table>

<tr valign="top">
<td align="right" class="bibtexnumber">
[<a name="Shneiderman2020">1</a>]
</td>
<td class="bibtexitem">
B.&nbsp;Shneiderman.
 Human-centered artificial intelligence: Three fresh ideas.
 In <em>AIS Transactions on Human-Computer Interaction</em>, volume&nbsp;12,
  pages 109--124, 2020.

</td>
</tr>


<tr valign="top">
<td align="right" class="bibtexnumber">
[<a name="Friedman1996">2</a>]
</td>
<td class="bibtexitem">
B.&nbsp;Friedman.
 Value-sensitive design.
 <em>interactions</em>, 3(6):16--23, 1996.

</td>
</tr>


<tr valign="top">
<td align="right" class="bibtexnumber">
[<a name="gabriel2021challenge">3</a>]
</td>
<td class="bibtexitem">
Iason Gabriel and Vafa Ghazavi.
 The challenge of value alignment: From fairer algorithms to ai
  safety.
 <em>Minds and Machines</em>, 31(4):629--653, 2021.
[&nbsp;<a href="http://dx.doi.org/10.1007/s11023-021-09563-0">DOI</a>&nbsp;]

</td>
</tr>


<tr valign="top">
<td align="right" class="bibtexnumber">
[<a name="munn2023uselessness">4</a>]
</td>
<td class="bibtexitem">
Luke Munn.
 The uselessness of ai ethics.
 <em>AI and Society</em>, 2023.
[&nbsp;<a href="http://dx.doi.org/10.1007/s00146-023-01673-z">DOI</a>&nbsp;]

</td>
</tr>


<tr valign="top">
<td align="right" class="bibtexnumber">
[<a name="Baeza2018">5</a>]
</td>
<td class="bibtexitem">
R.&nbsp;Baeza-Yates.
 Bias on the web.
 <em>Communications of the ACM</em>, 61(6):54--61, 2018.

</td>
</tr>


<tr valign="top">
<td align="right" class="bibtexnumber">
[<a name="associationforcomputingmachinery_2018_acm">6</a>]
</td>
<td class="bibtexitem">
Association for Computing&nbsp;Machinery.
 Acm code of ethics and professional conduct, 06 2018.
[&nbsp;<a href="https://www.acm.org/code-of-ethics">http</a>&nbsp;]

</td>
</tr>


<tr valign="top">
<td align="right" class="bibtexnumber">
[<a name="malizia2023xai">7</a>]
</td>
<td class="bibtexitem">
Alessio Malizia and Fabio Paternò.
 Why is the current xai not meeting the expectations?
 <em>Communications of the ACM</em>, 66(12):20--22, 2023.
[&nbsp;<a href="http://dx.doi.org/10.1145/3588313">DOI</a>&nbsp;]

</td>
</tr>


<tr valign="top">
<td align="right" class="bibtexnumber">
[<a name="bietti2020ethics">8</a>]
</td>
<td class="bibtexitem">
Elettra Bietti.
 From ethics washing to ethics bashing: A view on tech ethics from
  within moral philosophy.
 <em>Philosophy &amp; Technology</em>, 33(4):541--559, 2020.
[&nbsp;<a href="http://dx.doi.org/10.1007/s13347-020-00405-1">DOI</a>&nbsp;| 
<a href="https://doi.org/10.1007/s13347-020-00405-1">http</a>&nbsp;]

</td>
</tr>


<tr valign="top">
<td align="right" class="bibtexnumber">
[<a name="taddeo2022moral">9</a>]
</td>
<td class="bibtexitem">
Mariarosaria Taddeo and Alexander Blanchard.
 Accepting moral responsibility for the actions of autonomous weapons
  systems—a moral gambit.
 <em>Philosophy &amp; Technology</em>, 35(3):1--24, 2022.
[&nbsp;<a href="http://dx.doi.org/10.1007/s13347-022-00571-x">DOI</a>&nbsp;]

</td>
</tr>


<tr valign="top">
<td align="right" class="bibtexnumber">
[<a name="daniels2020reflective">10</a>]
</td>
<td class="bibtexitem">
Norman Daniels.
 Reflective equilibrium.
 
  <a href="https://plato.stanford.edu/archives/sum2020/entries/reflective-equilibrium/">https://plato.stanford.edu/archives/sum2020/entries/reflective-equilibrium/</a>,
  2020.
 Stanford Encyclopedia of Philosophy, Summer 2020 Edition, edited by
  Edward N. Zalta.

</td>
</tr>


<tr valign="top">
<td align="right" class="bibtexnumber">
[<a name="beauchamp2004principles">11</a>]
</td>
<td class="bibtexitem">
Tom&nbsp;L. Beauchamp and David DeGrazia.
 Principles and principlism.
 In Raanan Gillon, editor, <em>Principles of Health Care Ethics</em>,
  pages 55--66. John Wiley &amp; Sons, 2004.

</td>
</tr>


<tr valign="top">
<td align="right" class="bibtexnumber">
[<a name="iep_metaethics">12</a>]
</td>
<td class="bibtexitem">
Geoffrey Sayre-McCord.
 Metaethics.
 <a href="https://iep.utm.edu/metaethi/">https://iep.utm.edu/metaethi/</a>.
 Internet Encyclopedia of Philosophy.

</td>
</tr>


<tr valign="top">
<td align="right" class="bibtexnumber">
[<a name="ashford2012developing">13</a>]
</td>
<td class="bibtexitem">
Susan&nbsp;J. Ashford.
 Developing as a leader: The power of mindful engagement.
 <em>Organizational Dynamics</em>, 41(2):146--154, 2012.

</td>
</tr>


<tr valign="top">
<td align="right" class="bibtexnumber">
[<a name="formosa2021principlist">14</a>]
</td>
<td class="bibtexitem">
Paul Formosa, Michael Wilson, and Deborah Richards.
 A principlist framework for cybersecurity ethics.
 <em>Computers &amp; Security</em>, 105:102226, 2021.
[&nbsp;<a href="http://dx.doi.org/10.1016/j.cose.2021.102226">DOI</a>&nbsp;]

</td>
</tr>


<tr valign="top">
<td align="right" class="bibtexnumber">
[<a name="macnish2020ethics">15</a>]
</td>
<td class="bibtexitem">
Kevin Macnish and Jeroen van&nbsp;der Ham.
 Ethics in cybersecurity research and practice.
 <em>Technology in Society</em>, 63:101382, 2020.
[&nbsp;<a href="http://dx.doi.org/10.1016/j.techsoc.2020.101382">DOI</a>&nbsp;]

</td>
</tr>


<tr valign="top">
<td align="right" class="bibtexnumber">
[<a name="sequoiahgrayson_unsatisfiable_triade">16</a>]
</td>
<td class="bibtexitem">
Sebastian Sequoiah-Grayson.
 The unsatisfiable triad: A problem for automated decision making.
 Unpublished manuscript, 2025.
[&nbsp;<a href="https://logicalrockpools.com/">http</a>&nbsp;]

</td>
</tr>


<tr valign="top">
<td align="right" class="bibtexnumber">
[<a name="doshi2017towards">17</a>]
</td>
<td class="bibtexitem">
Finale Doshi-Velez and Been Kim.
 Towards a rigorous science of interpretable machine learning.
 <em>arXiv preprint arXiv:1702.08608</em>, 2017.
[&nbsp;<a href="https://arxiv.org/abs/1702.08608">http</a>&nbsp;]

</td>
</tr>


<tr valign="top">
<td align="right" class="bibtexnumber">
[<a name="cornacchia2023counterfactualreasoningbiasevaluation">18</a>]
</td>
<td class="bibtexitem">
Giandomenico Cornacchia, Vito&nbsp;Walter Anelli, Fedelucio Narducci, Azzurra
  Ragone, and Eugenio&nbsp;Di Sciascio.
 Counterfactual reasoning for bias evaluation and detection in a
  fairness under unawareness setting, 2023.
[&nbsp;<a href="http://arxiv.org/abs/2302.08204">arXiv</a>&nbsp;| 
<a href="https://arxiv.org/abs/2302.08204">http</a>&nbsp;]

</td>
</tr>


<tr valign="top">
<td align="right" class="bibtexnumber">
[<a name="10.1145/3433949">19</a>]
</td>
<td class="bibtexitem">
Sorelle&nbsp;A. Friedler, Carlos Scheidegger, and Suresh Venkatasubramanian.
 The (im)possibility of fairness: different value systems require
  different mechanisms for fair decision making.
 <em>Commun. ACM</em>, 64(4):136–143, March 2021.
[&nbsp;<a href="http://dx.doi.org/10.1145/3433949">DOI</a>&nbsp;| 
<a href="https://doi.org/10.1145/3433949">http</a>&nbsp;]
<blockquote>
What does it mean to be fair?
</blockquote>

</td>
</tr>


<tr valign="top">
<td align="right" class="bibtexnumber">
[<a name="Manna2021">20</a>]
</td>
<td class="bibtexitem">
R.&nbsp;Manna and R.&nbsp;Nath.
 Kantian moral agency and the ethics of artificial intelligence.
 <em>Problemos</em>, 100:139--151, 2021.

</td>
</tr>


<tr valign="top">
<td align="right" class="bibtexnumber">
[<a name="Nath2021">21</a>]
</td>
<td class="bibtexitem">
R.&nbsp;Nath and V.&nbsp;Sahu.
 The problem of machine ethics in artificial intelligence.
 <em>AI &amp; Society</em>, 35:103--111, 2021.

</td>
</tr>


<tr valign="top">
<td align="right" class="bibtexnumber">
[<a name="Tonkens2009">22</a>]
</td>
<td class="bibtexitem">
R.&nbsp;Tonkens.
 A challenge for machine ethics.
 <em>Minds &amp; Machines</em>, 19:421--438, 2009.

</td>
</tr>


<tr valign="top">
<td align="right" class="bibtexnumber">
[<a name="Singh2022">23</a>]
</td>
<td class="bibtexitem">
L.&nbsp;Singh.
 Automated kantian ethics: A faithful implementation, 2022.
 Online at <a href="https://github.com/lsingh123/automatedkantianethics">https://github.com/lsingh123/automatedkantianethics</a>.

</td>
</tr>


<tr valign="top">
<td align="right" class="bibtexnumber">
[<a name="EC_HLEG_AI">24</a>]
</td>
<td class="bibtexitem">
European Commission’s High-Level Expert&nbsp;Group on&nbsp;Artificial&nbsp;Intelligence.
 Ethics guidelines for trustworthy artificial intelligence.
 Technical Report&nbsp;6, European Commission, 2019.
 p. 17.

</td>
</tr>


<tr valign="top">
<td align="right" class="bibtexnumber">
[<a name="Fjeld2020">25</a>]
</td>
<td class="bibtexitem">
J.&nbsp;Fjeld, N.&nbsp;Achten, H.&nbsp;Hilligoss, A.&nbsp;C. Nagy, and M.&nbsp;Srikumar.
 Principled artificial intelligence: Mapping consensus in ethical and
  rights-based approaches to principles for ai.
 <em>arXiv preprint arXiv:2009.06350</em>, 2020.

</td>
</tr>


<tr valign="top">
<td align="right" class="bibtexnumber">
[<a name="BentzenLindner2018">26</a>]
</td>
<td class="bibtexitem">
M.&nbsp;M. Bentzen and F.&nbsp;Lindner.
 A formalization of kant's second formulation of the categorical
  imperative, 2018.
 CoRR abs/1801.03160.
[&nbsp;<a href="http://arxiv.org/abs/1801.03160">arXiv</a>&nbsp;| 
<a href="http://arxiv.org/abs/1801.03160">http</a>&nbsp;]

</td>
</tr>


<tr valign="top">
<td align="right" class="bibtexnumber">
[<a name="searle1996">27</a>]
</td>
<td class="bibtexitem">
John&nbsp;R. Searle.
 <em>Minds, Brains, and Science</em>.
 Harvard University Press, Cambridge, 1996.
 See p. 41.

</td>
</tr>


<tr valign="top">
<td align="right" class="bibtexnumber">
[<a name="Powers2006">28</a>]
</td>
<td class="bibtexitem">
Tom&nbsp;M. Powers.
 Prospects for a Kantian machine.
 <em>IEEE Intelligent Systems</em>, 21(4):46--51, 2006.

</td>
</tr>


<tr valign="top">
<td align="right" class="bibtexnumber">
[<a name="Kant1785p49">29</a>]
</td>
<td class="bibtexitem">
Immanuel Kant.
 <em>Fundamental Principles of the Metaphysic of Morals</em>.
 Prometheus Books, New York, 1785/1988.

</td>
</tr>


<tr valign="top">
<td align="right" class="bibtexnumber">
[<a name="Kant1785p58">30</a>]
</td>
<td class="bibtexitem">
Immanuel Kant.
 <em>Fundamental Principles of the Metaphysic of Morals</em>.
 Prometheus Books, New York, 1785/1988.

</td>
</tr>


<tr valign="top">
<td align="right" class="bibtexnumber">
[<a name="Bennett2015-ch4to6">31</a>]
</td>
<td class="bibtexitem">
Christopher Bennett.
 <em>What Is This Thing Called Ethics?</em>, chapter 4--6.
 Routledge, London, 2015.
 Chapters on Utilitarianism, Kantian Ethics, and Aristotelian Virtue
  Ethics.

</td>
</tr>


<tr valign="top">
<td align="right" class="bibtexnumber">
[<a name="Ahmed2024">32</a>]
</td>
<td class="bibtexitem">
I.&nbsp;Ahmed, M.&nbsp;Kajol, U.&nbsp;Hasan, P.&nbsp;P. Datta, A.&nbsp;Roy, and M.&nbsp;R. Reza.
 Chatgpt versus bard: A comparative study.
 <em>Engineering Reports</em>, 6(11):e12890, 2024.

</td>
</tr>


<tr valign="top">
<td align="right" class="bibtexnumber">
[<a name="Brown2020">33</a>]
</td>
<td class="bibtexitem">
T.&nbsp;Brown, B.&nbsp;Mann, N.&nbsp;Ryder, M.&nbsp;Subbiah, J.&nbsp;D. Kaplan, P.&nbsp;Dhariwal, and
  D.&nbsp;Amodei.
 Language models are few-shot learners.
 In <em>Advances in neural information processing systems</em>,
  volume&nbsp;33, pages 1877--1901, 2020.

</td>
</tr>


<tr valign="top">
<td align="right" class="bibtexnumber">
[<a name="Thoppilan2022">34</a>]
</td>
<td class="bibtexitem">
R.&nbsp;Thoppilan, D.&nbsp;De&nbsp;Freitas, J.&nbsp;Hall, N.&nbsp;Shazeer, A.&nbsp;Kulshreshtha, H.&nbsp;T. Cheng,
  and Q.&nbsp;Le.
 Lamda: Language models for dialog applications.
 <em>arXiv preprint arXiv:2201.08239</em>, 2022.

</td>
</tr>


<tr valign="top">
<td align="right" class="bibtexnumber">
[<a name="Alkaissi2023">35</a>]
</td>
<td class="bibtexitem">
H.&nbsp;Alkaissi and S.&nbsp;I. McFarlane.
 Artificial hallucinations in chatgpt: implications in scientific
  writing.
 <em>Cureus</em>, 15(2), 2023.

</td>
</tr>


<tr valign="top">
<td align="right" class="bibtexnumber">
[<a name="Chelli2024">36</a>]
</td>
<td class="bibtexitem">
M.&nbsp;Chelli, J.&nbsp;Descamps, V.&nbsp;Lavou&eacute;, C.&nbsp;Trojani, M.&nbsp;Azar, M.&nbsp;Deckert, and
  C.&nbsp;Ruetsch-Chelli.
 Hallucination rates and reference accuracy of chatgpt and bard for
  systematic reviews: comparative analysis.
 <em>Journal of medical Internet research</em>, 26:e53164, 2024.

</td>
</tr>


<tr valign="top">
<td align="right" class="bibtexnumber">
[<a name="Heck2023">37</a>]
</td>
<td class="bibtexitem">
T.&nbsp;G. Heck.
 What artificial intelligence knows about 70 kda heat shock proteins,
  and how we will face this chatgpt era.
 <em>Cell Stress and Chaperones</em>, 28(3):225--229, 2023.

</td>
</tr>


<tr valign="top">
<td align="right" class="bibtexnumber">
[<a name="Athaluri2023">38</a>]
</td>
<td class="bibtexitem">
S.&nbsp;A. Athaluri, S.&nbsp;V. Manthena, V.&nbsp;K.&nbsp;M. Kesapragada, V.&nbsp;Yarlagadda, T.&nbsp;Dave,
  and R.&nbsp;T.&nbsp;S. Duddumpudi.
 Exploring the boundaries of reality: investigating the phenomenon of
  artificial intelligence hallucination in scientific writing through chatgpt
  references.
 <em>Cureus</em>, 15(4), 2023.

</td>
</tr>


<tr valign="top">
<td align="right" class="bibtexnumber">
[<a name="Nasr2023">39</a>]
</td>
<td class="bibtexitem">
M.&nbsp;Nasr, N.&nbsp;Carlini, J.&nbsp;Hayase, M.&nbsp;Jagielski, A.&nbsp;F. Cooper, D.&nbsp;Ippolito, and
  K.&nbsp;Lee.
 Scalable extraction of training data from (production) language
  models.
 <em>arXiv preprint arXiv:2311.17035</em>, 2023.

</td>
</tr>


<tr valign="top">
<td align="right" class="bibtexnumber">
[<a name="Kikalishvili2023">40</a>]
</td>
<td class="bibtexitem">
Shalva Kikalishvili.
 Unlocking the potential of gpt-3 in education: opportunities,
  limitations, and recommendations for effective integration.
 <em>Interactive Learning Environments</em>, 32, 2023.

</td>
</tr>


<tr valign="top">
<td align="right" class="bibtexnumber">
[<a name="Tack2022">41</a>]
</td>
<td class="bibtexitem">
Ana&iuml;s Tack and Chris Piech.
 The ai teacher test: Measuring the pedagogical ability of blender and
  gpt-3 in educational dialogues.
 2022.

</td>
</tr>


<tr valign="top">
<td align="right" class="bibtexnumber">
[<a name="Sharma2023">42</a>]
</td>
<td class="bibtexitem">
Aditi Kavia and Kumari&nbsp;Simran Sharma.
 Chat gpt and copyright: Legal and ethical challenges.
 July 2023.

</td>
</tr>


<tr valign="top">
<td align="right" class="bibtexnumber">
[<a name="Ray2023">43</a>]
</td>
<td class="bibtexitem">
P.&nbsp;P. Ray.
 Chatgpt: a comprehensive review on background, applications, key
  challenges, bias, ethics, limitations and future scope.
 <em>Internet of Things and Cyber-Physical Systems</em>, 3(1):121--154,
  2023.

</td>
</tr>


<tr valign="top">
<td align="right" class="bibtexnumber">
[<a name="Sumbal2024">44</a>]
</td>
<td class="bibtexitem">
Anusha Sumbal, Ramish Sumbal, and A.&nbsp;Amir.
 Can chatgpt-3.5 pass a medical exam? a systematic review of chatgpt's
  performance in academic testing.
 <em>Journal of medical education and curricular development</em>, 11,
  2024.

</td>
</tr>


<tr valign="top">
<td align="right" class="bibtexnumber">
[<a name="Ouyang2022">45</a>]
</td>
<td class="bibtexitem">
L.&nbsp;Ouyang, J.&nbsp;Wu, X.&nbsp;Jiang, D.&nbsp;Almeida, C.L. Wainwright, P.&nbsp;Mishkin, C.&nbsp;Zhang,
  S.&nbsp;Agarwal, K.&nbsp;Slama, A.K. Ray, J.&nbsp;Schulman, J.K. Hilton, F.&nbsp;Kelton, L.P.
  Miller, M.&nbsp;Simens, A.&nbsp;Askell, P.&nbsp;Welinder, P.F. Christiano, J.&nbsp;Leike, and
  R.J. Lowe.
 Training language models to follow instructions with human feedback.
 <em>arXiv (Cornell University)</em>, 2022.

</td>
</tr>


<tr valign="top">
<td align="right" class="bibtexnumber">
[<a name="ACM2018">46</a>]
</td>
<td class="bibtexitem">
Association for Computing Machinery.
 Acm code of ethics and professional conduct, 2018.
[&nbsp;<a href="https://www.acm.org/code-of-ethics">http</a>&nbsp;]

</td>
</tr>
</table><hr><p><em>This file was generated by
<a href="http://www.lri.fr/~filliatr/bibtex2html/">bibtex2html</a> 1.99.</em></p>
</details>
