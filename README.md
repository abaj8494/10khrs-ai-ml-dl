# 10,000 Hours of Machine Learning.

> "Machine Learning is just lego for adults" - Dr. Kieran Samuel Owens

> "S/he who has a why, can bear almost any how." - Friedrich Nietzche

I've fallen in love with machine learning.
I'm going to spend 10,000 hours of **deliberate practise** on the subject, and construct the appropriate feedback loops.

Here is what I have done / what I intend to do:
- [X] UNSW AI
- [X] UNSW Machine Learning and Data Mining
- [ ] UNSW Deep Learning and Neural Networks
- [ ] UNSW Computer Vision
- [ ] Stanford CS229 (Machine Learning)
- [ ] Stanford CS230 (Deep Learning)
- [ ] Mathematics for Machine Learning, Ong et al.
- [ ] HOML (Hands on Machine Learning)
- [ ] All of Statistics, Larry Wasserman
- [X] Coursera Machine Learning Specialisation
- [X] Coursera Deep Learning Specialisation

# Papers

> "Read 2 papers a week" - Andrew Ng

| Paper | Read | Understood | Implemented | Link |
| ----- | ---- | ---------- | ----------- | ---- |
| AlexNet | 18/09/24| In progress | No | alexnet.pdf |
| Deep Learning Review  | 29/09/24 | Yes | N/A | lecun-bengio-hinto-review.pdf |
| Sequence to Sequence Learning with NN | 30/09/24 | Yes | Yes | No | en-fr-rnn.pdf |
| Attention is all you need | No | No | No | attention.pdf |


# Queen of the hill
Here are all of the best performances I have obtained on classical datasets

| Dataset | Accuracy | Model |
| ------- | -------- | ----- |
| MNIST   | 0%  | KNN   |
| FMNIST  | 0%  | Random Forest |
| KMNIST  | 93% | 2-layer CNN |
| CIFAR   | 91% | CNN |
| IRIS    | 97% | Support Vector Machine |
| ImageNet | 75% | ResNet50 |
| Sentiment140 | 85% | LSTM |
| Boston Housing | 92% | Linear Regression |
| Wine Quality | 88% | Gradient Boosting |
| Pima Indians Diabetes | 82% | Decision Tree |
| IMDB Reviews | 89% | BERT |
| KDD Cup 1999 | 85% | K-Means Clustering |
| Digits | 92% | Gaussian Mixture Model |
| CartPole | 96% | Deep Q-Network |


# Capstone Projects

Here are my non-trivial projects:

## Peg-Solitaire

### Motivations
I grew up as a child with this puzzle in my house. My mother could solve it, and maybe a couple of members on her side of the family.

Mum never knew the algorithm, or any techniques beyond "My hand just knows"; as a result I spent 4 days on it in my youth until solving it.
I learned that the trick is to consider the L shape `___|` and realise that for every set of this 4, you can perform legal operations until you are left with 1 marble.

Then, since there are 32 marbles, you do this 8 times until you have 4 left, and then finally you do it once more to go a single peg in the middle of the board.

```
    O O O      
    O O O      
O O O O O O O  
O O O . O O O  
O O O O O O O  
    O O O      
    O O O   
```
to
```
    · · ·      
    · · ·      
· · · . · · ·  
· · · O · · ·  
· · · · · · ·  
    · · ·      
    · · ·
```

After battling hard for this solution, I find the wikipedia page and associated [article](https://en.wikipedia.org/wiki/Peg_solitaire) only to learn that there are upward of 18,000 distinct solutions.

Anyways, fast-forward slightly, and now I can code so the above directory contains a *DFS* implementation that searches every possible move until it finds a winning configuration:

`s s w w s w w w w s a a s d d a d d d d a`

Here, the letters are the basic `wasd` movements, and the spaces are the execution of that move.

Ultimately the game logic looks something like this: 
`[[3, 3, 's', 10], [2, 3, 'd', 9], [2, 2, 's', 4], [0, 2, 'a', 2], [2, 1, 'a', 1], [2, 2, 'd', 8], [0, 4, 'w', 6], [2, 4, 'a', 12], [1, 2, 'w', 7], [2, 2, 's', 16], [3, 2, 'd', 15], [1, 2, 'w', 3], [1, 4, 'w', 13], [2, 4, 's', 17], [3, 4, 'a', 18], [1, 4, 'w', 11], [3, 2, 'w', 22], [4, 2, 'd', 21], [2, 2, 'w', 27], [3, 2, 's', 20], [3, 4, 'd', 5], [2, 4, 'w', 14], [3, 4, 's', 24], [4, 4, 'a', 25], [4, 5, 'd', 26], [4, 4, 'w', 29], [5, 4, 's', 32], [6, 4, 'd', 31], [4, 4, 'w', 19], [4, 3, 'a', 30], [3, 3, 'w', 23]]`

where the first 2 moves are the coordinates of the peg being moved, the letter is the move and the corresponding number is the 'id' of the marble being _killed_.

### Prospectives
Looking forwards, I want to train a learner to solve this puzzle via reinforcement learning.

## OCR

This was one of the first times I fell in love with Machine Learning, without knowing that the magic came from Machine Learning methods.

I simply had a `pdf` file with handwritten or scanned text, and desperately wanted to find something within the document without having to do it manually.

I google online for an OCR; upload my file; download it back again, and hey presto --- such magical, accurate results!

### LeNet

Truthfully, implementing the LeNet paper that accomplishes this is just an excuse to feel the magic of OCR once more, but this time of my own doing.
Reading the LeNet paper and a _PDF Parser in C_ video uploaded on [YouTube](https://www.youtube.com/watch?v=9NAsrWp9Wto) is inspiring to make a tool for my own learning and terminal utility.

### PDF

Finally, I am a massive `PDF` (Portable Document Format) fanboy; I write everything passionately in \(\TeX{}\) for absolutely no reason whatsoever. I have some of my favourite typeset poems [here](https://github.com/abaj8494/poems) and some lovely high-school math [handouts](https://github.com/abaj8494/handouts) that are the culmination of my \(\TeX{}\) wizardry in the `exam` documentclass.

## Non-descriptive frisbee stats
A computer vision model that takes in streamed games and outputs a player statistic that factors in non-descriptive events --- i.e. giving the correct call at the correct time, or poaching in the lane to force a bad throw.

I expect this to be trained using a transformer and written in Python. It is inspired by [Andrew Wood's](https://github.com/AndyWood91) analytical Ultimate dream.

## Kits19 Grand Challenge
https://kits19.grand-challenge.org/data/

