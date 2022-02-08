# Bayesian Statistics 



## Buzzwords and backstory

Bayesian statistics isn't new, it's been around since thomas bayes came up with it back in the 1700s. The formula is pretty bland, it's just some substituting and rearranging. But this shift in perspective gives birth to a whole new way of looking at probability. A whole new philosophy.

__Let me clairfy__:

I believe Statisitcs is about truth. But how do we describe truth? what are the underlying mechanics behind it? 

__The Frequentist__: 

The frequentist believes there exists only one true answer. 

But all our measurements have some error to it. Say you were measuring a mountain, you would say that each measurement you took had some error to it and at the heart of this attitude is the belief that a true answer already exists.
So we would take many many measurements of the moutain and then the one that pops up the ___most frequent___ is the real answer and that is the FREQUENTIST approach.


###### But the Bayesian

The Bayesian thinks a little different, he believes every observation has some informaiton in it and the bayesian attitude is believing that there isn't one true value for your process but in fact a distribution. So your mountain before, a Bayeisna would say that the true value of the height of the mountain is a meaningless idea. Instead, every measurement of the height of the mountain describes some point on the ground to some point near the top of the mountain but they wont be the identical two points every time. so even though every measurement has a different value, each one is an accurate measurement of something we could call the height of the mountain.

### Bayesian Babies

I think I've always been bayesian but never knew it was bayesian until recently. Whenever I was asked question I'd give a range of answers rather than a singluar answer. I always thought, well I dont wanna make a decision but these are your options and these are my opinions, you can choose. My family hated it and always found me unreliable but you know what? A lot of business decisions are taken into account through a bayesian perspective, doing this allows us to uncover and view risk in a different lens and we can start adding probabilities to our answers. 



### Theory

A little theory goes a long way so the structure of the rest of the article will go as follows

- Describing the mechanics of bayesian statistics
- Explaining the essential terms
- An example to demonstrate how bayesian models work by building an example in a PPL (probabalistic programming language) 
    - Pyro
    - PyMC3
    - Tensorflow Probability
    - Stan 
    
    
(I think ill just use pyro) 

## Mechanics

### Bayes Theorem

Bayes Theorem is a result from conditional probability 

- Condional Probability formula: $P(A|B) = \frac{P(A \cap B)}{P(B)}$

I remember when I was first learning this my teacher told me to bend the bar and then but the second thing underneath, safe to say that the intuition behind what was happening really didnt stick and I just ended up remembering like a computer instead of a mathematician but what does this formula really mean? 


The probability of A _GIVEN_ B means the probability of A happening if we know that B has already happened
Diagrams always help

```python
from matplotlib_venn import venn2, venn2_circles, venn2_unweighted
from matplotlib_venn import venn3, venn3_circles
from matplotlib import pyplot as plt
%matplotlib inline

venn2(subsets = (0.65, 0.25, 0.1), set_labels = ('Event A', 'Event B'))
```




    <matplotlib_venn._common.VennDiagram at 0x123fa7e80>




![png](Bayesian Statistics with Python_files/output_3_1.png)


We want to find out the probability of A happening if B has already happened. 

The Probability of B happening is 0.35
The Probability that A will happen after B has already happened is 0.1

So the Probability of A happening _given_ B = $P(A|B) = \frac{P(A \cap B)}{P(B)}$ = $\frac{0.1}{0.35}$ = 0.29

#### How does Bayes theorem come out of this? well

[1] $P(A|B) = \frac{P(A \cap B)}{P(B)}$

but! 

[2] $P(A \cap B) = P(B \cap A)$

and 

[3] $P(B|A) = \frac{P(B \cap A)}{P(A)}$

so rearranging and subbing [3] into [2] and then that into [1] you get the following

### Bayes Theorem

$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$
