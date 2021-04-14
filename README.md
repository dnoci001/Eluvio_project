## Eluvio_project
# Goal predict potential amount of upvotes from title of news article.
Serving relevant content to users is essiential to keep them engaged on any platform.
Thompson and UCB sampling can be used to asses the popularity of content.
Recomender systems can be used to choose content that aligns with a users interest.
However, having an a priori understanding of content's merit before serving can limit the users 
exposure to boring or unrelevant content that might drive them away from your platform.
To aid in this I have developed a classification model that will take the title of an 
article as input and classify the article as having < than the median number of upvotes
or >= to the median number of upvotes. While the model is not perfect it could potential
inform the decision to the extent to which content should be served to users. 

# Model
We are to treat the given csv as a very large file, so I have pandas read the csv file in chunks 
and save them in their respective folders (train,valid,test). A custom pytorch dataloader was written
to handle these chunks in the training of our model and tokenization. I used torchtext's tokenizer
to create a word based vocabulary from the training set. Below is the torchviz output for my model.
I use an embedding bag to go from the vocabulary of 120k words to a dimension of 2048. A dense
layer is used to further reduce this to 512. This is fed into a dense residual block of 4 layers 
which then feeds into a final layer to reduce to our two classes.

![](https://github.com/dnoci001/Eluvio_project/blob/main/images/torchviz.png)

# Results
With the median of upvotes being 5 the distinction between the two classes is quiete difficult to
define.

![](https://github.com/dnoci001/Eluvio_project/blob/main/images/upvotes_hist.png)

That being said my simple model is able to assign the correct class ~56% of the time.
Below is the confusion matrix from the test set where 0 corresponds to the class of less than 5 upvotes and 1
corresponds to the class of 5 or more upvotes.

![](https://github.com/dnoci001/Eluvio_project/blob/main/images/confusion_mat.png)

# Conclusions
I have developed a model that can aid in a pipeline used to serve content to users. Given the difficulty of the problem
the performance of the model is decent. Improvement could be accomplished by hyperparameter optimization of potentially
using an RNN (LSTM,GRU) in place of the embedding bag.
