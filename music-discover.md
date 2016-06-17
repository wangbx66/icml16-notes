# Deep (R-) Learning for Music Discover

This is a quick literature review over the application of deep learning algorithms, either reinforcement or not, on music generation, music classification, music representation, and content-based music recommendation. The contents should cover the following previous studies, as well as the domain studies on music and statistics conducted by Dmitri.

1. Dieleman aka /u/bannane's [nips-13 paper](http://papers.nips.cc/paper/5004-deep-content-based-music-recommendation.pdf) along with the [blogpost](http://benanne.github.io/2014/08/05/spotify-cnns.html) and [detailed reddit response](https://www.reddit.com/r/MachineLearning/comments/46i3f2/pooling_over_one_dimension_replicating_spotifys/)
1. [Juergen Blues Improvisation](http://people.idsia.ch/~juergen/blues/)
1. [Magenta](https://github.com/tensorflow/magenta), another melody composing implementation, conducted by google people, in tensorflow.
1. [Audiveris](https://audiveris.kenai.com/) for optic music recognition
1. [Hexahedria](http://www.hexahedria.com/2015/08/03/composing-music-with-recurrent-neural-networks/) by Daniel Johnson, on the music composing with biaxial LSTM.
1. [Polyphonic Music Generation and Transcription](http://www-etud.iro.umontreal.ca/~boulanni/ICML2012.pdf) paper on ICML-12

## Dieleman's

The work applies a 1-dimensional ConvNet architecture on the audio signals to predict the corresponding feature signals, which is extracted by WMF on user clickthrough history. It claims that although content-based method is not as good as WMF, it can solve the cold-starting problem using the audio information. One thing to note is that it's a rare case where the clickthrough as well the audio is simultaneously available, which happens when the authors were doing their internship at [Spotify](https://erikbern.com/).

Dieleman's has a few [follow-ups](https://scholar.google.com.hk/scholar?start=0&hl=en&as_sdt=5,33&sciodt=0,33&cites=9532659972049857239&scipsc=) with marginal significance, including a [music tagging with ConvNets](http://arxiv.org/abs/1606.00298), a [ConvNets on raw audios](http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=6854950&url=http%3A%2F%2Fieeexplore.ieee.org%2Fxpls%2Fabs_all.jsp%3Farnumber%3D6854950), also [another reinforcement learning approach](http://dl.acm.org/citation.cfm?id=2623372) who claims to deal with explore-exploit tradeoff.

## Juergen's

Blues is a project about music composition, made by an LSTM network. As is explained in the [report](http://people.idsia.ch/~juergen/blues/IDSIA-07-02.pdf), the LSTM tries to perform a next-step prediction, which is pretty similar to what it is in a language model, except that the information provided by a chord is so limited compared to the perplexity carried by a word. It claims that previous attempts at this task failed to capture the long-term dependencies that defines a musical form. The training set are mixed in MP3 and MIDI form and is publically available.

The chord in the report was simply represented by a boolean vector where each entry indicates the existence of one note. The time between consecutive LSTM iteration was discretized to be a slice of real time.

## Magenta

According to a [reddit comment](https://www.reddit.com/r/MachineLearning/comments/4m2o39/magenta_a_new_project_from_the_google_brain_team/d3sfib9), Magenta was like a failed project several years ago. But recently Google decide to give the community a shot by publicizing the Magenta code. The engine behind is a LSTM architecture called [basic_rnn](https://github.com/tensorflow/magenta/tree/master/magenta/models/basic_rnn).

Same as Juergen's, Magenta uses a language model to learn the conditional distribution over notes. One thing to note is that it relies on Tensorflow, and hence Bazel.

## Hexahedria

The first thing it mentions is the note-invariant property of melodies, which means music can be freely transposed up and down. To achieve this, Daniel made a stack of identical RNNs, one for each node, which accept the input within an octave. After two layers of RNNs looping over time, it puts two layers of RNNs looping over notes, in order to catch the spatial relation and perform decent chords.

Some detailed information listed at the end of its [blogpost](http://www.hexahedria.com/2015/08/03/composing-music-with-recurrent-neural-networks/), about input and output format, as well as the music representation. The model outputs a articulate probability along with the play probability. Theano code available on [github](https://github.com/hexahedria/biaxial-rnn-music-composition).

## Polyphonic Transcription

The objective of polyphonic transcription is to determine the underlying notes of a polyphonic audio signal without access to its score. The authors proposed RNN-RBM architecture, which is, an RNN for sequence modeling as usual, plus a generative RBM for transcript generation. 

## The Geometry of Music


