const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LogisticRegression {
  constructor(features, labels, options) {
    this.features = this.processFeatures(features);
    this.labels = tf.tensor(labels);
    this.costHistory = [];

    this.options = Object.assign(
      { learningRate: 0.1,
        iterations: 1000,
        decisionBoundary: 0.5
      },
      options);

    this.weights = tf.zeros([this.features.shape[1], this.labels.shape[1]]);

  }

  gradientDescent(features, labels) {
    const currentGuessers = features.matMul(this.weights).softmax(); // mx + b
    const differences = currentGuessers.sub(labels); //

    const slopes = features
      .transpose()
      .matMul(differences)
      .div(features.shape[0]);

    return this.weights.sub(slopes.mul(this.options.learningRate));
  }

  train() {
    const batchQuantity = Math.floor(this.features.shape[0] / this.options.batchSize);

    for(let i = 0; i < this.options.iterations; ++i) {
      for(let j = 0; j < batchQuantity; j++) {
        const {batchSize} = this.options;
        const startIndex = j * batchSize;

        this.weights = tf.tidy(() => {
          const featureSlice = this.features.slice([startIndex, 0],
            [batchSize, -1]);
          const labelSlice = this.labels.slice([startIndex, 0],
                  [batchSize, -1]);
          return this.gradientDescent(featureSlice, labelSlice);
        });
      }

      this.recordCost();
      this.updateLearningRate();
    }
  }

  predict(observations) {
    return this.processFeatures(observations).matMul(this.weights).softmax()
    .argMax(1);
  }

  test(testFeatures, testLabels) {
    const predictions = this.predict(testFeatures);
    testLabels = tf.tensor(testLabels).argMax(1);

    const incorrect = predictions
      .notEqual(testLabels)
      .sum()
      .get();

    return (predictions.shape[0] - incorrect) / predictions.shape[0];
  }

  processFeatures(features) {
    features = tf.tensor(features);

    if(this.mean && this.variance) {
      features = features.sub(this.mean).div(this.variance.pow(0.5));
    }
    else {
      features = this.standardize(features);
    }

    features = tf.ones([features.shape[0], 1]).concat(features, 1);
    return features;
  }

  standardize(features) {
    const { mean, variance } = tf.moments(features, 0);

    // prevent divide by zero
    const filler = variance.cast('bool').logicalNot().cast('float32');

    this.mean = mean;
    this.variance = variance.add(filler);

    return features.sub(mean).div(this.variance.pow(0.5));
  }

  recordCost() {
    const cost = tf.tidy(() => {
      const guesses = this.features.matMul(this.weights).sigmoid();

      const termOne = this.labels.transpose().matMul(guesses.add(1e-7).log());

      const termTwo = this.labels
        .mul(-1)
        .add(1)
        .transpose()
        .matMul(
          guesses
            .mul(-1)
            .add(1)
            .add(1e-7) // add small number to avoid log(0)
            .log()
        );

      const cost = termOne.add(termTwo)
        .div(this.features.shape[0])
        .mul(-1)
        .get(0,0);

      return cost;
    });

    this.costHistory.unshift(cost);
  }

  updateLearningRate() {
    if(this.costHistory < 2) return;

    if(this.costHistory[0] > this.costHistory[1]) {
      this.options.learningRate /= 2;
    }
    else {
      this.options.learningRate *= 1.05; // increase 5%
    }
  }

}

module.exports = LogisticRegression;