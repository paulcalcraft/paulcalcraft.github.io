console.log('Starting worker...');
importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.18.0/dist/tf.min.js');

let model;
let X, Y;
let training = false;
let stopTraining = false;
let pauseTraining = false;
let epoch = 0;
let embeddingsFrozen = false;
let denseFrozen = false;
let updateInterval = 1; // Start with updating every epoch
const maxUpdateInterval = 10; // Maximum number of epochs between updates

self.onmessage = async function(event) {
  const message = event.data;
  
  if (message.command === 'start' && !training) {
    training = true;
    stopTraining = false;
    pauseTraining = false;
    if (!model) {
      await initializeModel();
    }
    await trainModel();
  }
  
  if (message.command === 'stop' && training) {
    stopTraining = true;
  }

  if (message.command === 'pauseTraining') {
    pauseTraining = true;
  }

  if (message.command === 'resumeTraining') {
    pauseTraining = false;
    if (training) {
      await trainModel();
    }
  }

  if (message.command === 'reset') {
    await initializeModel();
    sendWeightsToMain();
  }

  if (message.command === 'initialize') {
    if (!model) {
      await initializeModel();
    }
    sendWeightsToMain();
  }

  if (message.command === 'updateEmbedding') {
    await updateEmbedding(message.index, message.newPosition);
  }

  if (message.command === 'updateDenseWeight') {
    await updateDenseWeight(message.index, message.newValue);
  }

  if (message.command === 'freezeEmbeddings') {
    embeddingsFrozen = message.freeze;
    updateLayerTrainability();
  }

  if (message.command === 'freezeDense') {
    denseFrozen = message.freeze;
    updateLayerTrainability();
  }
};

function updateLayerTrainability() {
  model.layers[0].trainable = !embeddingsFrozen;
  model.layers[2].trainable = !denseFrozen;
  
  // Recompile the model to apply changes
  model.compile({
    optimizer: tf.train.adam(0.01),
    loss: 'meanSquaredError',
  });
}

async function initializeModel() {
  const vocabSize = 5;
  const embedDim = 2;
  const seqLength = 3;
  const batchSize = 32;

  model = tf.sequential();
  model.add(tf.layers.embedding({inputDim: vocabSize, outputDim: embedDim, inputLength: seqLength}));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({units: 1}));

  model.compile({
    optimizer: tf.train.adam(0.01),
    loss: 'meanSquaredError',
  });

  // Generate fixed dataset
  X = tf.randomUniform([batchSize, seqLength], 0, vocabSize, 'int32');
  Y = X.sum(1).div(seqLength).expandDims(1);

  epoch = 0;
  updateInterval = 1; // Reset update interval
  updateLayerTrainability();
  
  // Send initial weights to main thread
  sendWeightsToMain();
}

async function trainModel() {
  while (!stopTraining) {
    if (pauseTraining) {
      await new Promise(resolve => setTimeout(resolve, 100));
      continue;
    }

    epoch++;
    
    // Train on batch
    const history = await model.fit(X, Y, {
      batchSize: X.shape[0],
      epochs: 1,
      verbose: 0
    });

    // Send updates less frequently as training progresses
    if (epoch % updateInterval === 0) {
      tf.tidy(() => {
        // Send loss to main thread
        const loss = history.history.loss[0];
        self.postMessage({type: 'loss', epoch: epoch, loss: loss});

        // Get embeddings
        const embeddings = model.layers[0].getWeights()[0].arraySync();
        
        // Get dense layer weights and bias
        const denseWeights = model.layers[2].getWeights()[0].arraySync();
        const denseBias = model.layers[2].getWeights()[1].arraySync()[0];
        
        self.postMessage({
          type: 'weights', 
          embeddings: embeddings, 
          denseWeights: denseWeights,
          denseBias: denseBias,
          embeddingsFrozen: embeddingsFrozen,
          denseFrozen: denseFrozen
        });
      });

      // Increase update interval gradually
      if (epoch % 100 === 0 && updateInterval < maxUpdateInterval) {
        updateInterval++;
      }
    }

    // Check for stop or pause signal more frequently
    if (epoch % 5 === 0) {
      await new Promise(resolve => setTimeout(resolve, 0));
      if (stopTraining || pauseTraining) break;
    }
  }

  if (stopTraining) {
    self.postMessage({type: 'training_paused'});
    training = false;
  }
}

async function updateEmbedding(index, newPosition) {
  tf.tidy(() => {
    const embeddings = model.layers[0].getWeights()[0];
    const updatedEmbeddings = embeddings.bufferSync();
    updatedEmbeddings.set(newPosition[0], index, 0);
    updatedEmbeddings.set(newPosition[1], index, 1);
    const newEmbeddingsTensor = updatedEmbeddings.toTensor();
    model.layers[0].setWeights([newEmbeddingsTensor]);

    // Recalculate loss
    const prediction = model.predict(X);
    const mse = tf.losses.meanSquaredError(Y, prediction);
    const loss = mse.dataSync()[0];

    self.postMessage({type: 'loss', epoch: 'manual', loss: loss});

    // Send updated embeddings to main thread
    const updatedEmbeddingsArray = model.layers[0].getWeights()[0].arraySync();
    self.postMessage({
      type: 'weights',
      embeddings: updatedEmbeddingsArray,
      embeddingsFrozen: embeddingsFrozen
    });
  });
}

async function updateDenseWeight(index, newValue) {
  tf.tidy(() => {
    const weights = model.layers[2].getWeights();
    const updatedWeights = weights[0].bufferSync();
    updatedWeights.set(newValue, index, 0);
    const newWeightsTensor = updatedWeights.toTensor();
    model.layers[2].setWeights([newWeightsTensor, weights[1]]);

    // Recalculate loss
    const prediction = model.predict(X);
    const mse = tf.losses.meanSquaredError(Y, prediction);
    const loss = mse.dataSync()[0];

    // Get updated dense weights and bias
    const denseWeights = model.layers[2].getWeights()[0].arraySync();
    const denseBias = model.layers[2].getWeights()[1].arraySync()[0];

    self.postMessage({
      type: 'weights',
      denseWeights: denseWeights,
      denseBias: denseBias,
      denseFrozen: denseFrozen
    });
    self.postMessage({type: 'loss', epoch: 'manual', loss: loss});
  });
}

function sendWeightsToMain() {
  const embeddings = model.layers[0].getWeights()[0].arraySync();
  const denseWeights = model.layers[2].getWeights()[0].arraySync();
  const denseBias = model.layers[2].getWeights()[1].arraySync()[0];

  self.postMessage({
    type: 'initialized',
    embeddings: embeddings,
    denseWeights: denseWeights,
    denseBias: denseBias,
    embeddingsFrozen: embeddingsFrozen,
    denseFrozen: denseFrozen
  });
}
