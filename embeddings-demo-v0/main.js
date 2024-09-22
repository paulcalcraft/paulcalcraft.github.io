// Initialize Web Worker
console.log('Initializing main...');
const worker = new Worker('worker.js');

// Get DOM elements
const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');
const freezeEmbeddingsButton = document.getElementById('freezeEmbeddingsButton');
const freezeDenseButton = document.getElementById('freezeDenseButton');
const lossCtx = document.getElementById('lossChart').getContext('2d');
const embeddingCtx = document.getElementById('embeddingChart').getContext('2d');
const denseWeightCtx = document.getElementById('denseWeightChart').getContext('2d');
const lossDisplay = document.getElementById('lossDisplay');

// Add these state variables at the top of the file
let embeddingsFrozen = false;
let denseFrozen = false;
let isTrainingInProgress = false;

Chart.defaults.font.size = 20;

// Initialize Charts
const lossChart = new Chart(lossCtx, {
  type: 'line',
  data: {
    labels: [],
    datasets: [{
      label: 'Loss',
      data: [],
      borderColor: 'rgba(255,99,132,1)',
      fill: false,
    }]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: {
          font: {
            // // size: 16
          }
        }
      }
    },
    scales: {
      x: { 
        title: { 
          display: true, 
          font: {
            // // size: 16
          }
        } 
      },
      y: { 
        title: { 
          display: true, 
          font: {
            // // size: 16
          }
        } 
      }
    }
  }
});
const embeddingChart = new Chart(embeddingCtx, {
  type: 'scatter',
  data: {
    datasets: [{
      label: 'Embeddings',
      data: [],
      backgroundColor: 'rgba(54, 162, 235, 1)',
      pointRadius: 5,
      pointHoverRadius: 12
    }]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      },
      tooltip: {
        enabled: false
      }
    },
    scales: {
      x: { 
        title: { 
          display: true,
          text: 'Dimension 1'
        } 
      },
      y: { 
        title: { 
          display: true,
          text: 'Dimension 2'
        } 
      }
    },
    onClick: handleChartClick,
    onHover: handleChartHover
  }
});

// Custom plugin to add labels
const customPlugin = {
  id: 'customPlugin',
  afterDatasetsDraw: (chart, args, options) => {
    if (chart === embeddingChart) {
      const ctx = chart.ctx;
      chart.data.datasets[0].data.forEach((datapoint, index) => {
        const meta = chart.getDatasetMeta(0);
        const { x, y } = meta.data[index].getCenterPoint();
        const radius = meta.data[index].options.radius;
        
        ctx.save();
        ctx.textAlign = 'left';
        ctx.textBaseline = 'bottom';
        ctx.fillStyle = 'black';
        ctx.fillText(index.toString(), x + radius + 2, y - radius - 2);
        ctx.restore();
      });
    }
  }
};

// Register the custom plugin
Chart.register(customPlugin);

// Initialize Dense Weight Chart
const denseWeightChart = new Chart(denseWeightCtx, {
  type: 'bar',
  data: {
    labels: ['W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'Bias'],
    datasets: [{
      label: 'Dense Layer Weights',
      data: [0, 0, 0, 0, 0, 0, 0],
      backgroundColor: 'rgba(75, 192, 192, 0.6)',
      borderColor: 'rgba(75, 192, 192, 1)',
      borderWidth: 1
    }]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: {
          font: {
            // // size: 16
          }
        }
      }
    },
    scales: {
      x: { 
        title: { 
          display: true, 
          font: {
            // // size: 16
          }
        } 
      },
      y: { 
        title: { 
          display: true, 
          font: {
            // // size: 16
          }
        },
        beginAtZero: true
      }
    },
    onClick: handleDenseWeightClick,
    onHover: handleDenseWeightHover
  }
});

// Update the chart options for all charts
const commonChartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      labels: {
        font: {
          // size: 16
        }
      }
    }
  },
  scales: {
    x: { 
      title: { 
        display: true, 
        font: {
          // size: 16
        }
      } 
    },
    y: { 
      title: { 
        display: true, 
        font: {
          // size: 16
        }
      } 
    }
  }
};

// Update Loss Chart options
lossChart.options = {
  ...commonChartOptions,
  scales: {
    ...commonChartOptions.scales,
    x: { ...commonChartOptions.scales.x, title: { ...commonChartOptions.scales.x.title, text: 'Epoch' } },
    y: { ...commonChartOptions.scales.y, title: { ...commonChartOptions.scales.y.title, text: 'Loss' }, min: 0 }
  }
};

// Update Embedding Chart options
embeddingChart.options = {
  ...commonChartOptions,
  scales: {
    ...commonChartOptions.scales,
    x: { ...commonChartOptions.scales.x, title: { ...commonChartOptions.scales.x.title, text: 'Dimension 1' } },
    y: { ...commonChartOptions.scales.y, title: { ...commonChartOptions.scales.y.title, text: 'Dimension 2' } }
  },
  plugins: {
    ...commonChartOptions.plugins,
    tooltip: {
      callbacks: {
        label: function(context) {
          return `Index: ${context.dataIndex}`;
        }
      }
    },
    customPlugin: {} // Add this line to enable the custom plugin for the embedding chart
  },
  onClick: handleChartClick,
  onHover: handleChartHover
};

// Update Dense Weight Chart options
denseWeightChart.options = {
  ...commonChartOptions,
  scales: {
    ...commonChartOptions.scales,
    x: { ...commonChartOptions.scales.x, title: { ...commonChartOptions.scales.x.title, text: 'Weight Index' } },
    y: { ...commonChartOptions.scales.y, title: { ...commonChartOptions.scales.y.title, text: 'Weight Value' }, beginAtZero: true }
  },
  onClick: handleDenseWeightClick,
  onHover: handleDenseWeightHover
};

// Update all charts
lossChart.update();
embeddingChart.update();
denseWeightChart.update();

// Handle messages from the worker
worker.onmessage = function(event) {
  const message = event.data;
  
  if (message.type === 'loss') {
    if (message.epoch === 'manual') {
      // Update the last point for manual updates
      lossChart.data.datasets[0].data[lossChart.data.datasets[0].data.length - 1] = message.loss;
    } else {
      // Only add new point if it's a new epoch
      if (lossChart.data.labels.length === 0 || message.epoch > lossChart.data.labels[lossChart.data.labels.length - 1]) {
        lossChart.data.labels.push(message.epoch);
        lossChart.data.datasets[0].data.push(message.loss);
      }
    }
    lossChart.update();
    
    // Update the loss display
    lossDisplay.textContent = `Loss: ${message.loss.toFixed(4)}`;
  }
  
  if (message.type === 'weights') {
    // Update embedding chart if embeddings are present
    if (message.embeddings) {
      const embeddingData = message.embeddings.map((point, index) => ({ x: point[0], y: point[1], index: index }));
      embeddingChart.data.datasets[0].data = embeddingData;
      embeddingChart.data.datasets[0].backgroundColor = message.embeddingsFrozen ? 'rgba(200, 200, 200, 0.6)' : 'rgba(54, 162, 235, 1)';
      embeddingChart.update();
    }

    // Update dense weight chart if dense weights are present
    if (message.denseWeights) {
      const denseWeights = message.denseWeights.flat();
      denseWeightChart.data.datasets[0].data = [...denseWeights, message.denseBias];
      denseWeightChart.data.datasets[0].backgroundColor = message.denseFrozen ? 'rgba(200, 200, 200, 0.6)' : 'rgba(75, 192, 192, 0.6)';
      denseWeightChart.update();
    }

    // Update frozen states if present
    if (message.embeddingsFrozen !== undefined) {
      embeddingsFrozen = message.embeddingsFrozen;
    }
    if (message.denseFrozen !== undefined) {
      denseFrozen = message.denseFrozen;
    }

    // Update button text
    startButton.textContent = isTrainingInProgress ? 'Resume Training' : 'Start Training';
  }
  
  if (message.type === 'training_paused') {
    startButton.disabled = false;
    stopButton.disabled = true;
    isTrainingInProgress = false;
    startButton.textContent = 'Resume Training';
  }

  if (message.type === 'initialized') {
    updateCharts(message);
  }
};

// Start Training
startButton.addEventListener('click', () => {
  worker.postMessage({ command: 'start' });
  startButton.disabled = true;
  stopButton.disabled = false;
  isTrainingInProgress = true;
  startButton.textContent = 'Resume Training';
});

// Stop Training
stopButton.addEventListener('click', () => {
  worker.postMessage({ command: 'stop' });
  startButton.disabled = false;
  stopButton.disabled = true;
  isTrainingInProgress = false;
  startButton.textContent = 'Resume Training';
});

let isDragging = false;
let draggedPointIndex = -1;
let wasTrainingBeforeDrag = false;

embeddingChart.options.onClick = handleChartClick;
embeddingChart.options.onHover = handleChartHover;

function handleChartClick(event, elements) {
  if (elements.length > 0) {
    const index = elements[0].index;
    console.log(`Clicked point index: ${index}`);
  }
}

function handleChartHover(event, elements) {
  if (elements.length > 0) {
    event.native.target.style.cursor = 'pointer';
  } else {
    event.native.target.style.cursor = 'default';
  }
}

embeddingCtx.canvas.addEventListener('mousedown', startDragging);
embeddingCtx.canvas.addEventListener('mousemove', drag);
embeddingCtx.canvas.addEventListener('mouseup', stopDragging);
embeddingCtx.canvas.addEventListener('mouseout', stopDragging);

function startDragging(event) {
  const points = embeddingChart.getElementsAtEventForMode(event, 'nearest', { intersect: true }, true);
  if (points.length > 0) {
    isDragging = true;
    draggedPointIndex = points[0].index;
    wasTrainingBeforeDrag = isTrainingInProgress;
    if (wasTrainingBeforeDrag) {
      worker.postMessage({ command: 'pauseTraining' });
    }
  }
}

function drag(event) {
  if (isDragging && !embeddingsFrozen) {
    const rect = event.target.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const xValue = embeddingChart.scales.x.getValueForPixel(x);
    const yValue = embeddingChart.scales.y.getValueForPixel(y);
    
    embeddingChart.data.datasets[0].data[draggedPointIndex] = {x: xValue, y: yValue};
    embeddingChart.update();
    
    worker.postMessage({
      command: 'updateEmbedding',
      index: draggedPointIndex,
      newPosition: [xValue, yValue]
    });
  }
}

function stopDragging() {
  if (isDragging) {
    isDragging = false;
    draggedPointIndex = -1;
    if (wasTrainingBeforeDrag) {
      worker.postMessage({ command: 'resumeTraining' });
      wasTrainingBeforeDrag = false;
    }
  }
}

// Dense Weight Chart Interaction
let isDraggingDenseWeight = false;
let draggedDenseWeightIndex = -1;

function handleDenseWeightClick(event, elements) {
  if (elements.length > 0) {
    const index = elements[0].index;
    const value = denseWeightChart.data.datasets[0].data[index];
    console.log(`Clicked dense weight index: ${index}, value: ${value}`);
    // Add any additional actions you want to perform when clicking on a dense weight
  }
}

function handleDenseWeightHover(event, elements) {
  if (elements.length > 0) {
    event.native.target.style.cursor = 'pointer';
  } else {
    event.native.target.style.cursor = 'default';
  }
}

denseWeightCtx.canvas.addEventListener('mousedown', startDraggingDenseWeight);
denseWeightCtx.canvas.addEventListener('mousemove', dragDenseWeight);
denseWeightCtx.canvas.addEventListener('mouseup', stopDraggingDenseWeight);
denseWeightCtx.canvas.addEventListener('mouseout', stopDraggingDenseWeight);

function startDraggingDenseWeight(event) {
  const points = denseWeightChart.getElementsAtEventForMode(event, 'nearest', { intersect: true }, true);
  if (points.length > 0) {
    isDraggingDenseWeight = true;
    draggedDenseWeightIndex = points[0].index;
    wasTrainingBeforeDrag = isTrainingInProgress;
    if (wasTrainingBeforeDrag) {
      worker.postMessage({ command: 'pauseTraining' });
    }
  }
}

function dragDenseWeight(event) {
  if (isDraggingDenseWeight && !denseFrozen) {
    const rect = event.target.getBoundingClientRect();
    const y = event.clientY - rect.top;
    const newValue = denseWeightChart.scales.y.getValueForPixel(y);
    
    denseWeightChart.data.datasets[0].data[draggedDenseWeightIndex] = newValue;
    denseWeightChart.update();
    
    worker.postMessage({
      command: 'updateDenseWeight',
      index: draggedDenseWeightIndex,
      newValue: newValue
    });
  }
}

function stopDraggingDenseWeight() {
  if (isDraggingDenseWeight) {
    isDraggingDenseWeight = false;
    draggedDenseWeightIndex = -1;
    if (wasTrainingBeforeDrag) {
      worker.postMessage({ command: 'resumeTraining' });
      wasTrainingBeforeDrag = false;
    }
  }
}

// Freeze/Unfreeze Embeddings
freezeEmbeddingsButton.addEventListener('click', () => {
  embeddingsFrozen = !embeddingsFrozen;
  worker.postMessage({ command: 'freezeEmbeddings', freeze: embeddingsFrozen });
  freezeEmbeddingsButton.textContent = embeddingsFrozen ? 'Unfreeze Embeddings' : 'Freeze Embeddings';
});

// Freeze/Unfreeze Dense Layer
freezeDenseButton.addEventListener('click', () => {
  denseFrozen = !denseFrozen;
  worker.postMessage({ command: 'freezeDense', freeze: denseFrozen });
  freezeDenseButton.textContent = denseFrozen ? 'Unfreeze Dense Layer' : 'Freeze Dense Layer';
});

// Add reset button event listener
const resetButton = document.getElementById('resetButton');
resetButton.addEventListener('click', () => {
  worker.postMessage({ command: 'reset' });
  isTrainingInProgress = false;
  startButton.textContent = 'Start Training';
  startButton.disabled = false;
  stopButton.disabled = true;
  
  // Reset the loss chart
  lossChart.data.labels = [];
  lossChart.data.datasets[0].data = [];
  lossChart.update();
  
  // Reset the loss display
  lossDisplay.textContent = 'Loss: N/A';
});

// Modify the updateCharts function
function updateCharts(data) {
  if (data.embeddings) {
    const embeddingData = data.embeddings.map((point, index) => ({ x: point[0], y: point[1], index: index }));
    embeddingChart.data.datasets[0].data = embeddingData;
    embeddingChart.update();
  }

  if (data.denseWeights) {
    const denseWeights = data.denseWeights.flat();
    denseWeightChart.data.datasets[0].data = [...denseWeights, data.denseBias];
    denseWeightChart.update();
  }

  // Reset loss chart if it's an initialization
  if (data.type === 'initialized') {
    lossChart.data.labels = [];
    lossChart.data.datasets[0].data = [];
    lossChart.update();
  }
}

// Initialize weights and embeddings on page load
worker.postMessage({ command: 'initialize' });
