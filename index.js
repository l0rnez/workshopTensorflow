// Use `tfjs`.
const tf = require('@tensorflow/tfjs');
// const tfvis = require('@tensorflow/tfjs-vis');

// Define a simple model with one input layer and one output layer
const model = tf.sequential();
model.add(tf.layers.dense({ units: 4, activation: 'sigmoid', inputShape: [2] }));
model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

// Compile the model with a mean squared error loss function
model.compile({ loss: 'binaryCrossentropy', optimizer: 'adam' });
// model.compile({ loss: 'meanAbsoluteError', optimizer: 'sgd' });

// Generate some training data
function generateData(amount) {
    const data = [];
    for (let i = 0; i < amount; i++) {
      const x1 = Math.round(Math.random());
      const x2 = Math.round(Math.random());
      const y = x1 ^ x2; // XOR operation
      data.push([x1, x2, y]);
    }
    return data;
}

const data = generateData(1000);
const xs = tf.tensor2d(data.map(d => d.slice(0, 2)));
console.log(data.map(d => d.slice(0, 2)));

// Create ys tensor with the correct shape
const ys = tf.tensor2d(data.map(d => d[2]), [data.length, 1]);
console.log(data.map(d => d[2]));


// Train the model
model.fit(xs, ys, { epochs: 1000 }).then(() => {
// Predict the sum of 7 and 8
const tensor = tf.tensor2d([[1, 1]]);
const prediction = model.predict(tensor);

prediction.print();

// Apply threshold
const threshold = 0.5;
const predictedClass = prediction.dataSync()[0] >= threshold ? 1 : 0;

tensor.print();
console.log('Predicted class:', predictedClass);
});

model.summary();