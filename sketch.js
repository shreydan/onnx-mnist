const outputElement = document.querySelector("#prediction");

const modelPath = "./model/mnist.onnx";
const sessionOptions = {
  executionProviders: ["wasm", "webgl"],
};
var inferenceSession;

async function createInferenceSession() {
  try {
    inferenceSession = await ort.InferenceSession.create(
      modelPath,
      sessionOptions
    );
  } catch (e) {
    console.log(`failed to load ONNX model: ${e}.`);
  }
}

function argmax(preds) {
  let max = 0;
  for (let i = 0; i < preds.length; i++) {
    if (preds[i] > preds[max]) max = i;
  }
  return max;
}

async function predict(matrix) {
  try {
    const inputTensor = new ort.Tensor("float32", matrix, [1, 1, 28, 28]);
    const feeds = {};
    feeds[inferenceSession.inputNames[0]] = inputTensor;
    const results = await inferenceSession.run(feeds);
    const preds = results[inferenceSession.outputNames[0]].data;
    outputElement.innerHTML = argmax(preds);
  } catch (e) {
    console.log(`failed to inference ONNX model: ${e}.`);
  }
}

function setup() {
  let canvas = createCanvas(400, 400);
  canvas.parent("#container");
  canvas.mouseReleased(process);
  background(0);
  strokeWeight(25);
  stroke(255);

  const clearBtn = document.querySelector("#clear");
  clearBtn.addEventListener("click", () => {
    background(0);
    outputElement.innerHTML = "_";
  });

  createInferenceSession();
}

function process() {
  let resizedCanvas = document.createElement("canvas");
  resizedCanvas.width = 28;
  resizedCanvas.height = 28;
  let resizedCtx = resizedCanvas.getContext("2d");
  resizedCtx.drawImage(canvas, 0, 0, 28, 28);

  const matrix = new Float32Array(1 * 28 * 28);

  const resizeImageData = resizedCtx.getImageData(
    0,
    0,
    resizedCanvas.width,
    resizedCanvas.height
  );
  const resizePixelData = resizeImageData.data;

  // Copy the pixel data into the matrix
  for (let i = 0; i < 28 * 28; i++) {
    const r = resizePixelData[i * 4] / 255.0;
    const g = resizePixelData[i * 4 + 1] / 255.0;
    const b = resizePixelData[i * 4 + 2] / 255.0;
    matrix[i] = (r + g + b) / 3.0;
  }

  predict(matrix);
}

function draw() {
  if (mouseIsPressed) {
    line(pmouseX, pmouseY, mouseX, mouseY);
  }
}
