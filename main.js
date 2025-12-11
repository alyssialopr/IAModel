let session = null;
let imageElement = null;

function setStatus(text) {
  const predDiv = document.getElementById('prediction');
  predDiv.textContent = text;
}

function setResult(text) {
  const resultEl = document.getElementById('result');
  resultEl.innerHTML = 'Résultat : ' + text;
}

async function loadModel() {
  try {
    setStatus('Chargement du modèle ONNX...');
    console.log('Chargement du modèle ONNX...');

    const modelUrl = 'model.onnx';
    session = await ort.InferenceSession.create(modelUrl);

    console.log('Modèle chargé avec succès');
    console.log('Input names :', session.inputNames);
    console.log('Output names :', session.outputNames);

    setStatus('Modèle chargé. Choisissez une image puis cliquez sur "Prédire".');
    document.getElementById('predict-button').disabled = false;
  } catch (error) {
    console.error('Erreur de chargement du modèle:', error);
    setStatus('Erreur : impossible de charger le modèle');
    document.getElementById('predict-button').disabled = true;
  }
}

function initImageUpload() {
  console.log('Init upload');
  const input = document.getElementById('imageUpload');
  const resultDiv = document.getElementById('result');
  const predDiv = document.getElementById('prediction');

  input.addEventListener('change', (event) => {
    console.log('Changement fichier');
    const file = event.target.files[0];
    if (!file) {
      console.log('Aucun fichier');
      return;
    }

    if (!imageElement) {
      imageElement = document.createElement('img');
      imageElement.id = 'preview-image';
      imageElement.style.maxWidth = '300px';
      imageElement.style.display = 'block';
      imageElement.style.marginTop = '10px';
      resultDiv.appendChild(imageElement);
    }

    const reader = new FileReader();
    reader.onload = function (e) {
      console.log('Image chargée pour preview');
      imageElement.src = e.target.result;
      predDiv.textContent = 'Image chargée. Cliquez sur "Prédire".';
    };
    reader.readAsDataURL(file);
  });
}

function preprocessImageFromElement(img) {
  console.log('Prétraitement image');
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');

  const targetSize = 224;
  canvas.width = targetSize;
  canvas.height = targetSize;

  ctx.drawImage(img, 0, 0, targetSize, targetSize);

  const imageData = ctx.getImageData(0, 0, targetSize, targetSize);
  const { data } = imageData;

  const width = targetSize;
  const height = targetSize;
  const float32Data = new Float32Array(1 * 3 * height * width);

  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const pixelIndex = (y * width + x) * 4;
      const r = data[pixelIndex] / 255.0;
      const g = data[pixelIndex + 1] / 255.0;
      const b = data[pixelIndex + 2] / 255.0;

      const rn = (r - mean[0]) / std[0];
      const gn = (g - mean[1]) / std[1];
      const bn = (b - mean[2]) / std[2];

      const idx = y * width + x;
      float32Data[idx] = rn;                    // canal R
      float32Data[height * width + idx] = gn;   // canal G
      float32Data[2 * height * width + idx] = bn; // canal B
    }
  }

  console.log('Prétraitement terminé, taille du tensor :', float32Data.length);
  return float32Data;
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((x) => x / sum);
}

async function predict() {
  console.log('Clique sur Prédire');

  if (!session) {
    console.warn('Session ONNX absente');
    setResult('Modèle non chargé');
    setStatus("Erreur : le modèle n'est pas encore chargé");
    return;
  }

  if (!imageElement || !imageElement.src) {
    console.warn('Aucune image');
    setResult('Aucune image');
    setStatus("Veuillez d'abord choisir une image.");
    return;
  }

  try {
    setStatus('Prédiction en cours...');

    const inputData = preprocessImageFromElement(imageElement);
    const inputTensor = new ort.Tensor('float32', inputData, [1, 3, 224, 224]);
    console.log('Tensor créé');

    const inputName = session.inputNames[0];
    console.log('Input name :', inputName);

    const output = await session.run({ [inputName]: inputTensor });
    console.log('session.run OK, outputs :', Object.keys(output));

    const outputName = session.outputNames[0];
    const outputData = output[outputName].data;
    console.log('Output length :', outputData.length);

    const probs = softmax(Array.from(outputData));
    const maxIndex = probs.indexOf(Math.max(...probs));
    const confidence = (probs[maxIndex] * 100).toFixed(1);

    console.log('Classe prédite :', maxIndex, 'conf :', confidence);
    setResult(`<strong>Classe ${maxIndex}</strong>`);
    setStatus(`Classe prédite : ${maxIndex} avec ${confidence}% de confiance`);
  } catch (error) {
    console.error('Erreur de prédiction:', error);
    setResult('Erreur');
    setStatus('Erreur lors de la prédiction');
  }
}

function clearImage() {
  const input = document.getElementById('imageUpload');
  input.value = '';

  if (imageElement) {
    imageElement.src = '';
  }

  setResult('');
  setStatus('Image effacée. Choisissez une nouvelle image.');
}

window.onload = function () {
  console.log('window.onload');
  initImageUpload();
  loadModel();

  document
    .getElementById('clear-button')
    .addEventListener('click', clearImage);

  document
    .getElementById('predict-button')
    .addEventListener('click', predict);

  document.getElementById('predict-button').disabled = true;
};
